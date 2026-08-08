#!/usr/bin/env python3
"""Build ``<prefix>_forward_pipeline_table.csv`` from an nsys SQLite export.

Unlike the six operator tables, this one describes a whole **forward step** (one
output token) rather than one repeating layer unit, so it reads the raw kernel
timeline instead of the origin CSV.

Phase segmentation prefers ``graphId``: on a CUDA-graph decode path the target
forward, the draft forward and the host-side bookkeeping between them land in
distinct graphs (and ``NULL`` for the non-captured bookkeeping). That is model
agnostic -- no kernel-name list is needed to find the phase boundaries.

Captures without CUDA graphs (prefill, eager decode -- with or without speculative
decoding) fall back to the step marker plus the layer segmentation the six-table
pipeline already does: the target forward ends at its last layer, the draft forward
starts at the marker's draft population and ends one layer stride after its last
layer core. Each forward's tail then lands in the following prep phase; see
``draft_boundary_source`` in the manifest.

See references/output-spec.md ("Forward pipeline table") for the column contract,
the closure invariants and the inter-token-gap definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import statistics as st
from typing import Any, Dict, List, Optional, Sequence, Tuple

FIELDS = (
    "环节", "环节类型", "层数", "子步数", "单次耗时(us)", "总耗时(us)",
    "占forward步(%)", "占父环节(%)", "样本数", "min_us", "max_us", "备注",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sqlite", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--prefix", required=True)
    p.add_argument("--device", type=int, default=None,
                   help="deviceId to analyse; default is the busiest device")
    p.add_argument("--step-marker", default="%vocab_parallel_embedding%",
                   help="SQL LIKE pattern for the once-per-forward marker kernel")
    p.add_argument("--variant-core", "--variant-marker", action="append", default=[],
                   dest="variant_marker", metavar="NAME=SUBSTRING",
                   help="kernel substring that identifies a layer variant, e.g. "
                        "KDA=kernel_cutlass_kda_decode_mtp (repeatable). The marker only "
                        "*labels* which variant a layer belongs to -- the reported time is "
                        "the whole layer's wall span, not the marker kernel's duration")
    p.add_argument("--taxonomy", default=None,
                   help="architecture taxonomy JSON from the same job; layer boundary and "
                        "variant markers are read from it when not given explicitly")
    p.add_argument("--layer-boundary", default="attn_res_fused_tma_kernel",
                   help="comma-separated kernel names that start a layer in the target forward")
    p.add_argument("--gap-threshold-us", type=float, default=50.0,
                   help="only holes longer than this count as inter-token gap")
    p.add_argument("--max-steps", type=int, default=40,
                   help="how many consecutive steps to sample for per-layer work")
    p.add_argument("--ignore-cuda-graphs", action="store_true",
                   help="pretend the capture has no graphId, forcing the marker + layer "
                        "segmentation path; useful to cross-check that path against a "
                        "graph-captured trace whose phases are already known")
    p.add_argument("--manifest-out", default=None,
                   help="write the manifest fragment here as JSON")
    return p.parse_args()


def trace_fingerprint(path: str) -> Dict[str, Any]:
    """Identify the analysed trace so a consumer can detect a mismatched package.

    Hashing a 1 MiB head plus the size is enough to tell two captures apart without
    reading a multi-hundred-MB file, and it survives copying the trace around.
    """
    size = os.path.getsize(path)
    with open(path, "rb") as fh:
        head = fh.read(1024 * 1024)
    return {
        "path": os.path.abspath(path),
        "size_bytes": size,
        "sha256_head_1mib": hashlib.sha256(head).hexdigest(),
    }


def device_count(cur: sqlite3.Cursor) -> int:
    """How many GPUs this capture covers, i.e. the node's visible device count.

    The batch size derived from the marker's gridX is per rank; multiplying by this
    gives the whole capture's batch. Counted from the trace instead of assumed so a
    2-GPU or 8-GPU capture both report honestly.
    """
    row = cur.execute(
        "select count(distinct deviceId) from CUPTI_ACTIVITY_KIND_KERNEL"
    ).fetchone()
    return int(row[0]) if row and row[0] else 0


def pick_device(cur: sqlite3.Cursor, requested: Optional[int]) -> int:
    if requested is not None:
        return requested
    row = cur.execute(
        "select deviceId, count(*) n from CUPTI_ACTIVITY_KIND_KERNEL "
        "group by deviceId order by n desc limit 1"
    ).fetchone()
    if not row:
        raise SystemExit("trace has no CUDA kernels")
    return int(row[0])


def union_busy(intervals: Sequence[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    ordered = sorted(intervals)
    total = 0
    cs, ce = ordered[0]
    for s, e in ordered[1:]:
        if s > ce:
            total += ce - cs
            cs, ce = s, e
        else:
            ce = max(ce, e)
    return total + ce - cs


def holes(intervals: Sequence[Tuple[int, int]], a: int, b: int) -> List[Tuple[int, int]]:
    """Idle sub-intervals of [a, b) not covered by any kernel."""
    clipped = sorted(
        (max(s, a), min(e, b)) for s, e in intervals if e > a and s < b
    )
    if not clipped:
        return [(a, b)]
    out: List[Tuple[int, int]] = []
    if clipped[0][0] > a:
        out.append((a, clipped[0][0]))
    ce = clipped[0][1]
    for s, e in clipped[1:]:
        if s > ce:
            out.append((ce, s))
        ce = max(ce, e)
    if ce < b:
        out.append((ce, b))
    return out


def auto_step_marker(
    cur: sqlite3.Cursor, device: int, min_steps: int = 3,
) -> Dict[str, Any]:
    """Pick a once-per-forward marker kernel from the timeline itself.

    The hard-coded vocab-embedding marker only exists on the rank that owns the
    embedding, so it is missing on pipeline-parallel middle ranks. Instead, look for
    kernels whose launches are evenly spaced: a kernel that fires exactly once per
    forward step has a near-constant inter-arrival time, while everything else
    (per-layer kernels, bursts) does not.

    Returns the chosen marker plus a second, independent candidate for cross-checking.
    """
    rows = cur.execute(
        "select s.value, k.start from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName where k.deviceId = ? order by k.start",
        (device,),
    ).fetchall()
    starts: Dict[str, List[int]] = {}
    for name, start in rows:
        starts.setdefault(str(name), []).append(int(start))

    candidates: List[Dict[str, Any]] = []
    for name, values in starts.items():
        if not (min_steps <= len(values) <= 20000):
            continue
        gaps = [b - a for a, b in zip(values, values[1:])]
        if not gaps or min(gaps) <= 0:
            continue
        mean = sum(gaps) / len(gaps)
        spread = (sum((g - mean) ** 2 for g in gaps) / len(gaps)) ** 0.5 / mean
        if spread > 0.05:  # not evenly spaced -> not once per step
            continue
        candidates.append({"name": name, "count": len(values), "cv": spread})
    if not candidates:
        raise SystemExit(
            "could not auto-detect a once-per-forward marker: no kernel has evenly "
            "spaced launches. Pass --step-marker with an explicit LIKE pattern."
        )

    # The step count is the launch count shared by most evenly spaced kernels; a
    # kernel firing twice per step would otherwise be mistaken for the boundary.
    counts: Dict[int, int] = {}
    for item in candidates:
        counts[item["count"]] = counts.get(item["count"], 0) + 1
    step_count = max(counts, key=lambda c: (counts[c], c))
    same = sorted(
        (c for c in candidates if c["count"] == step_count), key=lambda c: c["cv"],
    )
    return {
        "name": same[0]["name"],
        "count": step_count,
        "cv": same[0]["cv"],
        "cross_check": same[1]["name"] if len(same) > 1 else None,
        "agreeing_kernels": len(same),
    }


def detect_markers(cur: sqlite3.Cursor, device: int, pattern: str) -> Dict[str, Any]:
    """Locate the once-per-forward marker and split it into draft vs target.

    The marker's gridX scales with the token count of the forward it belongs to, so
    two distinct gridX populations mean speculative decoding: the smaller one is the
    draft (batch * draft_block), the larger the target verify (batch * (draft_block+1)).
    """
    rows = cur.execute(
        "select k.start, k.gridX, k.graphId from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName "
        "where k.deviceId = ? and s.value like ? order by k.start",
        (device, pattern),
    ).fetchall()
    auto: Optional[Dict[str, Any]] = None
    if not rows:
        # The default marker (vocab embedding) does not exist on pipeline-parallel
        # middle ranks, so derive one from the timeline instead of giving up.
        auto = auto_step_marker(cur, device)
        pattern = auto["name"]
        rows = cur.execute(
            "select k.start, k.gridX, k.graphId from CUPTI_ACTIVITY_KIND_KERNEL k "
            "join StringIds s on s.id = k.shortName "
            "where k.deviceId = ? and s.value = ? order by k.start",
            (device, pattern),
        ).fetchall()
    if not rows:
        raise SystemExit(f"no kernel matches the step marker pattern {pattern!r}")

    by_grid: Dict[int, List[Tuple[int, Optional[int]]]] = {}
    for start, grid, graph in rows:
        by_grid.setdefault(int(grid), []).append((int(start), graph))
    grids = sorted(by_grid)
    info: Dict[str, Any] = {
        "marker_pattern": pattern,
        "marker_launches": len(rows),
        "marker_grid_populations": {str(g): len(by_grid[g]) for g in grids},
        "marker_auto_selected": auto,
    }

    if len(grids) == 1:
        info.update(
            speculative=False, target_grid=grids[0], draft_grid=None,
            batch_size=None, speculative_tokens=0,
            target_graph=_dominant_graph(by_grid[grids[0]]), draft_graph=None,
            target_starts=[s for s, _ in by_grid[grids[0]]], draft_starts=[],
        )
        return info
    if len(grids) != 2:
        raise SystemExit(
            f"step marker has {len(grids)} gridX populations ({grids}); expected 1 or 2"
        )

    draft_grid, target_grid = grids
    batch = target_grid - draft_grid
    if batch <= 0 or draft_grid % batch:
        batch = None
    info.update(
        speculative=True, target_grid=target_grid, draft_grid=draft_grid,
        batch_size=batch,
        speculative_tokens=(draft_grid // batch) if batch else None,
        batch_size_evidence=(
            "target_gridX - draft_gridX (batch * (K+1) - batch * K)" if batch else None
        ),
        target_graph=_dominant_graph(by_grid[target_grid]),
        draft_graph=_dominant_graph(by_grid[draft_grid]),
        target_starts=[s for s, _ in by_grid[target_grid]],
        draft_starts=[s for s, _ in by_grid[draft_grid]],
    )
    return info


def _dominant_graph(entries: Sequence[Tuple[int, Optional[int]]]) -> Optional[int]:
    counts: Dict[Optional[int], int] = {}
    for _, graph in entries:
        counts[graph] = counts.get(graph, 0) + 1
    return max(counts, key=lambda k: counts[k]) if counts else None


def last_layer_end(
    rows: Sequence[Tuple[Any, ...]], window_start: int, window_end: int,
    layer_boundary: str, variant_cores: Dict[str, List[str]],
) -> Optional[int]:
    """End of the last attention-core-bearing layer block inside a window.

    This is the no-CUDA-graph substitute for "where does the forward end": prefill
    and eager decode have no graphId to key on, but the layer segmentation that the
    six-table pipeline already relies on does know where the last layer stops.
    Everything after it is inter-step bookkeeping.
    """
    if not variant_cores:
        return None
    inside = [r for r in rows if window_start <= int(r[0]) < window_end]
    needles = [n for n in layer_boundary.split(",") if n]
    bounds = sorted(int(r[0]) for r in inside if any(n in r[2] for n in needles))
    if not bounds:
        return None
    edges = bounds + [window_end]
    end: Optional[int] = None
    for lo, hi in zip(edges, edges[1:]):
        block = [r for r in inside if lo <= int(r[0]) < hi]
        if not block:
            continue
        if any(
            n in r[2] for r in block
            for needle_list in variant_cores.values() for n in needle_list
        ):
            end = max(int(r[1]) for r in block)
    return end


def draft_forward_end(
    rows: Sequence[Tuple[Any, ...]], draft_start: int, window_end: int,
) -> Optional[int]:
    """End of the draft model's layer forward without a graphId to key on.

    Same rule ``draft_children`` uses inside the graph-delimited draft phase: the
    widest kernel that repeats once per draft layer is the layer core, and the
    forward ends one median stride after its last occurrence. Whatever the draft
    does afterwards (lm_head, the sampling loop) therefore lands in prep verify
    rather than in the draft phase -- see ``draft_boundary_source`` in the manifest.
    """
    inside = [r for r in rows if draft_start <= int(r[0]) < window_end]
    by_name: Dict[str, List[Tuple[int, int]]] = {}
    for s, e, name, _ in inside:
        by_name.setdefault(name, []).append((int(s), int(e)))
    repeats = {n: v for n, v in by_name.items() if len(v) > 1}
    if not repeats:
        return None
    core = max(repeats, key=lambda n: sum(e - s for s, e in repeats[n]))
    occurrences = sorted(repeats[core])
    strides = [b[0] - a[0] for a, b in zip(occurrences, occurrences[1:])]
    stride = st.median(strides) if strides else 0
    return min(occurrences[-1][0] + stride, window_end)


def segment_steps(
    cur: sqlite3.Cursor, device: int, info: Dict[str, Any], max_steps: int,
    layer_boundary: str, variant_cores: Dict[str, str], gap_threshold_us: float,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Cut each sampled step into the four contiguous phases plus their children."""
    starts = info["target_starts"]
    if len(starts) < 2:
        raise SystemExit("need at least two complete forward steps")
    draft_starts = info["draft_starts"]
    target_graph, draft_graph = info["target_graph"], info["draft_graph"]
    graph_mode = target_graph is not None
    if not graph_mode:
        # prefill and eager decode are not CUDA-graph captured, so fall back to the
        # layer segmentation the six-table pipeline already does. With speculative
        # decoding on top of that, the step marker's own draft population supplies the
        # draft phase start and the draft layer stride supplies its end.
        if not variant_cores:
            raise SystemExit(
                "this capture has no CUDA graphs, so the target forward's end has to "
                "come from layer segmentation -- pass --variant-marker NAME=SUBSTRING "
                "or a --taxonomy so layers can be identified"
            )

    # Sample from the middle of the capture: the first steps often include warmup.
    total = len(starts) - 1
    begin = max(0, min(total // 4, total - max_steps))
    window = list(range(begin, min(begin + max_steps, total)))

    steps: List[Dict[str, Any]] = []
    gap_holes: List[Dict[str, Any]] = []
    for idx in window:
        a, a2 = starts[idx], starts[idx + 1]
        rows = cur.execute(
            "select k.start, k.end, s.value, k.graphId "
            "from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id = k.shortName "
            "where k.deviceId = ? and k.end > ? and k.start < ? order by k.start",
            (device, a, a2),
        ).fetchall()
        if not rows:
            continue
        intervals = [(int(s), int(e)) for s, e, _, _ in rows]

        step: Dict[str, Any] = {"period": (a2 - a) / 1e3}
        draft_start: Optional[int] = None
        draft_end: Optional[int] = None
        if info["speculative"]:
            nxt = [s for s in draft_starts if a <= s < a2]
            if not nxt:
                continue
            if graph_mode:
                draft_kernels = [r for r in rows if r[3] == draft_graph]
                if not draft_kernels:
                    continue
                draft_start = min(int(r[0]) for r in draft_kernels)
                draft_end = max(int(r[1]) for r in draft_kernels)
            else:
                # The draft population of the step marker fires once at the head of
                # the draft forward, so it is the phase start; its prologue folds into
                # prep draft the same way the target's does.
                draft_start = nxt[0]
                draft_end = draft_forward_end(rows, draft_start, a2)
                if draft_end is None:
                    continue

        if graph_mode:
            target_kernels = [r for r in rows if r[3] == target_graph]
            if not target_kernels:
                continue
            target_end = max(int(r[1]) for r in target_kernels)
        else:
            # Stop looking for target layers where the draft forward begins, otherwise
            # the draft's own layers would be mistaken for the tail of the target.
            derived = last_layer_end(
                rows, a, draft_start if draft_start is not None else a2,
                layer_boundary, variant_cores,
            )
            if derived is None:
                continue
            target_end = derived

        if info["speculative"]:
            assert draft_start is not None and draft_end is not None
            if not (a <= target_end <= draft_start <= draft_end <= a2):
                continue  # boundaries out of order: skip rather than publish
            step.update(
                target=(target_end - a) / 1e3,
                prep_draft=(draft_start - target_end) / 1e3,
                draft=(draft_end - draft_start) / 1e3,
                prep_verify=(a2 - draft_end) / 1e3,
            )
            prep_windows = [(target_end, draft_start), (draft_end, a2)]
            step["draft_children"] = draft_children(
                rows, draft_start, draft_end, draft_graph if graph_mode else None,
            )
        else:
            step.update(
                target=(target_end - a) / 1e3,
                prep_draft=(a2 - target_end) / 1e3,
                draft=None, prep_verify=None,
            )
            prep_windows = [(target_end, a2)]

        step["target_children"] = target_children(
            rows, a, target_end, layer_boundary, variant_cores,
        )
        gap = 0.0
        for lo, hi in prep_windows:
            for hs, he in holes(intervals, lo, hi):
                length = (he - hs) / 1e3
                if length > gap_threshold_us:
                    gap += length
                    gap_holes.append({
                        "step_index": idx, "start_ns": hs, "end_ns": he,
                        "length_us": round(length, 3),
                    })
        step["gap"] = gap
        steps.append(step)
    if not steps:
        raise SystemExit("no step could be segmented; check the marker and graph ids")
    return steps, gap_holes


def target_children(
    rows: Sequence[Tuple[Any, ...]], phase_start: int, phase_end: int,
    layer_boundary: str, variant_cores: Dict[str, str],
) -> Dict[str, float]:
    """Wall time per layer variant inside the target forward, plus the remainder.

    Layers are cut at ``layer_boundary`` and labelled by which attention core they
    contain. Whatever the variants do not cover -- embedding, lm_head, output
    all-gather, the forward tail -- becomes 其他, which by design does not scale with
    layer count.
    """
    phase_us = (phase_end - phase_start) / 1e3
    if not variant_cores:
        return {"其他": phase_us}
    inside = [r for r in rows if phase_start <= int(r[0]) < phase_end]
    needles = [n for n in layer_boundary.split(",") if n]
    bounds = sorted(int(r[0]) for r in inside if any(n in r[2] for n in needles))
    # The first layer often opens with a different norm kernel than the rest, so its
    # core sits before the first boundary. Anchor an extra edge at the phase start so
    # that layer is still counted; its prologue folds into it.
    if bounds and any(
        needle in r[2] for r in inside if int(r[0]) < bounds[0]
        for needles in variant_cores.values() for needle in needles
    ):
        bounds.insert(0, phase_start)
    per_variant: Dict[str, float] = {name: 0.0 for name in variant_cores}
    counts: Dict[str, int] = {name: 0 for name in variant_cores}
    if bounds:
        edges = bounds + [phase_end]
        blocks: List[Tuple[int, int, Optional[str]]] = []
        for lo, hi in zip(edges, edges[1:]):
            block = [r for r in inside if lo <= int(r[0]) < hi]
            if not block:
                continue
            label = None
            for name, needles in variant_cores.items():
                if any(n in r[2] for r in block for n in needles):
                    label = name
                    break
            blocks.append((lo, max(int(r[1]) for r in block), label))

        # The boundary kernel can fire more than once per layer (for example once
        # before attention and once before the MLP), so a block without an attention
        # core is not a layer of its own -- it is part of the layer whose core comes
        # next. Buffer such blocks and flush them into that layer. Core-less blocks
        # after the last core are the forward tail and stay in 其他.
        pending_start: Optional[int] = None
        for lo, hi, label in blocks:
            if pending_start is None:
                pending_start = lo
            if label is None:
                continue
            per_variant[label] += (hi - pending_start) / 1e3
            counts[label] += 1
            pending_start = None
    covered = sum(per_variant.values())
    out = {name: value for name, value in per_variant.items() if counts[name]}
    out["_counts"] = counts  # type: ignore[assignment]
    out["其他"] = max(phase_us - covered, 0.0)
    return out


def draft_children(
    rows: Sequence[Tuple[Any, ...]], phase_start: int, phase_end: int,
    draft_graph: Optional[int],
) -> Dict[str, float]:
    """Split the draft phase into its layer forward and everything else.

    The layer forward ends one layer stride after the last attention core, which is
    the widest kernel that repeats once per draft layer. Using the stride rather than
    the core's own end keeps the last layer's post-attention work inside the forward
    instead of leaking it into 其他.
    """
    phase_us = (phase_end - phase_start) / 1e3
    inside = [r for r in rows if phase_start <= int(r[0]) < phase_end
              and (draft_graph is None or r[3] == draft_graph)]
    if not inside:
        return {"其他": phase_us, "_layers": 0}
    by_name: Dict[str, List[Tuple[int, int]]] = {}
    for s, e, name, _ in inside:
        by_name.setdefault(name, []).append((int(s), int(e)))
    repeats = {n: v for n, v in by_name.items() if len(v) > 1}
    if not repeats:
        return {"其他": phase_us, "_layers": 0}
    core = max(repeats, key=lambda n: sum(e - s for s, e in repeats[n]))
    occurrences = sorted(repeats[core])
    strides = [b[0] - a[0] for a, b in zip(occurrences, occurrences[1:])]
    stride = st.median(strides) if strides else 0
    forward_end = min(occurrences[-1][0] + stride, phase_end)
    forward_us = (forward_end - phase_start) / 1e3
    return {
        f"draft {len(occurrences)} 层 forward": forward_us,
        "其他": max(phase_us - forward_us, 0.0),
        "_layers": len(occurrences),
        "_core": core,
    }


def stat(values: Sequence[float]) -> Tuple[float, float, float, int]:
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0, 0.0, 0.0, 0
    return st.median(clean), min(clean), max(clean), len(clean)


def build_rows(
    steps: List[Dict[str, Any]], info: Dict[str, Any], gap_threshold_us: float,
    gap_hole_count: int,
) -> List[Dict[str, str]]:
    def series(key: str) -> List[float]:
        return [s[key] for s in steps if s.get(key) is not None]

    step_med, step_min, step_max, step_n = stat(series("period"))
    rows: List[Dict[str, str]] = []

    def add(name: str, kind: str, values: Sequence[float], *, layers: Any = "",
            substeps: Any = "", parent: Optional[float] = None, note: str = "") -> float:
        med, lo, hi, n = stat(values)
        per_unit = ""
        if isinstance(layers, int) and layers > 0:
            per_unit = f"{med / layers:.1f}"
        elif kind == "stage" and isinstance(substeps, int) and substeps > 0:
            per_unit = f"{med / substeps:.1f}"
        rows.append({
            "环节": name,
            "环节类型": kind,
            "层数": str(layers) if layers not in ("", None) else "",
            "子步数": str(substeps) if substeps not in ("", None) else "",
            "单次耗时(us)": per_unit,
            "总耗时(us)": f"{med:.1f}",
            "占forward步(%)": f"{med / step_med * 100:.2f}" if step_med else "",
            "占父环节(%)": f"{med / parent * 100:.1f}" if parent else "",
            "样本数": str(n),
            "min_us": f"{lo:.1f}",
            "max_us": f"{hi:.1f}",
            "备注": note,
        })
        return med

    add("forward step 总计", "total", series("period"),
        note=f"锚点 {info['marker_pattern']}，共 {len(info['target_starts'])} 步，"
             f"取样 {step_n} 步")

    variant_counts: Dict[str, int] = {}
    for s in steps:
        for name, count in (s["target_children"].get("_counts") or {}).items():
            variant_counts[name] = max(variant_counts.get(name, 0), count)
    total_layers = sum(variant_counts.values())
    # Without graphId a forward's boundary is its last layer, so the tail after the
    # last layer (lm_head, output aggregation, the sampling loop) cannot be told from
    # the bookkeeping that follows it and is reported in the next prep phase.
    no_graph = info["target_graph"] is None
    tail_note = "无 CUDA graph：以最后一层结束为界，forward 尾部计入其他"
    # prep draft / prep verify are the target/verify path's own host-side bookkeeping and
    # do not scale with layer count, which is exactly what 其他 collects -- so they are
    # folded into it rather than reported as separate environments. A step then splits
    # into two top-level phases, target and draft.
    target_total = [
        s["target"] + (s.get("prep_draft") or 0.0) + (s.get("prep_verify") or 0.0)
        for s in steps
    ]
    target_med = add(
        "target 主模型", "phase", target_total, layers=total_layers or "",
        note="含 prep draft / prep verify" + ("；" + tail_note if no_graph else ""),
    )
    for name in [n for n in variant_counts if variant_counts[n]]:
        add(f"{name} 层", "variant",
            [s["target_children"].get(name, 0.0) for s in steps],
            layers=variant_counts[name], parent=target_med)
    add("其他", "other",
        [s["target_children"].get("其他", 0.0)
         + (s.get("prep_draft") or 0.0) + (s.get("prep_verify") or 0.0)
         for s in steps],
        parent=target_med,
        note="embedding / lm_head / 输出聚合 + prep draft / prep verify"
             "（verify 判定、KV 记账、投机 token 拼接），不随层数变化")

    if info["speculative"]:
        layers = max((s["draft_children"].get("_layers") or 0) for s in steps)
        draft_med = add("draft 模型", "phase", series("draft"),
                        layers=layers or "", substeps=info["speculative_tokens"] or "",
                        note=tail_note if no_graph else "")
        key = f"draft {layers} 层 forward"
        add(key, "stage", [s["draft_children"].get(key, 0.0) for s in steps],
            layers=layers or "", parent=draft_med)
        add("其他", "other", [s["draft_children"].get("其他", 0.0) for s in steps],
            substeps=info["speculative_tokens"] or "", parent=draft_med,
            note="draft 前导 / lm_head / 投机采样循环")

    # Spell the measurement windows out: prep draft / prep verify are folded into 其他,
    # so "prep 段" would not correspond to any row a reader can find in the table.
    windows = (
        "target forward 结束→draft 开始、draft 结束→下一步开始 两段"
        if info["speculative"] else "target forward 结束→下一步开始"
    )
    add("token 间间隙", "gap", series("gap"),
        note=f"仅统计 {windows}内 >{gap_threshold_us:g}us 的空洞，"
             f"命中 {gap_hole_count} 处；不参与求和")
    return rows


NEGATION_HINTS = (
    "absence", "absent", "not present", "without", "no ", "never", "excluded",
    "无", "不含", "没有", "缺少",
)


def kernel_tokens(text: str) -> List[str]:
    """Pull kernel-name-looking tokens out of a prose evidence string.

    Taxonomy evidence is written for humans ("sglang::attn_res_fused_tma_kernel
    (AttnResidual aggregation 1, ...)", "trace motif _causal_conv1d_fwd_kernel x3 +
    l2norm_fwd_kernel x2 + chunk_kda_fwd_kernel_*"), so the names have to be
    recovered rather than read from a field.

    Sentences phrased as a *negative* discriminator ("absence of fmhaSm103aKernel_*
    in the layer span") must be dropped whole: taking the kernel name out of them
    turns "this variant never runs X" into "this variant is identified by X".
    """
    lowered = (text or "").lower()
    if any(hint in lowered for hint in NEGATION_HINTS):
        return []
    out: List[str] = []
    for raw in re.findall(r"[A-Za-z_][A-Za-z0-9_:]*", text or ""):
        token = raw.split("::")[-1].rstrip("_")
        low = token.lower()
        if "kernel" in low or low.endswith(("_fwd", "_bwd")):
            if len(token) > 4 and token not in out:
                out.append(token)
    return out


def expected_variant_ratio(tax: Dict[str, Any]) -> Dict[str, int]:
    """Layer count per variant declared by the taxonomy's repeating-unit pattern."""
    unit = tax.get("repeating_unit") or {}
    counts: Dict[str, int] = {}
    for position in unit.get("positions") or []:
        name = position.get("unit_variant")
        if name:
            counts[name] = counts.get(name, 0) + 1
    if counts:
        return counts
    for token in str(unit.get("pattern") or "").split(","):
        name = token.strip()
        if name:
            counts[name] = counts.get(name, 0) + 1
    return counts


def markers_from_taxonomy(
    path: str,
) -> Tuple[List[str], Dict[str, List[str]], Dict[str, int]]:
    """Read the layer boundary and per-variant markers from a job's taxonomy.

    Prefers the machine-readable fields (``layer_start_kernel``,
    ``variants[].trace_marker_kernels``) and falls back to parsing the prose
    evidence, so packages produced before those fields existed still work.
    """
    with open(path, encoding="utf-8") as fh:
        tax = json.load(fh)
    unit = tax.get("repeating_unit") or {}
    evidence = unit.get("boundary_evidence") or {}
    boundaries: List[str] = []
    explicit = evidence.get("layer_start_kernel")
    if explicit:
        boundaries = [explicit] if isinstance(explicit, str) else list(explicit)
    else:
        boundaries = kernel_tokens(evidence.get("layer_start_marker", ""))[:1]

    variants: Dict[str, List[str]] = {}
    for variant in tax.get("variants") or []:
        name = variant.get("name")
        if not name:
            continue
        explicit_markers = variant.get("trace_marker_kernels")
        if explicit_markers:
            variants[name] = list(explicit_markers)
            continue
        tokens: List[str] = []
        for item in variant.get("discriminators") or []:
            tokens += kernel_tokens(str(item))
        if tokens:
            variants[name] = tokens
    return boundaries, variants, expected_variant_ratio(tax)


def main() -> None:
    args = parse_args()
    variant_cores: Dict[str, List[str]] = {}
    for item in args.variant_marker:
        if "=" not in item:
            raise SystemExit(f"--variant-marker needs NAME=SUBSTRING, got {item!r}")
        name, needle = item.split("=", 1)
        variant_cores.setdefault(name.strip(), []).append(needle.strip())

    layer_boundary = args.layer_boundary
    marker_source = "cli"
    expected_ratio: Dict[str, int] = {}
    if args.taxonomy and os.path.exists(args.taxonomy):
        boundaries, tax_variants, expected_ratio = markers_from_taxonomy(args.taxonomy)
        if not variant_cores and tax_variants:
            variant_cores = tax_variants
            marker_source = "taxonomy"
        if boundaries:
            layer_boundary = ",".join(boundaries)

    con = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    cur = con.cursor()
    device = pick_device(cur, args.device)
    info = detect_markers(cur, device, args.step_marker)
    if args.ignore_cuda_graphs:
        info["target_graph"] = info["draft_graph"] = None
    steps, gap_holes = segment_steps(
        cur, device, info, args.max_steps, layer_boundary, variant_cores,
        args.gap_threshold_us,
    )
    rows = build_rows(steps, info, args.gap_threshold_us, len(gap_holes))

    # Guard against markers that "work" but label the wrong layers -- the failure mode
    # of deriving them from prose, or of pairing a taxonomy with a different capture
    # (prefill kernel names do not appear in a decode trace). The taxonomy already
    # declares how many layers of each variant one unit has, so the detected counts
    # must reproduce that ratio.
    if marker_source == "taxonomy" and expected_ratio:
        detected: Dict[str, int] = {}
        for step in steps:
            for name, count in (step["target_children"].get("_counts") or {}).items():
                detected[name] = max(detected.get(name, 0), count)
        detected = {k: v for k, v in detected.items() if v}
        unit_layers = sum(expected_ratio.values()) or 1
        repeats = max(1, round((sum(detected.values()) or 0) / unit_layers))
        wanted = {k: v * repeats for k, v in expected_ratio.items()}
        if detected != wanted:
            raise SystemExit(
                "taxonomy-derived variant markers do not reproduce the declared "
                f"repeating unit: detected {detected}, expected {wanted} "
                f"({repeats}x {expected_ratio}). The markers were parsed from prose "
                "evidence and are unreliable, or the taxonomy belongs to a different "
                "capture than this trace. Pass --variant-marker NAME=SUBSTRING "
                "explicitly instead of publishing a mislabelled table."
            )

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir, f"{args.prefix}_forward_pipeline_table.csv"
    )
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    print(out_path)

    manifest = {
        "forward_pipeline": {
            "trace": trace_fingerprint(args.sqlite),
            "device": device,
            "gpu_count": device_count(cur),
            "step_marker": {
                "pattern": info["marker_pattern"],
                "launches": info["marker_launches"],
                "grid_populations": info["marker_grid_populations"],
                "step_count": len(info["target_starts"]),
            },
            "phase_discriminator": (
                "CUPTI graphId" if info["target_graph"] is not None
                else "marker + layer segmentation"
            ),
            "target_graph": info["target_graph"],
            "draft_graph": info["draft_graph"],
            "draft_boundary_source": (
                None if not info["speculative"]
                else "CUPTI graphId" if info["target_graph"] is not None
                else "draft step marker + draft layer stride "
                     "(draft lm_head/sampling land in prep verify)"
            ),
            "speculative": info["speculative"],
            "speculative_tokens": info["speculative_tokens"],
            "batch_size": info.get("batch_size"),
            "batch_size_evidence": info.get("batch_size_evidence"),
            "sampled_steps": len(steps),
            # folded into the target phase's 其他 row, so keep the split here where a
            # consumer can still report it without re-reading the trace
            "prep_draft_us": round(stat([s["prep_draft"] for s in steps
                                         if s.get("prep_draft") is not None])[0], 1),
            "prep_verify_us": round(stat([s["prep_verify"] for s in steps
                                          if s.get("prep_verify") is not None])[0], 1),
            "gap_threshold_us": args.gap_threshold_us,
            "gap_holes": gap_holes,
            "variant_markers": variant_cores,
            "variant_marker_source": marker_source,
            "layer_boundary": layer_boundary,
            "taxonomy": args.taxonomy,
        }
    }
    if args.manifest_out:
        with open(args.manifest_out, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, ensure_ascii=False, indent=2)
            fh.write("\n")
        print(args.manifest_out)
    else:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
