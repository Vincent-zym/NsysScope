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
decoding) fall back to the step marker plus the layer segmentation the table
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
    p.add_argument("--chunk-size", type=int, default=None,
                   help="chunked prefill size from the launch command; prefill steps run "
                        "with a full chunk, so this is the step's token count")
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
    avoid_count: Optional[float] = None,
    cv_tiers: Sequence[float] = (0.05, 0.20, 0.40),
) -> Dict[str, Any]:
    """Pick a once-per-forward marker kernel from the timeline itself.

    The hard-coded vocab-embedding marker only exists on the rank that owns the
    embedding, so it is missing on pipeline-parallel middle ranks. Instead, look for
    kernels whose launches are evenly spaced: a kernel that fires exactly once per
    forward step has a near-constant inter-arrival time, while everything else
    (per-layer kernels, bursts) does not.

    A server capture is not a tight benchmark loop: chunk sizes and request arrivals
    make the per-step period jitter, so uniformity alone can reject every candidate.
    Walk `cv_tiers` from strict to loose and, past the strict tier, only accept a
    launch count that several independent kernels agree on.

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
        # A kernel firing once per repeating unit is even *more* regular than a
        # per-forward one, so exclude that launch count explicitly when it is known.
        if avoid_count and abs(len(values) - avoid_count) <= max(1.0, avoid_count * 0.1):
            continue
        candidates.append({"name": name, "count": len(values), "cv": spread})
    if not candidates:
        raise SystemExit(
            "could not auto-detect a once-per-forward marker: no kernel has more than "
            f"{min_steps} launches with positive spacing. Pass --step-marker with an "
            "explicit LIKE pattern."
        )

    best_cv = min(c["cv"] for c in candidates)
    for tier, limit in enumerate(cv_tiers):
        pool = [c for c in candidates if c["cv"] <= limit]
        if not pool:
            continue
        # The step count is the launch count shared by most evenly spaced kernels; a
        # kernel firing twice per step would otherwise be mistaken for the boundary.
        counts: Dict[int, int] = {}
        for item in pool:
            counts[item["count"]] = counts.get(item["count"], 0) + 1
        step_count = max(counts, key=lambda c: (counts[c], c))
        same = sorted(
            (c for c in pool if c["count"] == step_count), key=lambda c: c["cv"],
        )
        # Jitter is only trustworthy when independent kernels land on the same count.
        if tier and len(same) < 3:
            continue
        return {
            "name": same[0]["name"],
            "count": step_count,
            "cv": same[0]["cv"],
            "cross_check": same[1]["name"] if len(same) > 1 else None,
            "agreeing_kernels": len(same),
            "cv_threshold": limit,
            "cv_relaxed": bool(tier),
        }
    raise SystemExit(
        "could not auto-detect a once-per-forward marker: the most evenly spaced "
        f"kernel has cv={best_cv:.3f} (limit {max(cv_tiers):.2f}) and no launch count "
        "is corroborated by 3 independent kernels. Pass --step-marker with an "
        "explicit LIKE pattern."
    )


def local_batch_size_from_eager_kernels(
    cur: sqlite3.Cursor, device: int, step_count: int, tolerance: float = 0.1,
) -> Optional[Dict[str, Any]]:
    """Read the true per-rank batch size off kernels CUDA graphs never capture.

    A graph-captured kernel's grid is baked in at capture time, so it reports the
    graph's padded bucket (e.g. 200 for a real batch of 197 -- see
    ``--cuda-graph-bs``), not the request count that actually ran. Kernels SGLang
    keeps outside the graph (``graphId IS NULL``) are launched fresh every step with
    grid dimensions sized to the real batch, and under DP attention that size is
    already this rank's local share (see ``srt/model_executor/forward_batch_info.py``
    -- per-rank tensors are padded to the local, not the global, token count).

    A single such kernel could still be coincidence, so this only returns a value
    when at least two independently-named kernels agree on both the launch count
    (matching step_count within `tolerance`) and one shared grid dimension.

    gridY/gridZ are absent from some exports (older nsys, or a minimal fixture);
    treat a missing column as a fixed 1 rather than failing the whole lookup.
    """
    columns = {row[1] for row in cur.execute(
        "pragma table_info(CUPTI_ACTIVITY_KIND_KERNEL)"
    )}
    grid_cols = ", ".join(
        f"k.{axis}" if axis in columns else "1" for axis in ("gridX", "gridY", "gridZ")
    )
    rows = cur.execute(
        f"select s.value, {grid_cols} from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName "
        "where k.deviceId = ? and k.graphId is null",
        (device,),
    ).fetchall()
    if not rows or step_count <= 0:
        return None

    per_kernel: Dict[str, Dict[Tuple[int, int, int], int]] = {}
    for name, gx, gy, gz in rows:
        grid = (int(gx), int(gy), int(gz))
        counts = per_kernel.setdefault(str(name), {})
        counts[grid] = counts.get(grid, 0) + 1

    band = max(1, round(step_count * tolerance))
    # For each kernel, keep its dominant grid only if that grid alone launches a
    # number of times close to step_count -- a kernel with mixed grids (e.g. one
    # warmup launch plus step_count steady-state ones) still qualifies on its
    # steady-state population, but a kernel that is not once-per-step at all won't.
    candidates: Dict[str, Tuple[int, int, int]] = {}
    for name, counts in per_kernel.items():
        grid, count = max(counts.items(), key=lambda kv: kv[1])
        if abs(count - step_count) <= band:
            candidates[name] = grid

    if len(candidates) < 2:
        return None

    # A degenerate axis (e.g. gridZ, or gridY on a 1-D-grid kernel) is 1 on almost
    # every kernel by construction -- that near-unanimous "vote" would outweigh the
    # one axis that actually carries the batch by sheer kernel count. The anchor
    # for "this axis carries the batch" is the scheduler's own request<->slot
    # bookkeeping kernels (alloc/assign/cache-loc -- one thread per request, by
    # name): whichever axis they themselves agree on, at whatever value that is,
    # is the batch axis; unrelated kernels merely corroborate by sharing it.
    scheduler_hint = re.compile(
        r"alloc_extend|assign_extend|assign_req_to_token_pool|extend_cache_locs"
    )
    scheduler_grids = {
        name: grid for name, grid in candidates.items() if scheduler_hint.search(name)
    }
    if len(scheduler_grids) < 2:
        return None
    axis = None
    value = None
    for probe_axis in range(3):
        values = {grid[probe_axis] for grid in scheduler_grids.values()}
        if len(values) == 1:
            axis, value = probe_axis, next(iter(values))
            break
    if axis is None:
        return None
    agreeing = [
        name for name, grid in candidates.items() if grid[axis] == value
    ]
    if len(agreeing) < 2:
        return None
    return {
        "batch_size": value,
        "evidence": (
            f"gridX/Y/Z[{axis}]={value} on {len(agreeing)} non-CUDA-graph kernels "
            f"({', '.join(sorted(agreeing))}) with ~{step_count} launches each; "
            "graph-captured kernels only expose the padded --cuda-graph-bs bucket"
        ),
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
        _apply_eager_batch_size(cur, device, info)
        return info

    split = _speculative_grids(rows)
    if split is None:
        # Several gridX populations that do not repeat as a draft/verify pattern: in
        # prefill the marker's gridX follows the per-step token count, which varies
        # with the chunk fill and the number of queued requests. Those launches are
        # still one-per-forward, so keep them as a single plain marker population.
        entries = sorted(e for values in by_grid.values() for e in values)
        info.update(
            speculative=False, target_grid=None, draft_grid=None,
            batch_size=None, speculative_tokens=0,
            marker_grid_note=(
                "gridX populations follow the per-step token count, not draft/verify"
            ),
            target_graph=_dominant_graph(entries), draft_graph=None,
            target_starts=[s for s, _ in entries], draft_starts=[],
        )
        _apply_eager_batch_size(cur, device, info)
        return info

    draft_grid, target_grid = split
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
    # The marker's own gridX split is speculative-token-population arithmetic, not
    # a token count carried by grid dimensions -- see the LM-head GEMM case, where
    # gridX is a tile count, not `bs * gamma`. Kernels outside the CUDA graph give
    # the real batch directly, so prefer that when it is available and corroborated.
    _apply_eager_batch_size(cur, device, info)
    return info


def _apply_eager_batch_size(
    cur: sqlite3.Cursor, device: int, info: Dict[str, Any],
) -> None:
    """Overwrite info's batch_size with the eager-kernel signal when corroborated."""
    step_count = len(info.get("target_starts") or [])
    found = local_batch_size_from_eager_kernels(cur, device, step_count)
    if found is None:
        return
    info["batch_size"] = found["batch_size"]
    info["batch_size_evidence"] = found["evidence"]


def _speculative_grids(
    rows: Sequence[Tuple[int, int, Optional[int]]]
) -> Optional[Tuple[int, int]]:
    """Return (draft_grid, target_grid) when the gridX populations are draft/verify.

    Speculative decoding launches the marker in a fixed per-step pattern: K draft
    forwards then one verify forward, forever. Anything else -- unequal periods, a
    third population, a period without exactly one verify -- is gridX tracking the
    per-step token count instead, which is normal for prefill.
    """
    grids = sorted({int(g) for _, g, _ in rows})
    if len(grids) != 2:
        return None
    draft_grid, target_grid = grids
    seq = [int(g) for _, g, _ in rows]
    verifies = seq.count(target_grid)
    if not verifies or len(seq) % verifies:
        return None
    period = len(seq) // verifies
    first = seq[:period]
    if first.count(target_grid) != 1:
        return None
    if any(seq[i * period:(i + 1) * period] != first for i in range(verifies)):
        return None
    return draft_grid, target_grid


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
    table pipeline already relies on does know where the last layer stops.
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
        # layer segmentation the table pipeline already does. With speculative
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
    """Mean, min, max and sample count of a per-step series.

    The mean, not the median. Each step closes exactly by construction -- 其他 is
    that step's residual, `phase_us - covered` -- but a median is not additive: on a
    jittery capture every row's median comes from a different step and the children
    then overshoot their phase (1.4% on the 0812 prefill trace) even though nothing
    is mis-attributed. The mean is linear, so the reported numbers close as well.
    `min_us` / `max_us` keep the spread, and the gap row keeps the capture holes,
    visible instead.
    """
    clean = [v for v in values if v is not None]
    if not clean:
        return 0.0, 0.0, 0.0, 0
    return st.fmean(clean), min(clean), max(clean), len(clean)


def build_rows(
    steps: List[Dict[str, Any]], info: Dict[str, Any], gap_threshold_us: float,
    gap_hole_count: int,
) -> List[Dict[str, str]]:
    def series(key: str) -> List[float]:
        return [s[key] for s in steps if s.get(key) is not None]

    step_avg, step_min, step_max, step_n = stat(series("period"))
    rows: List[Dict[str, str]] = []

    def add(name: str, kind: str, values: Sequence[float], *, layers: Any = "",
            substeps: Any = "", parent: Optional[float] = None, note: str = "") -> float:
        avg, lo, hi, n = stat(values)
        per_unit = ""
        if isinstance(layers, int) and layers > 0:
            per_unit = f"{avg / layers:.1f}"
        elif kind == "stage" and isinstance(substeps, int) and substeps > 0:
            per_unit = f"{avg / substeps:.1f}"
        rows.append({
            "环节": name,
            "环节类型": kind,
            "层数": str(layers) if layers not in ("", None) else "",
            "子步数": str(substeps) if substeps not in ("", None) else "",
            "单次耗时(us)": per_unit,
            "总耗时(us)": f"{avg:.1f}",
            "占forward步(%)": f"{avg / step_avg * 100:.2f}" if step_avg else "",
            "占父环节(%)": f"{avg / parent * 100:.1f}" if parent else "",
            "样本数": str(n),
            "min_us": f"{lo:.1f}",
            "max_us": f"{hi:.1f}",
            "备注": note,
        })
        return avg

    add("forward step 总计", "total", series("period"),
        note=f"锚点 {info['marker_pattern']}，共 {len(info['target_starts'])} 步，"
             f"取样 {step_n} 步；总耗时列为取样步均值（均值可加，各环节之和严格闭合）"
             + (f"；{info['shard_note']}" if info.get("shard_note") else ""))

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
    target_avg = add(
        "target 主模型", "phase", target_total, layers=total_layers or "",
        note="含 prep draft / prep verify" + ("；" + tail_note if no_graph else ""),
    )
    for name in [n for n in variant_counts if variant_counts[n]]:
        add(f"{name} 层", "variant",
            [s["target_children"].get(name, 0.0) for s in steps],
            layers=variant_counts[name], parent=target_avg)
    add("其他", "other",
        [s["target_children"].get("其他", 0.0)
         + (s.get("prep_draft") or 0.0) + (s.get("prep_verify") or 0.0)
         for s in steps],
        parent=target_avg,
        note="embedding / lm_head / 输出聚合 + prep draft / prep verify"
             "（verify 判定、KV 记账、投机 token 拼接），不随层数变化")

    if info["speculative"]:
        layers = max((s["draft_children"].get("_layers") or 0) for s in steps)
        draft_avg = add("draft 模型", "phase", series("draft"),
                        layers=layers or "", substeps=info["speculative_tokens"] or "",
                        note=tail_note if no_graph else "")
        key = f"draft {layers} 层 forward"
        add(key, "stage", [s["draft_children"].get(key, 0.0) for s in steps],
            layers=layers or "", parent=draft_avg)
        add("其他", "other", [s["draft_children"].get("其他", 0.0) for s in steps],
            substeps=info["speculative_tokens"] or "", parent=draft_avg,
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


def unit_cycle_count(
    cur: sqlite3.Cursor, device: int,
    variant_cores: Dict[str, List[str]], expected_ratio: Dict[str, int],
) -> Optional[float]:
    """How many repeating units the capture contains, from the variant core counts."""
    unit_layers = sum(expected_ratio.values())
    if not variant_cores or unit_layers <= 0:
        return None
    total_cores = 0
    for needles in variant_cores.values():
        for needle in needles:
            total_cores += cur.execute(
                "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k "
                "join StringIds s on s.id = k.shortName "
                "where k.deviceId = ? and s.value like ?",
                (device, f"%{needle}%"),
            ).fetchone()[0]
    return total_cores / unit_layers


def marks_repeating_unit(info: Dict[str, Any], cycles: Optional[float]) -> bool:
    """Is the marker firing once per repeating unit rather than once per forward?

    The uniformity test that picks a marker automatically is happy with a kernel
    that runs once per structural cycle (KDA,KDA,KDA,MLA for example) -- such a
    kernel is *more* regular than a real per-forward one. The variant-ratio check
    does not catch it either, because one cycle reproduces the declared ratio
    exactly, just with a repeat count of 1. The give-away is arithmetic: the number
    of cycles in the capture equals the marker's launch count.
    """
    if not cycles or cycles <= 1:
        return False
    return abs(info["marker_launches"] - cycles) <= max(1.0, cycles * 0.1)


def layer_shard_note(
    cur: sqlite3.Cursor, device: int, gpus: int, layers_per_step: int,
    variant_cores: Optional[Dict[str, List[str]]] = None,
) -> Optional[str]:
    """Warn when this device only carries part of the model's layers (real PP only).

    Point-to-point NCCL traffic (SendRecv) used to be read as the fingerprint of
    pipeline parallelism -- activations handed between devices -- but DCP (decode
    context parallel) also issues SendRecv, once per attention-variant layer, for
    an intra-layer all-to-all that has nothing to do with which ranks hold which
    layers (sglang/srt/layers/dcp/comm.py). The two are distinguished by *what*
    fires alongside the SendRecv: PP's handoff is unrelated to any specific layer
    variant's core kernel, while DCP's SendRecv count matches one variant's core
    kernel count exactly (one all-to-all per layer of that variant, every layer).
    When every SendRecv can be attributed to a variant this way, this is DCP, not
    a real per-rank layer split -- so no warning is warranted here.
    """
    if gpus <= 1 or layers_per_step <= 0:
        return None
    sendrecv = cur.execute(
        "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k "
        "join StringIds s on s.id = k.shortName "
        "where k.deviceId = ? and s.value like '%SendRecv%'",
        (device,),
    ).fetchone()[0]
    if not sendrecv:
        return None
    if variant_cores:
        core_counts = variant_core_counts(cur, device, variant_cores)
        if sendrecv in core_counts.values():
            return None
    return (
        f"层切分：本表为 device {device} 视角，层级行仅覆盖本卡承载的 {layers_per_step} 层"
        f"（共 {gpus} 卡），不是整模型；步耗时本身是整次 forward 的周期"
    )


def variant_core_counts(
    cur: sqlite3.Cursor, device: int, variant_cores: Dict[str, List[str]],
) -> Dict[str, int]:
    """Launch count of every variant's core kernels on one device."""
    counts: Dict[str, int] = {}
    for name, needles in variant_cores.items():
        total = 0
        for needle in needles:
            total += cur.execute(
                "select count(*) from CUPTI_ACTIVITY_KIND_KERNEL k "
                "join StringIds s on s.id = k.shortName "
                "where k.deviceId = ? and s.value like ?",
                (device, f"%{needle}%"),
            ).fetchone()[0]
        counts[name] = total
    return counts


def device_candidates(
    cur: sqlite3.Cursor, requested: Optional[int],
    variant_cores: Dict[str, List[str]], expected_ratio: Dict[str, int],
) -> List[int]:
    """Devices to try, best first.

    Under pipeline parallelism each rank owns a different slice of the layer stack,
    so the variant mix differs per rank: with a KDA/KDA/KDA/MLA pattern one rank can
    hold 9 KDA + 3 MLA while its neighbour holds 8 KDA + 4 MLA. Only a rank whose mix
    reproduces the taxonomy's declared unit can be labelled with that taxonomy, so
    rank the devices by how closely their variant mix matches it instead of blindly
    taking the busiest one.
    """
    rows = cur.execute(
        "select deviceId, count(*) n from CUPTI_ACTIVITY_KIND_KERNEL "
        "group by deviceId order by n desc"
    ).fetchall()
    if not rows:
        raise SystemExit("trace has no CUDA kernels")
    if requested is not None:
        return [requested]
    devices = [(int(d), int(n)) for d, n in rows]
    if not (variant_cores and expected_ratio):
        return [d for d, _ in devices]
    wanted_total = sum(expected_ratio.values()) or 1
    ranked = []
    for index, (device, n) in enumerate(devices):
        counts = variant_core_counts(cur, device, variant_cores)
        total = sum(counts.values())
        if total:
            deviation = sum(
                abs(counts.get(k, 0) / total - v / wanted_total)
                for k, v in expected_ratio.items()
            )
        else:
            deviation = float("inf")
        ranked.append((round(deviation, 3), index, device))
    ranked.sort()
    return [device for _, _, device in ranked]


def analyse_device(
    cur: sqlite3.Cursor, device: int, args: argparse.Namespace,
    variant_cores: Dict[str, List[str]], expected_ratio: Dict[str, int],
    marker_source: str, layer_boundary: str,
) -> Dict[str, Any]:
    """Segment one device's timeline into forward steps and build the table rows."""
    info = detect_markers(cur, device, args.step_marker)
    if args.ignore_cuda_graphs:
        info["target_graph"] = info["draft_graph"] = None
    cycles = unit_cycle_count(cur, device, variant_cores, expected_ratio)
    if marks_repeating_unit(info, cycles):
        # Re-pick, excluding the per-unit launch count, so a per-cycle kernel cannot
        # be reported as the forward step boundary.
        retry = auto_step_marker(cur, device, avoid_count=cycles)
        info = detect_markers(cur, device, retry["name"])
        info["marker_auto_selected"] = dict(retry, reason="previous marker was per-unit")
        if marks_repeating_unit(info, cycles):
            raise SystemExit(
                f"every candidate step marker fires about {cycles:.1f} times, which is "
                "the repeating-unit count, not the forward-step count; a forward "
                "boundary cannot be established from this capture -- pass "
                "--step-marker explicitly"
            )
    steps, gap_holes = segment_steps(
        cur, device, info, args.max_steps, layer_boundary, variant_cores,
        args.gap_threshold_us,
    )
    layers_per_step = max(
        (sum((s["target_children"].get("_counts") or {}).values()) for s in steps),
        default=0,
    )
    info["shard_note"] = layer_shard_note(
        cur, device, device_count(cur), layers_per_step, variant_cores,
    )
    rows = build_rows(steps, info, args.gap_threshold_us, len(gap_holes))

    # Guard against markers that "work" but label the wrong layers -- the failure mode
    # of deriving them from prose, or of pairing a taxonomy with a different capture
    # (prefill kernel names do not appear in a decode trace). The taxonomy already
    # declares how many layers of each variant one unit has, so the detected counts
    # must reproduce that ratio -- for as many full repeats as fit, plus at most one
    # partial leftover (a model whose layer count is not a multiple of the unit
    # length, e.g. 93 layers = 23 x 4-layer units + one trailing MLA layer, which is
    # a real, declared remainder, not a labelling error). expected_ratio's key order
    # is the pattern's variant order (see expected_variant_ratio), so the remainder
    # is checked against every rotation of that order, since a real trailing partial
    # unit can start at any position within the pattern.
    if marker_source == "taxonomy" and expected_ratio:
        detected: Dict[str, int] = {}
        for step in steps:
            for name, count in (step["target_children"].get("_counts") or {}).items():
                detected[name] = max(detected.get(name, 0), count)
        detected = {k: v for k, v in detected.items() if v}
        unit_layers = sum(expected_ratio.values()) or 1
        total_detected = sum(detected.values()) or 0
        repeats = total_detected // unit_layers
        remainder = total_detected - repeats * unit_layers
        full = {k: v * repeats for k, v in expected_ratio.items()}
        order = list(expected_ratio.keys())
        wanted_options = [full] if remainder == 0 else [
            {
                k: full.get(k, 0) + sum(
                    1 for offset in range(remainder)
                    if order[(start + offset) % len(order)] == k
                )
                for k in set(full) | set(order)
            }
            for start in range(len(order))
        ]
        if detected not in wanted_options:
            wanted_repr = (
                full if remainder == 0
                else f"{full} plus a {remainder}-layer trailing remainder"
            )
            raise SystemExit(
                "taxonomy-derived variant markers do not reproduce the declared "
                f"repeating unit: detected {detected}, expected {wanted_repr} "
                f"({repeats}x {expected_ratio}). The markers were parsed from prose "
                "evidence and are unreliable, or the taxonomy belongs to a different "
                "capture than this trace. Pass --variant-marker NAME=SUBSTRING "
                "explicitly instead of publishing a mislabelled table."
            )
    return {
        "device": device, "info": info, "steps": steps, "rows": rows,
        "layers_per_step": layers_per_step, "gap_holes": gap_holes,
    }


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
    candidates = device_candidates(cur, args.device, variant_cores, expected_ratio)
    rejected: List[str] = []
    result: Optional[Dict[str, Any]] = None
    for device in candidates:
        try:
            result = analyse_device(
                cur, device, args, variant_cores, expected_ratio, marker_source,
                layer_boundary,
            )
            break
        except SystemExit as exc:
            rejected.append(f"device {device}: {exc}")
    if result is None:
        raise SystemExit(
            "no device could be segmented into forward steps:\n" + "\n".join(rejected)
        )
    device = result["device"]
    info = result["info"]
    steps = result["steps"]
    rows = result["rows"]
    layers_per_step = result["layers_per_step"]
    gap_holes = result["gap_holes"]

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
            "device_candidates": candidates,
            "device_rejected": rejected or None,
            "gpu_count": device_count(cur),
            "layer_shard_note": info.get("shard_note"),
            "layers_per_step": layers_per_step,
            "step_marker": {
                "pattern": info["marker_pattern"],
                "launches": info["marker_launches"],
                "grid_populations": info["marker_grid_populations"],
                "step_count": len(info["target_starts"]),
            },
            "marker_auto_selected": info.get("marker_auto_selected"),
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
            "chunk_size": args.chunk_size,
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
