#!/usr/bin/env python3
"""Deterministic fusion-candidate prescan for the operator fusion advisor.

Reads a completed ``analysis.json`` from ``sglang-nsys-static-analysis`` plus the
fusion pattern registry, and emits ``candidates.json``: a source-backed,
LLM-free list of fusion candidates scoped to one repeating unit.

Design notes
------------
The upstream community skill (``llm-torch-profiler-analysis``) matches registry
keywords against the *whole* trace with no timeline or stream check. We scope
matching to a single ``(unitPosition, unitId, unitVariant)`` triple and add
timeline adjacency plus stream equality, because an nsys package gives us
per-operator ``startNs``/``stream``/``mbu`` that a torch-profiler trace does not.

This script never proposes a fusion on its own. It only reports:
  * which registry patterns are already active in this unit (``direct``)
  * which registry patterns appear in split form (``split``) -- the real targets
  * adjacent auxiliary/elementwise clusters that no registry row explains
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_REGISTRY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "references",
    "fusion-registry.json",
)

# Ported from the community skill's CATEGORY_PATTERNS. Order matters: the first
# matching family wins. These are prioritisation labels only -- they never
# override the package's own ``category`` field.
KERNEL_FAMILY_PATTERNS: Tuple[Tuple[str, Tuple[str, ...]], ...] = (
    ("hybrid_linear", ("gdn", "gated_delta", "mamba", "selective_scan", "ssd", "causal_conv", "ssm")),
    ("attention", ("flash_attn", "flashattention", "flash_attention", "fmha", "attention", "mla",
                   "paged_attention", "decode_attention")),
    ("moe", ("fused_moe", "grouped_mm", "groupgemm", "group_gemm", "moe", "expert", "groupproblemshape")),
    ("gemm", ("gemm", "gemv", "matmul", "cublas", "cutlass", "wgmma", "mma", "bmm", "nvjet")),
    ("norm", ("rmsnorm", "layernorm", "_norm_", " norm", "normkernel")),
    ("rope", ("rotary", "rope", "mrope")),
    ("softmax", ("softmax",)),
    ("activation", ("silu", "gelu", "relu", "act_and_mul", "sigmoid")),
    ("reduce_topk", ("topk", "reduce", "argmax", "argtopk", "sampling", "multinomial")),
    ("sampling_io", ("prepare_inputs", "write_req_to", "catarraybatched", "prepare_next", "copy_next")),
    ("elementwise", ("elementwise", "vectorized_elementwise_kernel", "unrolled_elementwise_kernel",
                     "gpu_kernel_impl", "binary_internal", "unaryfunctor", "add_kernel", "sub_kernel",
                     "mul_kernel", "div_", "floor_kernel", "log_kernel", "neg_kernel")),
)

# A kernel whose name matches one of these is a raw framework elementwise kernel,
# i.e. it went through no fusion path at all. These are the highest-value fusion
# targets and the reason this prescan exists.
NATIVE_ELEMENTWISE_KEYWORDS: Tuple[str, ...] = (
    "vectorized_elementwise_kernel",
    "unrolled_elementwise_kernel",
    "elementwise_kernel",
    "gpu_kernel_impl",
    "binary_internal",
    "unaryfunctor",
    "add_kernel",
    "sub_kernel",
    "mul_kernel",
    "div_kernel",
    "floor_kernel",
    "log_kernel",
    "neg_kernel",
    "copy_kernel",
    "fill_kernel",
    "cattarraybatchedcopy",
    "catarraybatchedcopy",
)

# Signals that a kernel is already a fused implementation; never re-suggest it.
ALREADY_FUSED_KEYWORDS: Tuple[str, ...] = ("fused", "_fuse_", "and_mul", "combo_kernel")
ALREADY_FUSED_PROSE: Tuple[str, ...] = ("融合", "fused")

VERDICT_BY_STATUS: Dict[str, str] = {
    "mainline direct": "已有实现已生效",
    "mainline split": "已有实现未启用",
    "upstream direct": "上游实现已生效",
    "upstream split": "上游已实现待迁移",
    "inflight direct": "在飞PR已生效",
    "inflight split": "在飞PR待跟进",
}
UNMATCHED_VERDICT = "疑似新机会"
NEEDS_REVIEW_VERDICT = "需人工确认"


@dataclass
class Operator:
    index: int
    kernel_name: str
    full_name: str
    category: str
    module: str
    stage: str
    duration_us: float
    start_ns: int
    end_ns: int
    stream: Optional[int]
    mbu: Optional[float]
    mfu: Optional[float]
    python_function: str
    introduction: str
    unit_position: Optional[int]
    unit_id: Optional[str]
    unit_variant: Optional[str]

    @property
    def kernel_identity(self) -> str:
        """Kernel name only.

        Registry keywords are kernel-name oriented. Matching them against
        ``pythonFunction``/``module`` too makes almost every pattern fire, because
        dispatch paths mention ``mla``/``quant``/``cache`` everywhere.
        """
        return f"{self.kernel_name} {self.full_name}".lower()

    @property
    def dispatch_identity(self) -> str:
        """Python dispatch chain, used only for path-shaped keywords."""
        return f"{self.python_function} {self.module}".lower()

    @property
    def kernel_family(self) -> str:
        lowered = f"{self.kernel_name} {self.full_name}".lower()
        for family, keywords in KERNEL_FAMILY_PATTERNS:
            if any(k in lowered for k in keywords):
                return family
        return "other"

    @property
    def native_elementwise(self) -> bool:
        lowered = f"{self.kernel_name} {self.full_name}".lower()
        return any(k in lowered for k in NATIVE_ELEMENTWISE_KEYWORDS)

    @property
    def already_fused(self) -> bool:
        lowered = f"{self.kernel_name} {self.full_name}".lower()
        if any(k in lowered for k in ALREADY_FUSED_KEYWORDS):
            return True
        prose = self.introduction.lower()
        return any(k in prose for k in ALREADY_FUSED_PROSE) and "不可拆分" in self.introduction


def load_operators(analysis: dict) -> List[Operator]:
    out: List[Operator] = []
    for raw in analysis.get("operators", []) or []:
        out.append(
            Operator(
                index=int(raw.get("index", -1)),
                kernel_name=str(raw.get("kernelName") or raw.get("name") or ""),
                full_name=str(raw.get("fullName") or ""),
                category=str(raw.get("category") or ""),
                module=str(raw.get("module") or ""),
                stage=str(raw.get("stage") or ""),
                duration_us=float(raw.get("durationUs") or 0.0),
                start_ns=int(raw.get("startNs") or 0),
                end_ns=int(raw.get("endNs") or 0),
                stream=raw.get("stream"),
                mbu=raw.get("mbu"),
                mfu=raw.get("mfu"),
                python_function=str(raw.get("pythonFunction") or ""),
                introduction=str(raw.get("introduction") or ""),
                unit_position=raw.get("unitPosition"),
                unit_id=raw.get("unitId"),
                unit_variant=raw.get("unitVariant"),
            )
        )
    return out


UnitKey = Tuple[Optional[int], Optional[str], Optional[str]]


def unit_key(op: Operator) -> UnitKey:
    return (op.unit_position, op.unit_id, op.unit_variant)


def pick_unit(ops: Sequence[Operator], requested: UnitKey) -> UnitKey:
    """Pick the target repeating unit.

    If the caller pinned any component of the key, honour it. Otherwise choose
    the unit with the largest summed duration, which is the one worth optimising.
    """
    groups: Dict[UnitKey, float] = {}
    for op in ops:
        groups[unit_key(op)] = groups.get(unit_key(op), 0.0) + op.duration_us
    if not groups:
        raise SystemExit("analysis.json has no operators")

    def matches(key: UnitKey) -> bool:
        return all(want is None or want == have for want, have in zip(requested, key))

    eligible = [k for k in groups if matches(k)]
    if not eligible:
        raise SystemExit(f"no repeating unit matches {requested!r}")
    return max(eligible, key=lambda k: groups[k])


def resolve_candidate_path(path: str, source_root: Optional[str]) -> dict:
    """Resolve a registry path hint against the supplied source tree.

    Registry paths are recorded against an upstream checkout and drift over time
    (files move between ``srt/layers`` and ``kernels/ops``). Resolving by
    basename keeps the registry usable without hand-maintaining a rename map.
    """
    bare = path.split("::", 1)[0]
    symbol = path.split("::", 1)[1] if "::" in path else None
    record = {"hint": path, "symbol": symbol, "resolved": None, "exists": False}
    if not source_root:
        return record
    direct = os.path.join(source_root, bare)
    if os.path.exists(direct):
        record.update(resolved=os.path.relpath(direct, source_root), exists=True)
        return record
    target = os.path.basename(bare)
    for root, dirs, files in os.walk(source_root):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__"]
        if target in files:
            record.update(resolved=os.path.relpath(os.path.join(root, target), source_root),
                          exists=True)
            return record
    return record


def match_keywords(ops: Sequence[Operator], keywords: Iterable[str]) -> List[Operator]:
    """Match registry keywords against kernel identity.

    A keyword that looks like a source path (``layernorm.py``, ``moe/router.py``)
    is matched against the Python dispatch chain instead, since it can never
    appear in a CUDA kernel name.
    """
    kernel_kws = []
    path_kws = []
    for raw in keywords:
        if not raw:
            continue
        kw = raw.lower()
        (path_kws if (".py" in kw or "/" in kw) else kernel_kws).append(kw)
    if not kernel_kws and not path_kws:
        return []
    hits = []
    for op in ops:
        if kernel_kws and any(k in op.kernel_identity for k in kernel_kws):
            hits.append(op)
        elif path_kws and any(k in op.dispatch_identity for k in path_kws):
            hits.append(op)
    return hits


def best_witness_set(
    groups: Sequence[Sequence[Operator]],
    rank: Dict[int, int],
    window_ops: int,
    same_stream: bool,
) -> Optional[List[Operator]]:
    """Pick one operator per split group so the whole set is timeline-local.

    Without this, "some kernel in the unit matches group A and some other matches
    group B" fires on nearly every pattern. Locality is measured in *operator
    positions* rather than microseconds, because kernel durations vary by orders
    of magnitude between prefill and decode captures and an absolute microsecond
    window would not transfer across runs.
    """
    if not groups or any(not g for g in groups):
        return None
    best: Optional[List[Operator]] = None
    best_spread = None
    for anchor in groups[0]:
        chosen = [anchor]
        for group in groups[1:]:
            nearest = min(group, key=lambda op: abs(rank[op.index] - rank[anchor.index]))
            chosen.append(nearest)
        if len({op.index for op in chosen}) != len(chosen):
            continue
        if same_stream and len({op.stream for op in chosen}) > 1:
            continue
        ranks = [rank[op.index] for op in chosen]
        spread = max(ranks) - min(ranks)
        if best_spread is None or spread < best_spread:
            best_spread, best = spread, sorted(chosen, key=lambda op: op.start_ns)
    if best is None or best_spread > window_ops:
        return None
    return best


def match_pattern(
    spec: dict,
    ops: Sequence[Operator],
    unit_total_us: float,
    model_hint: str,
    tp_size: int,
    source_root: Optional[str],
    rank: Dict[int, int],
    window_ops: int,
    same_stream: bool,
) -> Optional[dict]:
    """Decide whether one registry pattern is direct/split within this unit."""
    includes = [s.lower() for s in spec.get("modelInclude", [])]
    excludes = [s.lower() for s in spec.get("modelExclude", [])]
    hint = model_hint.lower()
    if includes and not any(s in hint for s in includes):
        return None
    if excludes and any(s in hint for s in excludes):
        return None
    if spec.get("requireTp") and tp_size < int(spec.get("minTpSize", 1)):
        return None

    active = match_keywords(ops, spec.get("activeKeywords", []))
    if active:
        related_ops = sorted(active, key=lambda o: o.start_ns)
        family_ops = related_ops
        witness_span_us = None
        witness_spread_ops = None
    else:
        groups = [match_keywords(ops, g) for g in spec.get("splitGroups", [])]
        witnesses = best_witness_set(groups, rank, window_ops, same_stream)
        if not witnesses:
            return None
        # A witness that is itself a fused kernel means the fusion is present in
        # some form; do not report the pattern as missing on that evidence.
        if any(op.already_fused for op in witnesses):
            return None
        related_ops = witnesses
        # The witness set proves the pattern is real and local. The share gate is
        # evaluated over every operator the pattern's groups matched, which is what
        # the upstream min_share / likely_share thresholds were calibrated against.
        seen: Dict[int, Operator] = {}
        for group in groups:
            for op in group:
                seen[op.index] = op
        family_ops = sorted(seen.values(), key=lambda o: o.start_ns)
        witness_span_us = round(
            (related_ops[-1].end_ns - related_ops[0].start_ns) / 1000.0, 3
        )
        witness_spread_ops = max(rank[o.index] for o in witnesses) - min(
            rank[o.index] for o in witnesses
        )

    related_us = sum(o.duration_us for o in related_ops)
    family_us = sum(o.duration_us for o in family_ops)
    if related_us <= 0:
        return None
    share_pct = related_us / unit_total_us * 100.0 if unit_total_us > 0 else 0.0
    family_share_pct = family_us / unit_total_us * 100.0 if unit_total_us > 0 else 0.0
    has_active = bool(active)
    if not has_active and family_share_pct < float(spec.get("minSharePct", 0.25)):
        return None

    status = f"{spec.get('origin', 'mainline')} {'direct' if has_active else 'split'}"
    paths = [resolve_candidate_path(p, source_root) for p in spec.get("candidatePaths", [])]
    any_resolved = any(p["exists"] for p in paths)

    if not has_active and not any_resolved and source_root:
        # The registry claims an implementation exists but we cannot see it in the
        # supplied tree. Do not present that as an actionable "just enable it".
        verdict = NEEDS_REVIEW_VERDICT
        confidence = "low"
    else:
        verdict = VERDICT_BY_STATUS.get(status, NEEDS_REVIEW_VERDICT)
        if family_share_pct >= float(spec.get("likelySharePct", 3.0)) and any_resolved:
            confidence = "high"
        elif any_resolved:
            confidence = "medium"
        else:
            confidence = "low"

    return {
        "registryPattern": spec["pattern"],
        "registryPatternZh": spec.get("patternZh", spec["pattern"]),
        "registryId": spec["id"],
        "origin": spec.get("origin", "mainline"),
        "status": status,
        "verdict": verdict,
        "confidence": confidence,
        "priority": int(spec.get("priority", 0)),
        "subsumes": list(spec.get("subsumes", [])),
        "relatedUs": round(related_us, 3),
        "relatedSharePctOfUnit": round(share_pct, 3),
        "familyUs": round(family_us, 3),
        "familySharePctOfUnit": round(family_share_pct, 3),
        "familyOperators": [o.index for o in family_ops],
        "witnessSpanUs": witness_span_us,
        "witnessSpreadOps": witness_spread_ops,
        "minSharePct": spec.get("minSharePct", 0.25),
        "likelySharePct": spec.get("likelySharePct", 3.0),
        "targetOperators": [o.index for o in related_ops],
        "targetOperatorNames": [o.kernel_name for o in related_ops],
        "candidatePaths": paths,
        "rationaleHint": spec.get("rationaleHint", ""),
        "isActionable": not has_active,
    }


def dedupe_by_subsumption(matches: List[dict]) -> List[dict]:
    """Drop patterns that a higher-priority kept pattern declares it subsumes."""
    kept: List[dict] = []
    suppressed: List[dict] = []
    for match in sorted(matches, key=lambda m: (-m["priority"], m["registryPattern"])):
        if any(match["registryPattern"] in k["subsumes"] for k in kept):
            match["suppressedBy"] = next(
                k["registryPattern"] for k in kept if match["registryPattern"] in k["subsumes"]
            )
            suppressed.append(match)
            continue
        kept.append(match)
    return kept, suppressed


def build_clusters(ops: Sequence[Operator], max_gap_us: float) -> List[List[Operator]]:
    """Group timeline-adjacent auxiliary kernels on the same stream.

    A cluster needs same stream and a launch gap under ``max_gap_us``; a core or
    communication kernel in between always breaks the run. This is the check the
    upstream community skill does not perform.
    """
    clusters: List[List[Operator]] = []
    current: List[Operator] = []
    previous: Optional[Operator] = None
    for op in sorted(ops, key=lambda o: o.start_ns):
        breaks = False
        if op.category != "auxiliary":
            breaks = True
        elif previous is not None:
            gap_us = (op.start_ns - previous.end_ns) / 1000.0
            if op.stream != previous.stream or gap_us > max_gap_us:
                breaks = True
        if breaks:
            if len(current) >= 2:
                clusters.append(current)
            current = [] if op.category != "auxiliary" else [op]
        else:
            current.append(op)
        previous = op if op.category == "auxiliary" else None
    if len(current) >= 2:
        clusters.append(current)
    return clusters


def describe_cluster(cluster: Sequence[Operator], unit_total_us: float,
                     covered: set, min_share_pct: float) -> Optional[dict]:
    duration = sum(o.duration_us for o in cluster)
    share = duration / unit_total_us * 100.0 if unit_total_us > 0 else 0.0
    indices = [o.index for o in cluster]
    native = [o for o in cluster if o.native_elementwise]
    fused = [o for o in cluster if o.already_fused]
    # A cluster that a registry row already explains is not a "new" opportunity.
    explained = bool(set(indices) & covered)
    if share < min_share_pct and not native:
        return None
    if len(fused) == len(cluster):
        return None
    return {
        "clusterId": f"c{indices[0]}",
        "targetOperators": indices,
        "targetOperatorNames": [o.kernel_name for o in cluster],
        "kernelFamilies": [o.kernel_family for o in cluster],
        "stream": cluster[0].stream,
        "spanUs": round((cluster[-1].end_ns - cluster[0].start_ns) / 1000.0, 3),
        "durationUs": round(duration, 3),
        "sharePctOfUnit": round(share, 3),
        "nativeElementwiseCount": len(native),
        "alreadyFusedCount": len(fused),
        "maxGapUs": round(
            max(
                ((b.start_ns - a.end_ns) / 1000.0)
                for a, b in zip(cluster, cluster[1:])
            )
            if len(cluster) > 1
            else 0.0,
            3,
        ),
        "mbuValues": [o.mbu for o in cluster],
        "pythonFunctions": sorted({o.python_function for o in cluster if o.python_function}),
        "modules": sorted({o.module for o in cluster if o.module}),
        "explainedByRegistry": explained,
        "verdict": None if explained else UNMATCHED_VERDICT,
    }


LIMITS = [
    "本扫描只反映本次 trace 实际发生的 kernel 序列，不能证明两个算子在数据依赖上合法可融合。",
    "相邻且同 stream 只是融合候选的必要线索，不是充分条件；融合前仍需确认数据依赖与 in-place 合法性。",
    "registry 命中基于 kernelName / pythonFunction 关键词匹配，可能存在同名不同实现的误判。",
    "candidatePaths 的 resolved 结果来自对所给源码树的按文件名解析，符号级存在性未校验。",
    "缺失的融合有可能是有意禁用（例如 CUDA graph 下刻意关闭的分流），不要一律当作缺陷。",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", required=True, help="path to analysis.json")
    parser.add_argument("--registry", default=DEFAULT_REGISTRY)
    parser.add_argument("--source", default=None, help="model source tree root")
    parser.add_argument("--unit-position", type=int, default=None)
    parser.add_argument("--unit-id", default=None)
    parser.add_argument("--unit-variant", default=None)
    parser.add_argument("--max-gap-us", type=float, default=2.0,
                        help="max launch gap inside one adjacency cluster")
    parser.add_argument("--min-cluster-share", type=float, default=0.5,
                        help="min %% of unit duration for a cluster with no native elementwise")
    parser.add_argument("--pattern-window-ops", type=int, default=6,
                        help="max operator-position spread across a split pattern's "
                             "witness kernels (scale-free substitute for a us window)")
    parser.add_argument("--pattern-any-stream", action="store_true",
                        help="allow a split pattern's witnesses to sit on different streams")
    parser.add_argument("--include-origins", default="mainline,inflight",
                        help="comma-separated registry origins to evaluate "
                             "(mainline,inflight,upstream); upstream rows are "
                             "cross-framework and noisy, so they are off by default")
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--out", default=None, help="write candidates.json here")
    args = parser.parse_args(argv)

    with open(args.analysis, encoding="utf-8") as fh:
        analysis = json.load(fh)
    with open(args.registry, encoding="utf-8") as fh:
        registry = json.load(fh)

    all_ops = load_operators(analysis)
    target = pick_unit(all_ops, (args.unit_position, args.unit_id, args.unit_variant))
    ops = [op for op in all_ops if unit_key(op) == target]
    unit_total_us = sum(op.duration_us for op in ops)

    metadata = analysis.get("metadata", {}) or {}
    summary = analysis.get("summary", {}) or {}
    model_hint = " ".join(
        str(metadata.get(k, "")) for k in ("model", "modelName", "modelPath", "title")
    )

    raw_matches = []
    allowed_origins = {o.strip() for o in args.include_origins.split(",") if o.strip()}
    rank = {op.index: i for i, op in enumerate(sorted(ops, key=lambda o: o.start_ns))}
    for spec in registry.get("patterns", []):
        if spec.get("origin", "mainline") not in allowed_origins:
            continue
        match = match_pattern(
            spec,
            ops,
            unit_total_us,
            model_hint,
            args.tp_size,
            args.source,
            rank,
            args.pattern_window_ops,
            not args.pattern_any_stream,
        )
        if match:
            raw_matches.append(match)
    matches, suppressed = dedupe_by_subsumption(raw_matches)

    covered = {i for m in matches if m["isActionable"] for i in m["targetOperators"]}
    clusters = []
    for cluster in build_clusters(ops, args.max_gap_us):
        described = describe_cluster(cluster, unit_total_us, covered, args.min_cluster_share)
        if described:
            clusters.append(described)

    operator_rows = [
        {
            "index": op.index,
            "kernelName": op.kernel_name,
            "category": op.category,
            "kernelFamily": op.kernel_family,
            "module": op.module,
            "stage": op.stage,
            "durationUs": round(op.duration_us, 3),
            "sharePctOfUnit": round(op.duration_us / unit_total_us * 100.0, 3)
            if unit_total_us > 0
            else 0.0,
            "stream": op.stream,
            "mbu": op.mbu,
            "mfu": op.mfu,
            "nativeElementwise": op.native_elementwise,
            "alreadyFused": op.already_fused,
        }
        for op in sorted(ops, key=lambda o: o.start_ns)
    ]

    result = {
        "schemaVersion": "1.0",
        "generatedBy": "scan_fusion_candidates.py",
        "inputs": {
            "analysis": os.path.abspath(args.analysis),
            "registry": os.path.abspath(args.registry),
            "registryProvenance": registry.get("provenance", {}),
            "source": os.path.abspath(args.source) if args.source else None,
            "maxGapUs": args.max_gap_us,
            "minClusterSharePct": args.min_cluster_share,
            "patternWindowOps": args.pattern_window_ops,
            "patternRequiresSameStream": not args.pattern_any_stream,
            "includeOrigins": sorted(allowed_origins),
            "tpSize": args.tp_size,
        },
        "scope": {
            "unitPosition": target[0],
            "unitId": target[1],
            "unitVariant": target[2],
            "operatorCount": len(ops),
            "unitTotalDurationUs": round(unit_total_us, 3),
            "normalizedLayerDurationUs": summary.get("normalizedLayerDurationUs"),
            "availableVariants": summary.get("distinctUnitVariants"),
        },
        "registryMatches": matches,
        "suppressedMatches": suppressed,
        "adjacencyClusters": clusters,
        "operators": operator_rows,
        "limits": LIMITS,
    }

    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(payload + "\n")
        actionable = sum(1 for m in matches if m["isActionable"])
        new_clusters = sum(1 for c in clusters if not c["explainedByRegistry"])
        print(
            f"unit={target} operators={len(ops)} "
            f"registry_actionable={actionable} registry_active={len(matches) - actionable} "
            f"unexplained_clusters={new_clusters} -> {args.out}"
        )
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    sys.exit(main())
