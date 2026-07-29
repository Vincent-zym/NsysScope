#!/usr/bin/env python3
"""Build six normalized SGLang Nsight static-analysis CSV tables.

The input is the detailed operator-origin CSV produced after selecting and
mapping one complete repeating unit. Model-aware semantic rules may override
the conservative built-in fallback classification.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


ORIGIN_COLUMNS = [
    "序号", "module", "operator_name", "duration_us", "start_ns", "end_ns",
    "device", "stream", "layer_id", "duration_min_us", "duration_max_us",
    "duration_diff_us", "duration_avg_us", "duration_avg_pct_of_total",
    "python_function", "function_introduction", "mapping_reason",
]
OPERATOR_COLUMNS = [
    "序号", "功能模块", "算子名称", "算子耗时(us)", "算子耗时占比(%)",
    "shape", "mfu", "模块耗时(us)", "模块耗时占比(%)", "python_function",
    "功能介绍",
]
CORE_COLUMNS = [
    "序号", "功能模块", "module", "算子名称", "算子耗时(us)",
    "算子耗时占比(%)", "shape", "mfu", "python_function", "功能介绍",
]
AUX_COLUMNS = [
    "序号", "功能模块", "算子名称", "算子耗时(us)", "算子耗时占比(%)",
    "python_function", "功能介绍",
]
CLASS_COLUMNS = ["序号", "算子类型", "算子数量", "总耗时(us)", "耗时占比(%)"]
STAGE_COLUMNS = ["序号", "功能模块", "模块耗时(us)", "模块耗时占比(%)", "功能介绍"]

CATEGORY_NAMES = {
    "core": "核心计算",
    "communication": "通信",
    "auxiliary": "辅助算子",
}

# These are fallbacks only. Model-description/source rules take precedence.
MODULE_FALLBACKS = [
    (r"(input|pre[_/-]?attn).*norm|attention/pre.*norm", "Attention 前置归一化"),
    (r"qkv|q_proj|k_proj|v_proj|query|key_value", "Attention QKV 投影"),
    (r"rope|rotary", "Attention RoPE"),
    (r"cache|kv[_/-]?store|indexer", "Attention KV Cache/Indexer"),
    (r"attention.*core|attn[_/-]?core|flash", "Attention 核心计算"),
    (r"o_proj|output_proj|wo_b|attention/post", "Attention 输出投影"),
    (r"router|topk|gate", "MoE Gate/TopK 路由"),
    (r"dispatch|permute|scatter", "MoE Dispatch"),
    (r"expert", "MoE Experts"),
    (r"combine|unpermute|gather", "MoE Combine"),
    (r"mlp|ffn", "FFN/MLP"),
    (r"norm", "归一化"),
    (r"communication|allreduce|all_reduce|alltoall|all_to_all", "通信"),
    (r"embedding", "Embedding"),
    (r"lm_head", "LM Head"),
]
COMM_RE = re.compile(
    r"nccl|allreduce|all_reduce|alltoall|all_to_all|reduce_scatter|"
    r"allgather|all_gather|broadcast|sendrecv|send_recv",
    re.I,
)
# These operator families are never core compute, even when they run inside an
# Attention, projection, Indexer, or MoE functional stage.
AUXILIARY_RE = re.compile(
    r"quant|dequant|requant|layer[_]?norm|layernorm|rms[_]?norm|rmsnorm|"
    r"generalLayerNorm|rope|rotary|topk|sort|dispatch|permute|unpermute|"
    r"scatter|gather|allgather|cache|memcpy|memset|copy|cast|convert|"
    r"transpose|reshape|index(?:ing)?|hadamard|activation|silu|gelu",
    re.I,
)
GEMM_RE = re.compile(
    r"gemm|matmul|(?:^|[_:])mma|wgmma|cutlass|deep[_]?gemm|cublas|"
    r"grouped[_:].*(?:gemm|matmul)|mega[_]?moe[_:].*impl|"
    r"nvjet_.*(?:TNT|NTN|NN|TN)",
    re.I,
)
ATTENTION_RE = re.compile(
    r"flash.*attn|flashmla|sparse[_]?attn|attention.*(?:fwd|forward)|"
    r"attn.*(?:fwd|forward)|fmha|paged[_]?attention|mqa[_]?logits",
    re.I,
)
ATTENTION_CORE_MODULE_RE = re.compile(
    r"attention.*core|attn[_/-]?core|sparse[_/-]?attention|indexer[_/-]?logits",
    re.I,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate six normalized CSV tables from an operator-origin CSV."
    )
    parser.add_argument("--origin-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", required=True, help="Output prefix, e.g. glm52")
    parser.add_argument(
        "--semantic-map",
        type=Path,
        help="Optional JSON containing ordered model-aware rules and hardware peaks.",
    )
    parser.add_argument("--hardware", help="Hardware key used by semantic-map hardware_tflops")
    parser.add_argument("--stage", choices=["prefill", "decode"])
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--batch-size", type=int)
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader.fieldnames), [dict(row) for row in reader]


def write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_semantics(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("semantic-map must be a JSON object")
    rules = data.get("rules", [])
    if not isinstance(rules, list):
        raise ValueError("semantic-map.rules must be a list")
    for index, rule in enumerate(rules):
        if not isinstance(rule, dict) or not any(k in rule for k in ("module_regex", "operator_regex")):
            raise ValueError(f"semantic rule {index} needs module_regex or operator_regex")
        if rule.get("category") not in (None, *CATEGORY_NAMES):
            raise ValueError(f"semantic rule {index} has invalid category")
        if rule.get("core_kind") not in (None, "gemm", "attention"):
            raise ValueError(f"semantic rule {index} has invalid core_kind")
    return data


def number(value: Any) -> float | None:
    try:
        if value in (None, ""):
            return None
        return float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return None


def fmt(value: float | None, digits: int = 3) -> str:
    return "" if value is None else f"{value:.{digits}f}"


def simplify_operator(name: str) -> str:
    """Return a compact CUDA leaf symbol without runtime arguments."""
    text = re.sub(r"^(?:void|int|float|double|bool)\s+", "", name.strip())
    depth = 0
    cut = len(text)
    for i, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif char == "(" and depth == 0:
            cut = i
            break
    text = text[:cut].strip()
    # Split namespaces only outside template brackets.
    depth = 0
    last = 0
    i = 0
    while i + 1 < len(text):
        if text[i] == "<":
            depth += 1
        elif text[i] == ">" and depth:
            depth -= 1
        elif text[i:i + 2] == "::" and depth == 0:
            last = i + 2
            i += 1
        i += 1
    leaf = text[last:].strip() or name.strip()
    # Preserve short template payloads because they distinguish kernel variants.
    # Abbreviate only payloads too large for a human-facing table.
    template_at = leaf.find("<")
    if template_at > 0 and len(leaf) > 96:
        leaf = f"{leaf[:template_at]}<…>"
    return leaf.strip()


def operator_table_name(raw_name: str, override: Any = None) -> str:
    """Use only kernel-like overrides; reject semantic descriptions as names."""
    if override not in (None, ""):
        candidate = simplify_operator(str(override))
        if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_.$]*(?:<[^>]+>)?", candidate):
            return candidate
    return simplify_operator(raw_name)


def match_rule(row: dict[str, str], rules: list[dict[str, Any]]) -> dict[str, Any]:
    for rule in rules:
        module_ok = (
            "module_regex" not in rule
            or re.search(str(rule["module_regex"]), row.get("module", ""), re.I)
        )
        operator_ok = (
            "operator_regex" not in rule
            or re.search(str(rule["operator_regex"]), row.get("operator_name", ""), re.I)
        )
        if module_ok and operator_ok:
            return rule
    return {}


def fallback_functional_module(module: str) -> str:
    for pattern, label in MODULE_FALLBACKS:
        if re.search(pattern, module, re.I):
            return label
    return "其他模型计算"


def classify_category(
    module: str, operator: str, rule: dict[str, Any] | None = None,
) -> str:
    """Classify each operator independently with strict core-compute guardrails."""
    rule = rule or {}
    text = f"{module} {operator}"
    leaf = simplify_operator(operator).split("<", 1)[0]
    requested = rule.get("category")
    if requested == "communication" or COMM_RE.search(text):
        return "communication"
    # Explicit auxiliary signatures outrank a broad semantic-map module rule.
    if AUXILIARY_RE.search(leaf):
        return "auxiliary"
    if requested == "auxiliary":
        return "auxiliary"
    if rule.get("core_kind") in ("gemm", "attention"):
        return "core"
    if GEMM_RE.search(operator):
        return "core"
    if ATTENTION_RE.search(operator) and ATTENTION_CORE_MODULE_RE.search(module):
        return "core"
    # Permit a cryptically named kernel only when the fine-grained module itself
    # is proven to be Attention core; never inherit from the broader stage.
    if requested == "core" and ATTENTION_CORE_MODULE_RE.search(module):
        return "core"
    return "auxiliary"


def parse_shape(value: Any) -> tuple[int, int, int] | None:
    if isinstance(value, dict):
        try:
            return int(value["M"]), int(value["N"]), int(value["K"])
        except (KeyError, TypeError, ValueError):
            return None
    if isinstance(value, str):
        match = re.search(r"M\s*=\s*(\d+).*?N\s*=\s*(\d+).*?K\s*=\s*(\d+)", value, re.I)
        if match:
            return tuple(map(int, match.groups()))  # type: ignore[return-value]
    return None


def shape_text(shape: tuple[int, int, int] | None) -> str:
    if shape is None:
        return ""
    m, n, k = shape
    return f"(M={m},N={n},K={k})"


def compute_mfu(
    shape: tuple[int, int, int] | None,
    duration_us: float,
    rule: dict[str, Any],
    semantics: dict[str, Any],
    hardware: str | None,
) -> str:
    if shape is None or duration_us <= 0 or not hardware:
        return ""
    peaks = semantics.get("hardware_tflops", {})
    device_peaks = peaks.get(hardware, {}) if isinstance(peaks, dict) else {}
    dtypes = rule.get("dtypes")
    if not isinstance(dtypes, list):
        dtype = str(rule.get("dtype", "")).lower()
        dtypes = [dtype] if dtype else []
    # Mixed-precision GEMM throughput is bounded by the slowest participating
    # operand/accumulation path. Never select peak from weight dtype alone.
    candidate_peaks = [number(device_peaks.get(str(dtype).lower())) for dtype in dtypes]
    candidate_peaks = [value for value in candidate_peaks if value and value > 0]
    peak = min(candidate_peaks) if len(candidate_peaks) == len(dtypes) and dtypes else None
    if not peak or peak <= 0:
        return ""
    m, n, k = shape
    utilization = (2.0 * m * n * k) / (duration_us * peak * 1_000_000.0) * 100.0
    if utilization > 100.0:
        raise ValueError(
            f"MFU {utilization:.2f}% exceeds 100%; verify shape, duration, dtypes and dense peak"
        )
    return f"{utilization:.2f}%"


def main() -> None:
    args = parse_args()
    _, source_rows = read_csv(args.origin_csv)
    semantics = load_semantics(args.semantic_map)
    rules = semantics.get("rules", [])

    total_candidates = [r for r in source_rows if r.get("module") == "__layer_total__"]
    kernel_rows = [r for r in source_rows if r.get("module") != "__layer_total__"]
    total_duration = None
    if total_candidates:
        total_duration = number(total_candidates[-1].get("duration_avg_us"))
        total_duration = total_duration or number(total_candidates[-1].get("duration_us"))
    if not total_duration:
        starts = [number(r.get("start_ns")) for r in kernel_rows]
        ends = [number(r.get("end_ns")) for r in kernel_rows]
        valid_starts = [x for x in starts if x is not None]
        valid_ends = [x for x in ends if x is not None]
        if valid_starts and valid_ends:
            total_duration = (max(valid_ends) - min(valid_starts)) / 1000.0
    if not total_duration or total_duration <= 0:
        raise ValueError("cannot derive positive layer/repeating-unit duration")

    enriched: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    for index, row in enumerate(source_rows, 1):
        rule = {} if row.get("module") == "__layer_total__" else match_rule(row, rules)
        normalized = {column: row.get(column, "") for column in ORIGIN_COLUMNS}
        normalized["序号"] = index
        if not normalized["function_introduction"]:
            if row.get("module") == "__layer_total__":
                normalized["function_introduction"] = "完整重复单元时间窗"
            else:
                normalized["function_introduction"] = str(
                    rule.get("introduction")
                    or "依据模型模块、调用链和内核签名解释算子功能；需结合 manifest 复核。"
                )
        origin_rows.append(normalized)
        if row.get("module") == "__layer_total__":
            continue

        duration = number(row.get("duration_avg_us")) or number(row.get("duration_us")) or 0.0
        functional_module = str(
            rule.get("functional_module") or fallback_functional_module(row.get("module", ""))
        )
        category = classify_category(
            row.get("module", ""), row.get("operator_name", ""), rule,
        )
        introduction = str(
            rule.get("introduction")
            or row.get("function_introduction")
            or "依据模型模块、调用链和内核签名归类；需结合 manifest 复核。"
        )
        shape = parse_shape(rule.get("shape")) if category == "core" else None
        enriched.append({
            "module": row.get("module", ""),
            "功能模块": functional_module,
            "算子名称": operator_table_name(
                row.get("operator_name", ""), rule.get("operator_name"),
            ),
            "算子耗时(us)": duration,
            "算子耗时占比(%)": duration / total_duration * 100.0,
            "shape": shape_text(shape),
            "mfu": compute_mfu(shape, duration, rule, semantics, args.hardware),
            "python_function": row.get("python_function", ""),
            "功能介绍": introduction,
            "stage_introduction": str(rule.get("functional_introduction") or introduction),
            "category": category,
        })

    module_duration: dict[str, float] = defaultdict(float)
    module_intro: dict[str, str] = {}
    for row in enriched:
        module_duration[row["功能模块"]] += row["算子耗时(us)"]
        module_intro.setdefault(row["功能模块"], row["stage_introduction"])

    operator_rows = []
    for index, row in enumerate(enriched, 1):
        module_total = module_duration[row["功能模块"]]
        operator_rows.append({
            "序号": index,
            **row,
            "算子耗时(us)": fmt(row["算子耗时(us)"]),
            "算子耗时占比(%)": fmt(row["算子耗时占比(%)"]),
            "模块耗时(us)": fmt(module_total),
            "模块耗时占比(%)": fmt(module_total / total_duration * 100.0),
        })

    core_source = sorted(
        (r for r in enriched if r["category"] == "core"),
        key=lambda r: r["算子耗时(us)"],
        reverse=True,
    )
    core_rows = [{
        "序号": index, **row,
        "算子耗时(us)": fmt(row["算子耗时(us)"]),
        "算子耗时占比(%)": fmt(row["算子耗时占比(%)"]),
    } for index, row in enumerate(core_source, 1)]

    aux_source = sorted(
        (r for r in enriched if r["category"] == "auxiliary"),
        key=lambda r: r["算子耗时(us)"],
        reverse=True,
    )
    aux_rows = [{
        "序号": index, **row,
        "算子耗时(us)": fmt(row["算子耗时(us)"]),
        "算子耗时占比(%)": fmt(row["算子耗时占比(%)"]),
    } for index, row in enumerate(aux_source, 1)]

    class_rows = []
    for index, category in enumerate(("core", "communication", "auxiliary"), 1):
        selected = [r for r in enriched if r["category"] == category]
        duration = sum(r["算子耗时(us)"] for r in selected)
        class_rows.append({
            "序号": index,
            "算子类型": CATEGORY_NAMES[category],
            "算子数量": len(selected),
            "总耗时(us)": fmt(duration),
            "耗时占比(%)": fmt(duration / total_duration * 100.0),
        })

    stage_source = sorted(module_duration.items(), key=lambda item: item[1], reverse=True)
    stage_rows = [{
        "序号": index,
        "功能模块": module,
        "模块耗时(us)": fmt(duration),
        "模块耗时占比(%)": fmt(duration / total_duration * 100.0),
        "功能介绍": module_intro[module],
    } for index, (module, duration) in enumerate(stage_source, 1)]

    outputs = {
        f"{args.prefix}_operator_origin_table.csv": (ORIGIN_COLUMNS, origin_rows),
        f"{args.prefix}_opreator_table.csv": (OPERATOR_COLUMNS, operator_rows),
        f"{args.prefix}_core_compute_table.csv": (CORE_COLUMNS, core_rows),
        f"{args.prefix}_auxiliary_operator_table.csv": (AUX_COLUMNS, aux_rows),
        f"{args.prefix}_op_classification_table.csv": (CLASS_COLUMNS, class_rows),
        f"{args.prefix}_stage_table.csv": (STAGE_COLUMNS, stage_rows),
    }
    for filename, (columns, rows) in outputs.items():
        write_csv(args.output_dir / filename, columns, rows)

    manifest = {
        "origin_csv": str(args.origin_csv),
        "output_dir": str(args.output_dir),
        "prefix": args.prefix,
        "stage": args.stage,
        "chunk_size": args.chunk_size,
        "batch_size": args.batch_size,
        "hardware": args.hardware,
        "total_duration_us": round(total_duration, 3),
        "duration_basis": "duration_avg_us when available, otherwise duration_us",
        "percentage_denominator": "sqlite repeating-unit wall-span (__layer_total__)",
        "semantic_map": str(args.semantic_map) if args.semantic_map else None,
        "fallback_warning": (
            "Rows not matched by semantic-map use conservative cross-model heuristics; "
            "review model-specific functional modules before final handoff."
        ),
        "outputs": list(outputs),
    }
    (args.output_dir / f"{args.prefix}_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for filename in outputs:
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
