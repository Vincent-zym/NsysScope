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
    "device", "stream", "layer_id", "unit_position", "unit_id", "unit_variant",
    "duration_min_us", "duration_max_us",
    "duration_diff_us", "duration_avg_us", "duration_avg_pct_of_total",
    "python_function", "function_introduction", "mapping_reason",
]
OPERATOR_COLUMNS = [
    "序号", "单元位置", "单元ID", "单元类型", "功能模块", "module",
    "算子名称",
    "算子耗时(us)", "算子耗时占比(%)",
    "shape", "mfu", "模块耗时(us)", "模块耗时占比(%)", "python_function",
    "功能介绍",
]
CORE_COLUMNS = [
    "序号", "单元位置", "单元ID", "单元类型", "功能模块", "module",
    "算子名称", "算子耗时(us)",
    "算子耗时占比(%)", "模块耗时(us)", "模块耗时占比(%)", "shape", "mfu",
    "python_function", "功能介绍",
]
AUX_COLUMNS = [
    "序号", "单元位置", "单元ID", "单元类型", "功能模块", "算子名称",
    "算子耗时(us)", "算子耗时占比(%)", "模块耗时(us)", "模块耗时占比(%)",
    "python_function", "功能介绍",
]
CLASS_COLUMNS = ["序号", "算子类型", "算子数量", "总耗时(us)", "耗时占比(%)"]
STAGE_COLUMNS = [
    "序号", "单元位置", "单元ID", "单元类型", "功能模块",
    "模块耗时(us)", "模块耗时占比(%)", "代表区间并集(us)",
    "代表墙钟跨度(us)", "耗时口径", "功能介绍",
]

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
# Match the actual leaf kernel family, not arbitrary template metadata.
# A real attention kernel can contain implementation types such as TiledCopy or
# CopyAtom in its encoded symbol without being a copy/helper operator.
AUXILIARY_FAMILY_RE = re.compile(
    r"(?:^|[_:$])(?:quant|dequant|requant|rope|rotary|topk|sort|dispatch|"
    r"permute|unpermute|scatter|gather|allgather|memcpy|memset|copy|cast|"
    r"convert|transpose|reshape|index|indexing|hadamard|activation|silu|gelu)"
    r"(?:[_:$]|$)|layer[_]?norm|layernorm|rms[_]?norm|rmsnorm|"
    r"generalLayerNorm|cache|elementwise|splitKreduce|finalizeKernel",
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
    parser.add_argument(
        "--taxonomy",
        type=Path,
        help="Validated architecture taxonomy describing the repeating-unit positions and variants.",
    )
    parser.add_argument("--hardware", help="Hardware key used by semantic-map hardware_tflops")
    parser.add_argument(
        "--hardware-profiles",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "references/hardware-peaks.json",
        help="Verified per-GPU dense hardware peak registry.",
    )
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


def load_taxonomy(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("architecture taxonomy must be a JSON object")
    if data.get("schema_version") != "1.0":
        raise ValueError("architecture taxonomy schema_version must be 1.0")
    repeating = data.get("repeating_unit")
    variants = data.get("variants")
    evidence = data.get("evidence")
    if not isinstance(repeating, dict) or not isinstance(repeating.get("positions"), list):
        raise ValueError("architecture taxonomy needs repeating_unit.positions")
    if not isinstance(variants, list) or not variants:
        raise ValueError("architecture taxonomy needs a non-empty variants list")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("architecture taxonomy needs model-specific evidence")

    variant_names = {
        item.get("name") for item in variants
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    if len(variant_names) != len(variants):
        raise ValueError("architecture taxonomy variant names must be non-empty and unique")
    positions = repeating["positions"]
    seen_positions: set[int] = set()
    seen_unit_ids: set[str] = set()
    for item in positions:
        if not isinstance(item, dict):
            raise ValueError("each repeating-unit position must be an object")
        position = item.get("position")
        variant = item.get("unit_variant")
        if not isinstance(position, int) or position <= 0 or position in seen_positions:
            raise ValueError("repeating-unit positions must be unique positive integers")
        if variant not in variant_names:
            raise ValueError(f"position {position} references undeclared variant {variant!r}")
        if not item.get("unit_id"):
            raise ValueError(f"position {position} needs unit_id")
        if str(item["unit_id"]) in seen_unit_ids:
            raise ValueError(f"position {position} repeats unit_id {item['unit_id']!r}")
        seen_positions.add(position)
        seen_unit_ids.add(str(item["unit_id"]))
    if seen_positions != set(range(1, len(positions) + 1)):
        raise ValueError("repeating-unit positions must be contiguous from 1")

    for item in variants:
        modules = item.get("ordered_functional_modules")
        if not item.get("source_evidence"):
            raise ValueError(f"variant {item.get('name')!r} needs source_evidence")
        if (
            not isinstance(modules, list)
            or not modules
            or any(not isinstance(module, str) or not module.strip() for module in modules)
            or len(set(modules)) != len(modules)
        ):
            raise ValueError(
                f"variant {item.get('name')!r} needs unique ordered_functional_modules"
            )
        policy = data.get("functional_module_policy") or {}
        target_max = policy.get("target_max", 8) if isinstance(policy, dict) else 8
        if not isinstance(target_max, int) or target_max <= 0 or target_max > 8:
            raise ValueError(
                "functional_module_policy.target_max must be a positive integer <= 8"
            )
        if len(modules) > target_max and not str(
            item.get("granularity_exception") or ""
        ).strip():
            raise ValueError(
                f"variant {item.get('name')!r} has {len(modules)} functional "
                f"modules; more than {target_max} requires granularity_exception"
            )
        if len(variant_names) > 1:
            if not isinstance(item.get("discriminators"), list) or not item["discriminators"]:
                raise ValueError(
                    f"variant {item.get('name')!r} needs model/trace discriminators"
                )
    declared_modules = {
        module
        for item in variants
        for module in item.get("ordered_functional_modules", [])
    }
    for group in data.get("fusion_groups", []):
        functional_module = (
            group.get("functional_module") or group.get("name")
            if isinstance(group, dict) else None
        )
        if (
            not isinstance(group, dict)
            or len(group.get("logical_owners") or []) < 2
            or group.get("attribution_policy") != "indivisible"
            or functional_module not in declared_modules
        ):
            raise ValueError(
                "fusion_groups need a name, a declared functional_module, at least "
                "two logical_owners and attribution_policy=indivisible"
            )
    return data


def taxonomy_position_for_row(
    row: dict[str, str], taxonomy: dict[str, Any],
) -> dict[str, Any]:
    positions = (taxonomy.get("repeating_unit") or {}).get("positions") or []
    layer_id = str(row.get("layer_id", "")).strip()
    module = row.get("module", "")
    matches = []
    for item in positions:
        layer_match = item.get("layer_id")
        module_regex = item.get("module_regex")
        if layer_match not in (None, "") and str(layer_match) == layer_id:
            matches.append(item)
        elif module_regex and re.search(str(module_regex), module, re.I):
            matches.append(item)
    if len(matches) > 1:
        raise ValueError(f"row matches multiple taxonomy positions: {module}")
    return matches[0] if matches else {}


def resolve_unit_fields(
    row: dict[str, str], rule: dict[str, Any], taxonomy: dict[str, Any],
) -> tuple[str, str, str]:
    position = taxonomy_position_for_row(row, taxonomy)
    unit_position = str(
        row.get("unit_position")
        or rule.get("unit_position")
        or position.get("position")
        or ""
    )
    unit_id = str(
        row.get("unit_id")
        or rule.get("unit_id")
        or position.get("unit_id")
        or ""
    )
    unit_variant = str(
        row.get("unit_variant")
        or rule.get("unit_variant")
        or position.get("unit_variant")
        or ""
    )
    if taxonomy and row.get("module") != "__layer_total__":
        if not all((unit_position, unit_id, unit_variant)):
            raise ValueError(
                f"taxonomy could not resolve unit position/id/variant for {row.get('module')}"
            )
    return unit_position, unit_id, unit_variant


def interval_metrics(rows: list[dict[str, Any]]) -> tuple[float, float]:
    intervals = sorted(
        (
            int(row["start_ns"]),
            int(row["end_ns"]),
        )
        for row in rows
        if str(row.get("start_ns", "")).isdigit()
        and str(row.get("end_ns", "")).isdigit()
        and int(row["end_ns"]) > int(row["start_ns"])
    )
    if not intervals:
        return 0.0, 0.0
    union_ns = 0
    current_start, current_end = intervals[0]
    for start, end in intervals[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            union_ns += current_end - current_start
            current_start, current_end = start, end
    union_ns += current_end - current_start
    wall_ns = max(end for _, end in intervals) - min(start for start, _ in intervals)
    return union_ns / 1000.0, wall_ns / 1000.0


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
    # Only actual helper leaf families override a semantic rule. Never search
    # arbitrary encoded template metadata such as CopyAtom/TiledCopy.
    if AUXILIARY_FAMILY_RE.search(leaf):
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


def normalize_hardware(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def load_hardware_profiles(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text())
    profiles = payload.get("profiles", {})
    if not isinstance(profiles, dict):
        raise ValueError("hardware profile registry has invalid profiles")
    return payload


def resolve_peak(
    rule: dict[str, Any],
    semantics: dict[str, Any],
    hardware_profiles: dict[str, Any],
    hardware: str | None,
) -> tuple[float | None, str | None, str | None]:
    explicit = number(rule.get("mfu_peak_tflops"))
    if explicit and explicit > 0:
        return explicit, str(rule.get("compute_dtype") or "explicit"), "semantic rule"
    if not hardware:
        return None, None, None
    compute_dtype = str(
        rule.get("compute_dtype")
        or rule.get("mfu_dtype")
        or ""
    ).lower()
    if not compute_dtype:
        dtypes = rule.get("dtypes")
        if isinstance(dtypes, list):
            tensor_inputs = [
                str(item).lower()
                for item in dtypes
                if str(item).lower() not in {"fp32_accum", "fp32-accum", "accum_fp32"}
            ]
            if len(set(tensor_inputs)) == 1:
                compute_dtype = tensor_inputs[0]
    if not compute_dtype:
        return None, None, None

    legacy = semantics.get("hardware_tflops", {})
    if isinstance(legacy, dict):
        device = legacy.get(hardware, {})
        if isinstance(device, dict):
            peak = number(device.get(compute_dtype))
            if peak and peak > 0:
                return peak, compute_dtype, "semantic-map hardware_tflops"

    wanted = normalize_hardware(hardware)
    for profile in (hardware_profiles.get("profiles") or {}).values():
        if not isinstance(profile, dict):
            continue
        aliases = [profile.get("display_name"), *(profile.get("aliases") or [])]
        if wanted not in {normalize_hardware(str(alias)) for alias in aliases if alias}:
            continue
        peak = number((profile.get("dense_tflops_per_gpu") or {}).get(compute_dtype))
        if peak and peak > 0:
            source = (hardware_profiles.get("source") or {}).get("url")
            return peak, compute_dtype, str(source or "hardware profile registry")
    return None, compute_dtype, None


def compute_mfu(
    shape: tuple[int, int, int] | None,
    duration_us: float,
    rule: dict[str, Any],
    semantics: dict[str, Any],
    hardware_profiles: dict[str, Any],
    hardware: str | None,
) -> tuple[str, dict[str, Any] | None]:
    if shape is None or duration_us <= 0 or not hardware:
        return "", None
    peak, compute_dtype, source = resolve_peak(
        rule, semantics, hardware_profiles, hardware,
    )
    if not peak or peak <= 0:
        return "", None
    m, n, k = shape
    utilization = (2.0 * m * n * k) / (duration_us * peak * 1_000_000.0) * 100.0
    if utilization > 100.0:
        raise ValueError(
            f"MFU {utilization:.2f}% exceeds 100%; verify shape, duration, dtypes and dense peak"
        )
    evidence = {
        "shape": {"M": m, "N": n, "K": k},
        "duration_us": duration_us,
        "compute_dtype": compute_dtype,
        "operand_dtypes": rule.get("dtypes") or [],
        "dense_peak_tflops_per_gpu": peak,
        "peak_source": source,
        "formula": "2*M*N*K/(duration_seconds*dense_peak_flops)",
        "mfu_pct": round(utilization, 4),
    }
    return f"{utilization:.2f}%", evidence


def main() -> None:
    args = parse_args()
    _, source_rows = read_csv(args.origin_csv)
    semantics = load_semantics(args.semantic_map)
    taxonomy = load_taxonomy(args.taxonomy)
    hardware_profiles = load_hardware_profiles(args.hardware_profiles)
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

    total_source = dict(total_candidates[-1]) if total_candidates else {}
    total_source.update({
        "module": "__layer_total__",
        "operator_name": total_source.get("operator_name") or "__layer_total__",
        "duration_us": total_source.get("duration_us") or fmt(total_duration),
        "duration_avg_us": total_source.get("duration_avg_us") or fmt(total_duration),
        "duration_avg_pct_of_total": "100.000",
        "start_ns": total_source.get("start_ns") or min(
            (row.get("start_ns", "") for row in kernel_rows),
            key=lambda value: int(value) if str(value).isdigit() else 2**63,
            default="",
        ),
        "end_ns": total_source.get("end_ns") or max(
            (row.get("end_ns", "") for row in kernel_rows),
            key=lambda value: int(value) if str(value).isdigit() else -1,
            default="",
        ),
        "unit_position": "",
        "unit_id": "",
        "unit_variant": "",
    })
    ordered_source_rows = [*kernel_rows, total_source]

    enriched: list[dict[str, Any]] = []
    origin_rows: list[dict[str, Any]] = []
    mfu_evidence: list[dict[str, Any]] = []
    for index, row in enumerate(ordered_source_rows, 1):
        rule = {} if row.get("module") == "__layer_total__" else match_rule(row, rules)
        unit_position, unit_id, unit_variant = resolve_unit_fields(
            row, rule, taxonomy,
        )
        normalized = {column: row.get(column, "") for column in ORIGIN_COLUMNS}
        normalized["序号"] = index
        normalized["unit_position"] = unit_position
        normalized["unit_id"] = unit_id
        normalized["unit_variant"] = unit_variant
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
        mfu, evidence = compute_mfu(
            shape, duration, rule, semantics, hardware_profiles, args.hardware,
        )
        if evidence is not None:
            evidence["origin_index"] = int(row.get("序号") or index)
            evidence["module"] = row.get("module", "")
            evidence["operator_name"] = operator_table_name(
                row.get("operator_name", ""), rule.get("operator_name"),
            )
            mfu_evidence.append(evidence)
        enriched.append({
            "module": row.get("module", ""),
            "单元位置": unit_position,
            "单元ID": unit_id,
            "单元类型": unit_variant,
            "功能模块": functional_module,
            "算子名称": operator_table_name(
                row.get("operator_name", ""), rule.get("operator_name"),
            ),
            "算子耗时(us)": duration,
            "算子耗时占比(%)": duration / total_duration * 100.0,
            "shape": shape_text(shape),
            "mfu": mfu,
            "python_function": row.get("python_function", ""),
            "功能介绍": introduction,
            "stage_introduction": str(rule.get("functional_introduction") or introduction),
            "category": category,
            "start_ns": row.get("start_ns", ""),
            "end_ns": row.get("end_ns", ""),
        })

    def stage_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
        return (
            str(row["单元位置"]),
            str(row["单元ID"]),
            str(row["单元类型"]),
            str(row["功能模块"]),
        )

    module_duration: dict[tuple[str, str, str, str], float] = defaultdict(float)
    module_rows: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    module_intro: dict[tuple[str, str, str, str], str] = {}
    module_intro_priority: dict[tuple[str, str, str, str], int] = {}
    for row in enriched:
        key = stage_key(row)
        module_duration[key] += row["算子耗时(us)"]
        module_rows[key].append(row)
        priority = {"core": 3, "communication": 2, "auxiliary": 1}[row["category"]]
        if priority > module_intro_priority.get(key, -1):
            module_intro[key] = row["stage_introduction"]
            module_intro_priority[key] = priority

    operator_rows = []
    for index, row in enumerate(enriched, 1):
        module_total = module_duration[stage_key(row)]
        operator_rows.append({
            "序号": index,
            **row,
            "算子耗时(us)": fmt(row["算子耗时(us)"]),
            "算子耗时占比(%)": fmt(row["算子耗时占比(%)"]),
            "模块耗时(us)": fmt(module_total),
            "模块耗时占比(%)": fmt(module_total / total_duration * 100.0),
        })
    accumulated_duration = sum(row["算子耗时(us)"] for row in enriched)
    accumulated_pct = accumulated_duration / total_duration * 100.0
    operator_rows.append({
        "序号": "总计",
        "功能模块": "总计",
        "module": "__total__",
        "算子名称": "总计",
        "算子耗时(us)": fmt(accumulated_duration),
        "算子耗时占比(%)": fmt(accumulated_pct),
        "模块耗时(us)": fmt(accumulated_duration),
        "模块耗时占比(%)": fmt(accumulated_pct),
        "功能介绍": "全部算子稳定样本平均耗时之和；因并行重叠可能超过重复单元墙钟耗时。",
    })

    pattern_module_duration: dict[str, float] = defaultdict(float)
    pattern_module_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pattern_module_intro: dict[str, str] = {}
    for key, duration in module_duration.items():
        name = key[3]
        pattern_module_duration[name] += duration
        pattern_module_rows[name].extend(module_rows[key])
        pattern_module_intro.setdefault(name, module_intro[key])

    core_source = sorted(
        (r for r in enriched if r["category"] == "core"),
        key=lambda r: (r["功能模块"], -r["算子耗时(us)"]),
    )
    core_rows = [{
        "序号": index, **row,
        "算子耗时(us)": fmt(row["算子耗时(us)"]),
        "算子耗时占比(%)": fmt(row["算子耗时占比(%)"]),
        "模块耗时(us)": fmt(pattern_module_duration[row["功能模块"]]),
        "模块耗时占比(%)": fmt(pattern_module_duration[row["功能模块"]] / total_duration * 100.0),
    } for index, row in enumerate(core_source, 1)]
    core_duration = sum(row["算子耗时(us)"] for row in core_source)
    core_rows.append({
        "序号": "总计",
        "功能模块": "总计",
        "module": "__total__",
        "算子名称": "总计",
        "算子耗时(us)": fmt(core_duration),
        "算子耗时占比(%)": fmt(core_duration / total_duration * 100.0),
        "模块耗时(us)": fmt(core_duration),
        "模块耗时占比(%)": fmt(core_duration / total_duration * 100.0),
        "功能介绍": "核心计算算子稳定样本平均耗时之和。",
    })

    aux_source = sorted(
        (r for r in enriched if r["category"] == "auxiliary"),
        key=lambda r: (r["功能模块"], -r["算子耗时(us)"]),
    )
    aux_rows = [{
        "序号": index, **row,
        "算子耗时(us)": fmt(row["算子耗时(us)"]),
        "算子耗时占比(%)": fmt(row["算子耗时占比(%)"]),
        "模块耗时(us)": fmt(pattern_module_duration[row["功能模块"]]),
        "模块耗时占比(%)": fmt(pattern_module_duration[row["功能模块"]] / total_duration * 100.0),
    } for index, row in enumerate(aux_source, 1)]
    auxiliary_duration = sum(row["算子耗时(us)"] for row in aux_source)
    aux_rows.append({
        "序号": "总计",
        "功能模块": "总计",
        "算子名称": "总计",
        "算子耗时(us)": fmt(auxiliary_duration),
        "算子耗时占比(%)": fmt(auxiliary_duration / total_duration * 100.0),
        "模块耗时(us)": fmt(auxiliary_duration),
        "模块耗时占比(%)": fmt(auxiliary_duration / total_duration * 100.0),
        "功能介绍": "辅助算子稳定样本平均耗时之和。",
    })

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
    class_rows.append({
        "序号": "总计",
        "算子类型": "总计",
        "算子数量": len(enriched),
        "总耗时(us)": fmt(accumulated_duration),
        "耗时占比(%)": fmt(accumulated_pct),
    })

    stage_source = sorted(
        module_duration.items(),
        key=lambda item: (
            int(item[0][0]) if item[0][0].isdigit() else 0,
            -item[1],
            item[0][3],
        ),
    )
    stage_rows = [{
        "序号": index,
        "单元位置": key[0],
        "单元ID": key[1],
        "单元类型": key[2],
        "功能模块": key[3],
        "模块耗时(us)": fmt(duration),
        "模块耗时占比(%)": fmt(duration / total_duration * 100.0),
        "代表区间并集(us)": fmt(interval_metrics(module_rows[key])[0]),
        "代表墙钟跨度(us)": fmt(interval_metrics(module_rows[key])[1]),
        "耗时口径": "稳定样本逐算子平均耗时之和；区间指标来自代表样本且不拆分融合算子",
        "功能介绍": module_intro[key],
    } for index, (key, duration) in enumerate(stage_source, 1)]
    # Keep the position-aware rows for traceability, and add a pattern-level
    # view for the secondary analysis tables.  The latter is the view used by
    # the dashboard: equal functional modules from different layers are
    # intentionally accumulated together instead of appearing as four layer
    # entries.
    pattern_stage_rows = [{
        "序号": f"P{index}",
        "单元位置": "",
        "单元ID": "__pattern_total__",
        "单元类型": "",
        "功能模块": name,
        "模块耗时(us)": fmt(duration),
        "模块耗时占比(%)": fmt(duration / total_duration * 100.0),
        "代表区间并集(us)": fmt(interval_metrics(pattern_module_rows[name])[0]),
        "代表墙钟跨度(us)": fmt(interval_metrics(pattern_module_rows[name])[1]),
        "耗时口径": "Pattern 内同名功能模块跨层汇总；稳定样本逐算子平均耗时之和",
        "功能介绍": pattern_module_intro[name],
    } for index, (name, duration) in enumerate(sorted(
        pattern_module_duration.items(), key=lambda item: (-item[1], item[0]),
    ), 1)]
    stage_rows.extend(pattern_stage_rows)
    representative_union, representative_wall = interval_metrics(enriched)
    stage_rows.append({
        "序号": "总计",
        "功能模块": "总计",
        "模块耗时(us)": fmt(accumulated_duration),
        "模块耗时占比(%)": fmt(accumulated_pct),
        "代表区间并集(us)": fmt(representative_union),
        "代表墙钟跨度(us)": fmt(representative_wall),
        "耗时口径": "全部功能模块的稳定样本算子平均耗时之和；区间指标覆盖完整代表样本",
        "功能介绍": "全部结构位置和功能模块总计。",
    })

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
        "hardware_profiles": str(args.hardware_profiles) if args.hardware_profiles else None,
        "total_duration_us": round(total_duration, 3),
        "duration_basis": "duration_avg_us when available, otherwise duration_us",
        "percentage_denominator": "sqlite repeating-unit wall-span (__layer_total__)",
        "semantic_map": str(args.semantic_map) if args.semantic_map else None,
        "architecture_taxonomy": str(args.taxonomy) if args.taxonomy else None,
        "taxonomy_schema_version": taxonomy.get("schema_version") if taxonomy else None,
        "table_contract_version": "1.2",
        "total_rows": {
            "required": True,
            "marker": "序号=总计；原始表使用 module=__layer_total__",
            "operator_tables": "sum of stable per-operator average durations",
            "origin_table": "repeating-unit wall-span",
            "percentages_may_exceed_100_due_to_overlap": True,
        },
        "fallback_warning": (
            "Rows not matched by semantic-map use conservative cross-model heuristics; "
            "review model-specific functional modules before final handoff."
        ),
        "mfu_formula": "2*M*N*K/(duration_seconds*dense_peak_flops)",
        "mfu_evidence": mfu_evidence,
        "module_duration_basis": {
            "模块耗时(us)": "sum of position-aware average kernel durations",
            "代表区间并集(us)": "representative-sample interval union",
            "代表墙钟跨度(us)": "representative-sample first-start to last-end",
            "fusion_policy": "indivisible fused kernels are never split among logical owners",
        },
        "outputs": list(outputs),
    }
    (args.output_dir / f"{args.prefix}_analysis_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    for filename in outputs:
        print(args.output_dir / filename)


if __name__ == "__main__":
    main()
