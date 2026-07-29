#!/usr/bin/env python3
"""Convert the six-table package into the stable NsysScope frontend contract."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


COMMUNICATION_RE = re.compile(
    r"nccl|allreduce|all_reduce|alltoall|all_to_all|reduce_scatter|"
    r"allgather|all_gather|broadcast|sendrecv|send_recv",
    re.I,
)
AUXILIARY_RE = re.compile(
    r"quant|dequant|requant|layer[_]?norm|layernorm|rms[_]?norm|rmsnorm|"
    r"generalLayerNorm|rope|rotary|topk|sort|dispatch|permute|unpermute|"
    r"scatter|gather|allgather|cache|memcpy|memset|copy|cast|convert|"
    r"transpose|reshape|index(?:ing)?|hadamard|activation|silu|gelu",
    re.I,
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle, skipinitialspace=True):
            normalized = {}
            for key, value in row.items():
                if key is None:
                    continue
                if isinstance(value, list):
                    value = ",".join(value)
                normalized[key.strip()] = (value or "").strip()
            rows.append(normalized)
        return rows


def number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(str(value).rstrip("%"))
    except ValueError:
        return None


def first_value(mapping: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Return the first present, non-null value without treating 0/False as missing."""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def portable_artifact(root: Path, configured: Any, default_name: str) -> Path:
    """Prefer a package-local sidecar so copied analysis packages remain portable."""
    local = root / default_name
    if local.exists():
        return local
    if configured:
        candidate = Path(str(configured))
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"required analysis sidecar is missing: {default_name}")


def repeating_unit_label(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, dict):
        return None
    composition = value.get("composition")
    if isinstance(composition, list):
        descriptions = [
            item.get("description")
            for item in composition
            if isinstance(item, dict) and item.get("description")
        ]
        if len(descriptions) == 1:
            return descriptions[0]
        if descriptions:
            return " + ".join(descriptions)
    return first_value(
        value, "smallest_complete_repeating_sequence", "description",
        default="one complete repeating unit",
    )


def model_label(manifest: dict[str, Any]) -> str | None:
    model = manifest.get("model")
    architecture = manifest.get("model_architecture")
    value = str(model or architecture or "").strip()
    if not value:
        return None
    for separator in (" / ", " (", ", architectures=", ", model_type="):
        value = value.split(separator, 1)[0].strip()
    return value[:80]


def mfu_peak_label(manifest: dict[str, Any]) -> str | None:
    value = manifest.get("mfu_effective_peak")
    if value is not None:
        return str(value)
    value = (manifest.get("shape_mfu_evidence") or {}).get("effective_dense_peak_tflops")
    return f"{value:g} TFLOPS/GPU effective dense peak" if isinstance(value, (int, float)) else None


def stable_sample_count(stats: dict[str, Any], manifest: dict[str, Any]) -> int:
    stable = manifest.get("stable_aggregation") or manifest.get("stable_stats") or {}
    value = first_value(
        stats,
        "accepted_unit_count",
        "accepted_full_template_sample_count",
        "accepted_occurrence_count",
    )
    if value is None and isinstance(stable, dict):
        value = first_value(
            stable,
            "accepted_unit_count",
            "accepted_full_template_sample_count",
            "accepted_occurrence_count",
        )
    if value is None:
        raise KeyError("stable sample count is missing from statistics and manifest")
    return int(value)


def included_devices(stats: dict[str, Any], manifest: dict[str, Any]) -> list[int]:
    stable = manifest.get("stable_aggregation") or manifest.get("stable_stats") or {}
    devices = stats.get("included_devices")
    if devices is None and isinstance(stable, dict):
        devices = stable.get("included_devices")
    if devices is None:
        counts = stats.get("per_device_sample_counts") or (
            stable.get("per_device_sample_counts") if isinstance(stable, dict) else None
        )
        if isinstance(counts, dict):
            devices = counts.keys()
    if devices is None:
        raise KeyError("included devices are missing from statistics and manifest")
    return sorted(int(device) for device in devices)


def match_rule(module: str, operator: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    for rule in rules:
        module_ok = "module_regex" not in rule or re.search(rule["module_regex"], module, re.I)
        operator_ok = "operator_regex" not in rule or re.search(rule["operator_regex"], operator, re.I)
        if module_ok and operator_ok:
            return rule
    return {}


def compact_kernel_name(raw_name: str, overview_name: str = "") -> str:
    """Prefer a kernel-like overview name and repair legacy semantic aliases."""
    candidate = overview_name.strip()
    if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_.$]*(?:<[^>]+>)?", candidate):
        return candidate

    text = re.sub(r"^(?:void|int|float|double|bool)\s+", "", raw_name.strip())
    depth = 0
    cut = len(text)
    for index, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">" and depth:
            depth -= 1
        elif char == "(" and depth == 0:
            cut = index
            break
    text = text[:cut].strip()
    depth = 0
    leaf_start = 0
    index = 0
    while index + 1 < len(text):
        if text[index] == "<":
            depth += 1
        elif text[index] == ">" and depth:
            depth -= 1
        elif text[index:index + 2] == "::" and depth == 0:
            leaf_start = index + 2
            index += 1
        index += 1
    leaf = text[leaf_start:].strip() or raw_name.strip()
    template_at = leaf.find("<")
    if template_at > 0 and len(leaf) > 96:
        leaf = f"{leaf[:template_at]}<…>"
    return leaf


def build_operator_payload(
    raw: dict[str, str],
    view: dict[str, str],
    rule: dict[str, Any],
) -> dict[str, Any]:
    """Build one frontend row, keeping the overview table's human operator name."""
    operator = raw["operator_name"]
    leaf = compact_kernel_name(operator).split("<", 1)[0]
    category = rule.get("category", "auxiliary")
    if category == "communication" or COMMUNICATION_RE.search(f"{raw['module']} {operator}"):
        category = "communication"
    elif AUXILIARY_RE.search(leaf):
        # Never let a broad semantic-map rule display an obvious helper kernel
        # as core compute.
        category = "auxiliary"
    return {
        "index": int(raw["序号"]),
        "module": raw["module"],
        "stage": view["功能模块"],
        "name": view["算子名称"],
        "kernelName": compact_kernel_name(raw["operator_name"], view["算子名称"]),
        "fullName": raw["operator_name"],
        "category": category,
        "durationUs": number(view["算子耗时(us)"]),
        "durationPct": number(view["算子耗时占比(%)"]),
        "minUs": number(raw["duration_min_us"]),
        "maxUs": number(raw["duration_max_us"]),
        "diffUs": number(raw["duration_diff_us"]),
        "shape": view["shape"] or None,
        "mfu": number(view["mfu"]),
        "startNs": int(raw["start_ns"]),
        "endNs": int(raw["end_ns"]),
        "device": int(raw["device"]),
        "stream": int(raw["stream"]),
        "pythonFunction": raw["python_function"],
        "introduction": view["功能介绍"],
        "mappingReason": raw["mapping_reason"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", default="glm52")
    args = parser.parse_args()

    root, prefix = args.input_dir, args.prefix
    origin = read_csv(root / f"{prefix}_operator_origin_table.csv")
    overview = read_csv(root / f"{prefix}_opreator_table.csv")
    stages = read_csv(root / f"{prefix}_stage_table.csv")
    classes = read_csv(root / f"{prefix}_op_classification_table.csv")
    manifest = json.loads((root / f"{prefix}_analysis_manifest.json").read_text())
    stats_path = portable_artifact(
        root,
        manifest.get("position_statistics_sidecar")
        or (manifest.get("stable_stats") or {}).get("sidecar"),
        "position_operator_stats.json",
    )
    semantic_path = portable_artifact(
        root, manifest.get("semantic_map"), f"{prefix}_semantic_map.json",
    )
    validation_path = portable_artifact(
        root,
        manifest.get("validation_report")
        or (manifest.get("validation") or {}).get("report"),
        "validation_report.json",
    )
    stats = json.loads(stats_path.read_text())
    semantic = json.loads(semantic_path.read_text())
    validation = json.loads(validation_path.read_text())

    total = next(row for row in origin if row["module"] == "__layer_total__")
    origin_ops = [row for row in origin if row["module"] != "__layer_total__"]
    operators = []
    for raw, view in zip(origin_ops, overview, strict=True):
        rule = match_rule(raw["module"], raw["operator_name"], semantic.get("rules", []))
        operators.append(build_operator_payload(raw, view, rule))

    payload = {
        "schemaVersion": "1.0",
        "metadata": {
            "model": model_label(manifest),
            "stage": manifest.get("stage"),
            "hardware": manifest.get("hardware"),
            "report": manifest.get("input_report") or first_value(
                manifest.get("inputs") or {}, "original_report", "nsys_rep", "sqlite",
            ),
            "repeatingUnit": repeating_unit_label(
                manifest.get("repeating_unit")
                or manifest.get("repeating_unit_selection")
            ),
            "layerIdEvidence": manifest.get("layer_id_evidence"),
            "generatedFrom": str(root),
        },
        "summary": {
            "totalDurationUs": number(total["duration_avg_us"]),
            "operatorCount": len(operators),
            "stableSamples": stable_sample_count(stats, manifest),
            "devices": included_devices(stats, manifest),
            "maxMfu": max((op["mfu"] or 0 for op in operators), default=0),
        },
        "stages": [{
            "name": row["功能模块"],
            "durationUs": number(row["模块耗时(us)"]),
            "durationPct": number(row["模块耗时占比(%)"]),
            "introduction": row["功能介绍"],
        } for row in stages],
        "classifications": [{
            "name": row["算子类型"],
            "count": int(row["算子数量"]),
            "durationUs": number(row["总耗时(us)"]),
            "durationPct": number(row["耗时占比(%)"]),
        } for row in classes],
        "operators": operators,
        "evidence": {
            "boundary": manifest.get("boundary_evidence") or (
                manifest.get("repeating_unit_selection") or {}
            ).get("boundary_evidence"),
            "uncertainMappings": [
                *manifest.get("uncertain_mappings", []),
                *manifest.get("recorded_uncertainties", []),
            ],
            "taxonomySource": manifest.get("functional_module_taxonomy_source"),
            "mfuFormula": manifest.get("mfu_formula") or (
                "2*M*N*K / (duration_seconds * effective_dense_peak_flops)"
                if manifest.get("shape_mfu_evidence") else None
            ),
            "mfuPeak": mfu_peak_label(manifest),
            "mfuEvidence": manifest.get("shape_mfu_evidence"),
            "completeSublayerCheck": manifest.get("complete_sublayer_check"),
            "validation": validation,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
