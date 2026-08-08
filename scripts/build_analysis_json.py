#!/usr/bin/env python3
"""Convert the six-table package into the stable NsysScope frontend contract."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any


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


def is_total_row(row: dict[str, str]) -> bool:
    return row.get("序号") == "总计"


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


def optional_json_artifact(
    root: Path, configured: Any, default_name: str,
) -> dict[str, Any]:
    try:
        path = portable_artifact(root, configured, default_name)
    except FileNotFoundError:
        return {}
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON sidecar {path}: {exc}") from exc
    return value if isinstance(value, dict) else {}


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
    stable = (
        manifest.get("stable_statistics")
        or manifest.get("stable_aggregation")
        or manifest.get("stable_stats")
        or {}
    )
    value = first_value(
        stats,
        "accepted_unit_count",
        "accepted_full_template_sample_count",
        "accepted_occurrence_count",
        "accepted_sample_count",
    )
    if value is None and isinstance(stable, dict):
        value = first_value(
            stable,
            "accepted_unit_count",
            "accepted_full_template_sample_count",
            "accepted_occurrence_count",
            "accepted_sample_count",
        )
    if value is None:
        raise KeyError("stable sample count is missing from statistics and manifest")
    return int(value)


def device_id(value: Any) -> int:
    """Parse a device column that may carry a human-readable annotation.

    The six-table spec asks for a bare integer, but real packages have shipped
    values like ``cuda:3 (pp_rank 3)`` or ``GPU 3``. Take the first integer so a
    cosmetic annotation cannot fail the whole conversion; the original string is
    preserved separately as ``deviceLabel``.
    """
    if isinstance(value, bool):
        raise ValueError(f"device is not a device id: {value!r}")
    if isinstance(value, int):
        return value
    match = re.search(r"-?\d+", str(value))
    if not match:
        raise ValueError(f"device has no numeric id: {value!r}")
    return int(match.group())


def included_devices(stats: dict[str, Any], manifest: dict[str, Any]) -> list[int]:
    stable = (
        manifest.get("stable_statistics")
        or manifest.get("stable_aggregation")
        or manifest.get("stable_stats")
        or {}
    )
    devices = stats.get("included_devices") or stats.get("included_devices_ranks")
    if devices is None and isinstance(stable, dict):
        devices = stable.get("included_devices") or stable.get("included_devices_ranks")
    if devices is None:
        counts = stats.get("per_device_sample_counts") or (
            stable.get("per_device_sample_counts") if isinstance(stable, dict) else None
        )
        if isinstance(counts, dict):
            devices = counts.keys()
    if devices is None:
        raise KeyError("included devices are missing from statistics and manifest")
    return sorted(device_id(device) for device in devices)


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


def trace_fingerprint(path: Path) -> dict[str, Any]:
    """Same fingerprint the forward-pipeline builder records, for cross-checking."""
    with open(path, "rb") as fh:
        head = fh.read(1024 * 1024)
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256_head_1mib": hashlib.sha256(head).hexdigest(),
    }


def job_sqlite_path(metadata_root: Path) -> Path | None:
    """Locate this job's own trace, so a foreign table can be detected."""
    context_path = metadata_root / "context.json"
    if context_path.exists():
        try:
            recorded = json.loads(context_path.read_text()).get("sqlite_path")
        except (json.JSONDecodeError, OSError):
            recorded = None
        if recorded and Path(recorded).exists():
            return Path(recorded)
    trace_dir = metadata_root.parent / "trace"
    if trace_dir.is_dir():
        candidates = sorted(trace_dir.glob("*.sqlite"))
        if len(candidates) == 1:
            return candidates[0]
    return None


def build_forward_pipeline(
    root: Path, metadata_root: Path, prefix: str, manifest: dict[str, Any],
) -> dict[str, Any] | None:
    """Read the optional forward-pipeline table into a frontend-ready node.

    Returns None when the table is absent, so packages produced before this table
    existed keep converting unchanged and the frontend simply omits the module.
    """
    path = root / f"{prefix}_forward_pipeline_table.csv"
    if not path.exists():
        # In an organised result package this optional table sits in metadata/ rather
        # than csv/, so a re-convert of an exported package still finds it.
        path = metadata_root / f"{prefix}_forward_pipeline_table.csv"
    if not path.exists():
        return None
    rows = [row for row in read_csv(path) if row.get("环节")]
    if not rows:
        return None

    phases = [
        {
            "name": row["环节"],
            "kind": row["环节类型"],
            "layers": int(row["层数"]) if str(row.get("层数", "")).isdigit() else None,
            "subSteps": int(row["子步数"]) if str(row.get("子步数", "")).isdigit() else None,
            "perUnitUs": number(row.get("单次耗时(us)")),
            "durationUs": number(row.get("总耗时(us)")),
            "stepPct": number(row.get("占forward步(%)")),
            "parentPct": number(row.get("占父环节(%)")),
            "samples": int(row["样本数"]) if str(row.get("样本数", "")).isdigit() else None,
            "minUs": number(row.get("min_us")),
            "maxUs": number(row.get("max_us")),
            "note": row.get("备注") or None,
        }
        for row in rows
    ]

    def find(kind: str | None, name: str | None = None) -> dict[str, Any] | None:
        for item in phases:
            if (kind is None or item["kind"] == kind) and (
                name is None or item["name"] == name
            ):
                return item
        return None

    sidecar = optional_json_artifact(
        metadata_root, None, f"{prefix}_forward_pipeline.json",
    ) or {}
    # The sidecar is the builder's own output, so it wins over the agent-written
    # manifest block for any field both carry.
    info = {
        **(manifest.get("forward_pipeline") or {}),
        **(sidecar.get("forward_pipeline") or sidecar or {}),
    }
    total = find("total")
    # Guard against a table built from a different capture than this package. It is
    # easy to do by hand (copying a decode-derived table into a prefill job) and the
    # result silently misreports the pipeline, so drop the node instead of trusting it.
    recorded = (info or {}).get("trace") or {}
    job_trace = job_sqlite_path(metadata_root)
    if recorded and job_trace and job_trace.exists():
        actual = trace_fingerprint(job_trace)
        if (
            recorded.get("size_bytes") != actual["size_bytes"]
            or recorded.get("sha256_head_1mib") != actual["sha256_head_1mib"]
        ):
            print(
                "WARNING: forward pipeline table was built from "
                f"{recorded.get('path')} but this job's trace is {job_trace}; "
                "dropping forwardPipeline",
                file=sys.stderr,
            )
            return None
    batch = info.get("batch_size")
    gpus = info.get("gpu_count")
    return {
        "rows": phases,
        "summary": {
            "batchSize": batch,
            "gpuCount": gpus,
            # per-rank batch scaled by the GPUs this capture covers
            "clusterBatchSize": batch * gpus if batch and gpus else None,
            "speculativeTokens": info.get("speculative_tokens"),
            "stepCount": (info.get("step_marker") or {}).get("step_count"),
            "sampledSteps": info.get("sampled_steps"),
            "gapThresholdUs": info.get("gap_threshold_us"),
            "stepDurationUs": (total or {}).get("durationUs"),
            "targetUs": (find("phase", "target 主模型") or {}).get("durationUs"),
            "draftUs": (find("phase", "draft 模型") or {}).get("durationUs"),
            "prepDraftUs": info.get("prep_draft_us"),
            "prepVerifyUs": info.get("prep_verify_us"),
            "gapUs": (find("gap") or {}).get("durationUs"),
        },
        "evidence": info or None,
    }


def build_operator_payload(
    raw: dict[str, str],
    view: dict[str, str],
    rule: dict[str, Any],
) -> dict[str, Any]:
    """Build one frontend row, keeping the overview table's human operator name."""
    category = rule.get("category", "auxiliary")
    if category not in {"core", "communication", "auxiliary"}:
        raise ValueError(f"invalid six-table category: {category}")
    unit_position = view.get("单元位置") or raw.get("unit_position") or None
    unit_id = view.get("单元ID") or raw.get("unit_id") or None
    unit_variant = view.get("单元类型") or raw.get("unit_variant") or None
    stage = view["功能模块"]
    return {
        "index": int(raw["序号"]),
        "module": raw["module"],
        "stage": stage,
        # Functional-module selection is intentionally pattern-wide.  Layer
        # identity remains available in unitPosition/unitId/unitVariant.
        "stageKey": stage,
        "unitPosition": int(unit_position) if str(unit_position or "").isdigit() else None,
        "unitId": unit_id,
        "unitVariant": unit_variant,
        "layerId": int(raw["layer_id"]) if str(raw.get("layer_id", "")).isdigit() else None,
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
        # MBU is a percentage like MFU. Older packages emitted a raw bandwidth
        # string ("3977.64GB/s"), which number() cannot parse -- keep the raw text
        # in mbuLabel so those packages still render something meaningful.
        "mbu": number(view.get("mbu")),
        "mbuLabel": (str(view.get("mbu")).strip() or None) if view.get("mbu") else None,
        "startNs": int(raw["start_ns"]),
        "endNs": int(raw["end_ns"]),
        "device": device_id(raw["device"]),
        "deviceLabel": str(raw["device"]).strip() or None,
        "stream": int(raw["stream"]),
        "pythonFunction": raw["python_function"],
        "introduction": view["功能介绍"],
        "mappingReason": raw["mapping_reason"],
        "dispatchCodeSnippet": raw.get("dispatch_code_snippet") or None,
    }


def classification_key(
    row: dict[str, str],
) -> tuple[str, str, str, str, str, float | None, str]:
    return (
        row.get("单元位置", ""),
        row.get("单元ID", ""),
        row.get("单元类型", ""),
        row.get("功能模块", ""),
        row.get("算子名称", ""),
        number(row.get("算子耗时(us)")),
        row.get("python_function", ""),
    )


def table_categories(
    overview: list[dict[str, str]],
    core_rows: list[dict[str, str]],
    auxiliary_rows: list[dict[str, str]],
) -> list[str]:
    """Recover per-operator categories from the normalized six-table contract."""
    core = Counter(classification_key(row) for row in core_rows)
    auxiliary = Counter(classification_key(row) for row in auxiliary_rows)
    categories: list[str] = []
    for row in overview:
        key = classification_key(row)
        if core[key]:
            core[key] -= 1
            categories.append("core")
        elif auxiliary[key]:
            auxiliary[key] -= 1
            categories.append("auxiliary")
        else:
            # The six-table contract has no per-operator communication table.
            # Rows absent from both core and auxiliary tables are communication.
            categories.append("communication")
    unmatched = sum(core.values()) + sum(auxiliary.values())
    if unmatched:
        raise ValueError(
            f"core/auxiliary tables contain {unmatched} rows that do not match the overview"
        )
    return categories


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--prefix", default="glm52")
    parser.add_argument("--model", default="")
    parser.add_argument("--stage", default="")
    parser.add_argument("--hardware", default="")
    args = parser.parse_args()

    root, prefix = args.input_dir, args.prefix
    metadata_root = (
        root.parent / "metadata"
        if root.name == "csv" and (root.parent / "metadata").is_dir()
        else root
    )
    origin = read_csv(root / f"{prefix}_operator_origin_table.csv")
    overview = [
        row for row in read_csv(root / f"{prefix}_opreator_table.csv")
        if not is_total_row(row)
    ]
    core_rows = [
        row for row in read_csv(root / f"{prefix}_core_compute_table.csv")
        if not is_total_row(row)
    ]
    auxiliary_rows = [
        row for row in read_csv(root / f"{prefix}_auxiliary_operator_table.csv")
        if not is_total_row(row)
    ]
    stages = [
        row for row in read_csv(root / f"{prefix}_stage_table.csv")
        if not is_total_row(row)
    ]
    classes = [
        row for row in read_csv(root / f"{prefix}_op_classification_table.csv")
        if not is_total_row(row)
    ]
    manifest_path = metadata_root / f"{prefix}_analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    stats = optional_json_artifact(
        metadata_root,
        manifest.get("position_statistics_sidecar")
        or (manifest.get("stable_statistics") or {}).get("sidecar")
        or (manifest.get("stable_stats") or {}).get("sidecar"),
        f"{prefix}_position_operator_stats.json",
    )
    semantic = optional_json_artifact(
        metadata_root, manifest.get("semantic_map"), f"{prefix}_semantic_map.json",
    )
    validation = optional_json_artifact(
        metadata_root,
        manifest.get("validation_report")
        or (manifest.get("validation") or {}).get("report"),
        "validation_report.json",
    )
    if not validation:
        validation = {
            "status": "not_provided",
            "message": "Imported from the normalized six-table contract without sidecars.",
        }

    total = next(row for row in origin if row["module"] == "__layer_total__")
    origin_ops = [row for row in origin if row["module"] != "__layer_total__"]
    categories = table_categories(overview, core_rows, auxiliary_rows)
    operators = []
    for raw, view, category in zip(origin_ops, overview, categories, strict=True):
        if view.get("module") and view["module"] != raw["module"]:
            raise ValueError(
                f"overview module disagrees with origin at row {raw.get('序号')}"
            )
        rule = match_rule(raw["module"], raw["operator_name"], semantic.get("rules", []))
        rule = {**rule, "category": category}
        operators.append(build_operator_payload(raw, view, rule))
    total_duration = number(total.get("duration_avg_us")) or number(total.get("duration_us"))
    if not total_duration:
        raise ValueError("origin table has no positive repeating-unit total duration")
    classification_names = (
        ("core", "核心计算"),
        ("communication", "通信"),
        ("auxiliary", "辅助算子"),
    )
    classifications = []
    for category, label in classification_names:
        selected = [operator for operator in operators if operator["category"] == category]
        duration = sum(operator["durationUs"] or 0 for operator in selected)
        classifications.append({
            "name": label,
            "count": len(selected),
            "durationUs": duration,
            "durationPct": duration / total_duration * 100,
        })
    if len(classes) != 3:
        raise ValueError("classification table must contain exactly three rows")
    class_rows = {row["算子类型"]: row for row in classes}
    for computed in classifications:
        source = class_rows.get(computed["name"])
        if source is None:
            raise ValueError(f"classification table misses {computed['name']}")
        source_count = number(source.get("算子数量"))
        if source_count is None or int(source_count) != computed["count"]:
            raise ValueError(f"classification count mismatch for {computed['name']}")
        source_duration = number(source.get("总耗时(us)"))
        if source_duration is None or abs(source_duration - computed["durationUs"]) > 0.01:
            raise ValueError(f"classification duration mismatch for {computed['name']}")
    try:
        stable_samples = stable_sample_count(stats, manifest)
    except KeyError:
        stable_samples = 1
    try:
        devices = included_devices(stats, manifest)
    except KeyError:
        devices = sorted({device_id(row["device"]) for row in origin_ops})

    selected_unit = (
        manifest.get("selected_unit")
        or manifest.get("repeating_unit")
        or manifest.get("repeating_unit_selection")
    )
    layer_count = first_value(
        selected_unit if isinstance(selected_unit, dict) else {},
        "layer_count", "unit_layer_count",
        default=manifest.get("unit_layer_count"),
    )
    composition = (
        selected_unit.get("composition", [])
        if isinstance(selected_unit, dict)
        else []
    )
    distinct_variants = (
        selected_unit.get("distinct_layer_variants")
        if isinstance(selected_unit, dict)
        else None
    )
    if not distinct_variants:
        distinct_variants = sorted({
            operator["unitVariant"]
            for operator in operators
            if operator.get("unitVariant")
        })
    heterogeneous = len(distinct_variants or []) > 1
    normalized_layer_duration = manifest.get("normalized_single_layer_duration_us")
    if normalized_layer_duration is None and isinstance(layer_count, int) and layer_count > 0:
        normalized_layer_duration = total_duration / layer_count
    if heterogeneous:
        duration_label = "结构周期耗时"
        primary_duration = total_duration
    elif isinstance(layer_count, int) and layer_count > 1:
        duration_label = "平均结构单元耗时"
        primary_duration = normalized_layer_duration or total_duration
    else:
        duration_label = "结构单元耗时"
        primary_duration = total_duration

    # Prefer the explicit pattern-level stage rows when present.  The CSV
    # retains position-aware rows for auditability, while the frontend's
    # functional-module view intentionally uses same-name modules aggregated
    # across all layers in the repeating pattern.
    pattern_stages = [
        row for row in stages if row.get("单元ID") == "__pattern_total__"
    ]
    stage_source = pattern_stages or stages
    stage_payload = []
    for row in stage_source:
        unit_position = row.get("单元位置") or None
        unit_id = row.get("单元ID") or None
        unit_variant = row.get("单元类型") or None
        name = row["功能模块"]
        # Pattern-level rows deliberately use the plain functional-module
        # name so they match operator stageKey values across all layers.
        stage_key = name if unit_id == "__pattern_total__" else "::".join(
            str(value) for value in (unit_position, unit_id, unit_variant, name)
            if value not in (None, "")
        )
        stage_payload.append({
            "key": stage_key or name,
            "name": name,
            "unitPosition": int(unit_position) if str(unit_position or "").isdigit() else None,
            "unitId": unit_id,
            "unitVariant": unit_variant,
            "durationUs": number(row["模块耗时(us)"]),
            "durationPct": number(row["模块耗时占比(%)"]),
            "busyUnionUs": number(row.get("代表区间并集(us)")),
            "wallSpanUs": number(row.get("代表墙钟跨度(us)")),
            "durationBasis": row.get("耗时口径") or "算子平均耗时之和",
            "introduction": row["功能介绍"],
        })

    unit_groups: dict[tuple[int | None, str | None, str | None], list[dict[str, Any]]] = {}
    for operator in operators:
        group_key = (
            operator.get("unitPosition"),
            operator.get("unitId"),
            operator.get("unitVariant"),
        )
        unit_groups.setdefault(group_key, []).append(operator)
    units = []
    for (position, unit_id, variant), rows in sorted(
        unit_groups.items(), key=lambda item: (item[0][0] is None, item[0][0] or 0),
    ):
        starts = [row["startNs"] for row in rows]
        ends = [row["endNs"] for row in rows]
        units.append({
            "position": position,
            "id": unit_id,
            "variant": variant,
            "layerId": next((row["layerId"] for row in rows if row.get("layerId") is not None), None),
            "operatorCount": len(rows),
            "kernelTimeSumUs": sum(row["durationUs"] or 0 for row in rows),
            "representativeWallSpanUs": (max(ends) - min(starts)) / 1000 if starts and ends else None,
            "stageCount": len(stage_payload) if pattern_stages else sum(
                1 for stage_row in stage_payload
                if stage_row["unitPosition"] == position
                and stage_row["unitId"] == unit_id
                and stage_row["unitVariant"] == variant
            ),
        })
    payload = {
        "schemaVersion": "1.0",
        "metadata": {
            "model": model_label(manifest) or args.model or "Imported analysis",
            "stage": manifest.get("stage") or args.stage or "unknown",
            "hardware": manifest.get("hardware") or args.hardware or "Unknown",
            "report": manifest.get("input_report") or first_value(
                manifest.get("inputs") or {}, "original_report", "nsys_rep", "sqlite",
            ) or (
                # The agent's manifest does not always record the input; the job's own
                # trace is still known, and naming it keeps the report identifiable.
                str(job_trace) if (job_trace := job_sqlite_path(metadata_root)) else None
            ),
            "repeatingUnit": repeating_unit_label(
                selected_unit
            ),
            "layerIdEvidence": manifest.get("layer_id_evidence") or (
                selected_unit.get("layer_id_evidence")
                if isinstance(selected_unit, dict) else None
            ),
            "generatedFrom": str(root.parent if root.name == "csv" else root),
        },
        "summary": {
            "totalDurationUs": total_duration,
            "primaryDurationUs": primary_duration,
            "normalizedLayerDurationUs": normalized_layer_duration or total_duration,
            "cycleAveragePerUnitUs": normalized_layer_duration,
            "durationLabel": duration_label,
            "unitLayerCount": layer_count or 1,
            "heterogeneous": heterogeneous,
            "distinctUnitVariants": distinct_variants or [],
            "operatorCount": len(operators),
            "stableSamples": stable_samples,
            "devices": devices,
            "maxMfu": max((op["mfu"] or 0 for op in operators), default=0),
            "maxMbu": max((op["mbu"] or 0 for op in operators), default=0),
        },
        "units": units,
        "stages": stage_payload,
        "classifications": classifications,
        "operators": operators,
        # None for packages produced before this table existed; the frontend then
        # omits the forward-pipeline module entirely.
        "forwardPipeline": build_forward_pipeline(
            root, metadata_root, prefix, manifest,
        ),
        "evidence": {
            "boundary": manifest.get("boundary_evidence") or (
                selected_unit if isinstance(selected_unit, dict) else {}
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
            "mfuEvidence": manifest.get("mfu_evidence") or manifest.get("shape_mfu_evidence"),
            "completeSublayerCheck": manifest.get("complete_sublayer_check"),
            "validation": validation,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
