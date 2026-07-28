#!/usr/bin/env python3
"""Convert the six-table package into the stable NsysScope frontend contract."""

from __future__ import annotations

import argparse
import csv
import json
import re
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


def match_rule(module: str, operator: str, rules: list[dict[str, Any]]) -> dict[str, Any]:
    for rule in rules:
        module_ok = "module_regex" not in rule or re.search(rule["module_regex"], module, re.I)
        operator_ok = "operator_regex" not in rule or re.search(rule["operator_regex"], operator, re.I)
        if module_ok and operator_ok:
            return rule
    return {}


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
    stats = json.loads((root / "position_operator_stats.json").read_text())
    semantic = json.loads(Path(manifest["semantic_map"]).read_text())

    total = next(row for row in origin if row["module"] == "__layer_total__")
    origin_ops = [row for row in origin if row["module"] != "__layer_total__"]
    operators = []
    for raw, view in zip(origin_ops, overview, strict=True):
        rule = match_rule(raw["module"], raw["operator_name"], semantic.get("rules", []))
        operators.append({
            "index": int(raw["序号"]),
            "module": raw["module"],
            "stage": view["功能模块"],
            "name": view["算子名称"],
            "fullName": raw["operator_name"],
            "category": rule.get("category", "auxiliary"),
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
        })

    payload = {
        "schemaVersion": "1.0",
        "metadata": {
            "model": manifest.get("model"),
            "stage": manifest.get("stage"),
            "hardware": manifest.get("hardware"),
            "report": manifest.get("input_report"),
            "repeatingUnit": manifest.get("repeating_unit"),
            "layerIdEvidence": manifest.get("layer_id_evidence"),
            "generatedFrom": str(root),
        },
        "summary": {
            "totalDurationUs": number(total["duration_avg_us"]),
            "operatorCount": len(operators),
            "stableSamples": stats["accepted_unit_count"],
            "devices": stats["included_devices"],
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
            "boundary": manifest.get("boundary_evidence"),
            "uncertainMappings": manifest.get("uncertain_mappings", []),
            "taxonomySource": manifest.get("functional_module_taxonomy_source"),
            "mfuFormula": manifest.get("mfu_formula"),
            "mfuPeak": manifest.get("mfu_effective_peak"),
            "completeSublayerCheck": manifest.get("complete_sublayer_check"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
