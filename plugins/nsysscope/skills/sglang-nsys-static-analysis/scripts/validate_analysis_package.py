#!/usr/bin/env python3
"""Validate six-table semantics, MFU evidence, manifest, and frontend parity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any


SUFFIXES = (
    "_operator_origin_table.csv",
    "_opreator_table.csv",
    "_core_compute_table.csv",
    "_auxiliary_operator_table.csv",
    "_op_classification_table.csv",
    "_stage_table.csv",
)
CATEGORY_LABELS = {
    "核心计算": "core",
    "通信": "communication",
    "辅助算子": "auxiliary",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--analysis-json", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).strip(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle, skipinitialspace=True)
        ]


def number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return None


def key(row: dict[str, str]) -> tuple[str, str, float | None, str]:
    return (
        row.get("功能模块", ""),
        row.get("算子名称", ""),
        number(row.get("算子耗时(us)")),
        row.get("python_function", ""),
    )


def main() -> None:
    args = parse_args()
    root = args.package
    missing = [f"{args.prefix}{suffix}" for suffix in SUFFIXES if not (root / f"{args.prefix}{suffix}").is_file()]
    errors: list[str] = []
    if missing:
        errors.append(f"missing required tables: {missing}")
    if errors:
        raise SystemExit("; ".join(errors))

    origin = read_csv(root / f"{args.prefix}_operator_origin_table.csv")
    overview = read_csv(root / f"{args.prefix}_opreator_table.csv")
    core = read_csv(root / f"{args.prefix}_core_compute_table.csv")
    auxiliary = read_csv(root / f"{args.prefix}_auxiliary_operator_table.csv")
    classes = read_csv(root / f"{args.prefix}_op_classification_table.csv")
    origin_ops = [row for row in origin if row.get("module") != "__layer_total__"]
    if len(origin_ops) != len(overview):
        errors.append(f"origin/overview row mismatch: {len(origin_ops)} != {len(overview)}")

    core_counter = Counter(key(row) for row in core)
    auxiliary_counter = Counter(key(row) for row in auxiliary)
    categories: list[str] = []
    for row in overview:
        row_key = key(row)
        if core_counter[row_key]:
            core_counter[row_key] -= 1
            categories.append("core")
        elif auxiliary_counter[row_key]:
            auxiliary_counter[row_key] -= 1
            categories.append("auxiliary")
        else:
            categories.append("communication")
    if sum(core_counter.values()) or sum(auxiliary_counter.values()):
        errors.append("core/auxiliary tables contain rows absent from overview")

    totals = {
        category: {
            "count": sum(item == category for item in categories),
            "duration": sum(
                number(row.get("算子耗时(us)")) or 0
                for row, item in zip(overview, categories, strict=True)
                if item == category
            ),
        }
        for category in ("core", "communication", "auxiliary")
    }
    if [row.get("算子类型") for row in classes] != list(CATEGORY_LABELS):
        errors.append("classification rows must be 核心计算, 通信, 辅助算子 in order")
    for row in classes:
        category = CATEGORY_LABELS.get(row.get("算子类型", ""))
        if not category:
            continue
        if int(number(row.get("算子数量")) or -1) != totals[category]["count"]:
            errors.append(f"{row['算子类型']} count disagrees with operator membership")
        if not math.isclose(
            number(row.get("总耗时(us)")) or -1,
            totals[category]["duration"],
            abs_tol=0.01,
        ):
            errors.append(f"{row['算子类型']} duration disagrees with operator membership")

    for row in core:
        mfu = number(row.get("mfu"))
        shape = row.get("shape", "")
        if shape and re.search(r"\bM\s*=", shape) and mfu is None:
            errors.append(f"core GEMM has shape but missing MFU: {row.get('算子名称')}")
        if mfu is not None and not 0 <= mfu <= 100:
            errors.append(f"MFU outside [0,100]: {row.get('算子名称')}={mfu}")

    metadata_root = (
        root.parent / "metadata"
        if root.name == "csv" and (root.parent / "metadata").is_dir()
        else root
    )
    manifest_path = metadata_root / f"{args.prefix}_analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    selected = (
        manifest.get("selected_unit")
        or manifest.get("repeating_unit")
        or manifest.get("repeating_unit_selection")
        or {}
    )
    variants = selected.get("distinct_layer_variants") if isinstance(selected, dict) else None
    if isinstance(variants, list) and len(variants) > 1:
        composition = selected.get("composition", [])
        represented = {
            item.get("variant")
            for item in composition
            if isinstance(item, dict) and item.get("variant")
        }
        missing_variants = set(variants) - represented
        if missing_variants:
            errors.append(f"composite repeating unit omits variants: {sorted(missing_variants)}")

    analysis = None
    if args.analysis_json and args.analysis_json.exists():
        analysis = json.loads(args.analysis_json.read_text())
        front = {row["name"]: row for row in analysis.get("classifications", [])}
        for row in classes:
            current = front.get(row["算子类型"])
            if not current:
                errors.append(f"analysis.json misses classification {row['算子类型']}")
                continue
            category = CATEGORY_LABELS[row["算子类型"]]
            if current.get("count") != totals[category]["count"]:
                errors.append(f"analysis.json changes {row['算子类型']} count")
            if not math.isclose(
                float(current.get("durationUs", -1)),
                totals[category]["duration"],
                abs_tol=0.01,
            ):
                errors.append(f"analysis.json changes {row['算子类型']} duration")
        stable = manifest.get("stable_statistics") or manifest.get("stable_stats") or {}
        expected_samples = (
            stable.get("accepted_sample_count")
            or stable.get("accepted_full_template_sample_count")
        )
        if expected_samples is not None and analysis.get("summary", {}).get("stableSamples") != int(expected_samples):
            errors.append("analysis.json stableSamples disagrees with manifest")

    report = {
        "schema_version": "1.0",
        "status": "failed" if errors else "passed",
        "errors": errors,
        "operator_count": len(overview),
        "category_totals": totals,
        "analysis_json_checked": analysis is not None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
