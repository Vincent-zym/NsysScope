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
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [
            {str(key).strip(): str(value or "").strip() for key, value in row.items()}
            for row in csv.DictReader(handle, skipinitialspace=True)
        ]


def read_fields(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [str(field).strip() for field in (csv.DictReader(handle).fieldnames or [])]


def is_total_row(row: dict[str, str]) -> bool:
    return row.get("序号") == "总计"


def number(value: Any) -> float | None:
    try:
        return None if value in (None, "") else float(str(value).rstrip("%"))
    except (TypeError, ValueError):
        return None


def key(row: dict[str, str]) -> tuple[str, str, str, str, str, float | None, str]:
    return (
        row.get("单元位置", ""),
        row.get("单元ID", ""),
        row.get("单元类型", ""),
        row.get("功能模块", ""),
        row.get("算子名称", ""),
        number(row.get("算子耗时(us)")),
        row.get("python_function", ""),
    )


def portable_json(
    metadata_root: Path, configured: Any, default_name: str,
) -> dict[str, Any]:
    local = metadata_root / default_name
    if local.exists():
        path = local
    elif configured:
        path = Path(str(configured))
        if not path.is_absolute():
            path = metadata_root / path
        if not path.exists():
            return {}
    else:
        return {}
    value = json.loads(path.read_text())
    return value if isinstance(value, dict) else {}


def main() -> None:
    args = parse_args()
    root = args.package
    missing = [f"{args.prefix}{suffix}" for suffix in SUFFIXES if not (root / f"{args.prefix}{suffix}").is_file()]
    errors: list[str] = []
    if missing:
        errors.append(f"missing required tables: {missing}")
    if errors:
        raise SystemExit("; ".join(errors))

    paths = {
        "origin": root / f"{args.prefix}_operator_origin_table.csv",
        "overview": root / f"{args.prefix}_opreator_table.csv",
        "core": root / f"{args.prefix}_core_compute_table.csv",
        "auxiliary": root / f"{args.prefix}_auxiliary_operator_table.csv",
        "classes": root / f"{args.prefix}_op_classification_table.csv",
        "stages": root / f"{args.prefix}_stage_table.csv",
    }
    origin = read_csv(paths["origin"])
    overview_all = read_csv(paths["overview"])
    core_all = read_csv(paths["core"])
    auxiliary_all = read_csv(paths["auxiliary"])
    classes_all = read_csv(paths["classes"])
    stages_all = read_csv(paths["stages"])
    overview = [row for row in overview_all if not is_total_row(row)]
    core = [row for row in core_all if not is_total_row(row)]
    auxiliary = [row for row in auxiliary_all if not is_total_row(row)]
    classes = [row for row in classes_all if not is_total_row(row)]
    stages = [row for row in stages_all if not is_total_row(row)]
    origin_ops = [row for row in origin if row.get("module") != "__layer_total__"]
    if len(origin_ops) != len(overview):
        errors.append(f"origin/overview row mismatch: {len(origin_ops)} != {len(overview)}")
    for raw, view in zip(origin_ops, overview):
        if view.get("module") and view["module"] != raw.get("module"):
            errors.append(
                f"overview module disagrees with origin at row {raw.get('序号')}"
            )

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
    total_rows_required = bool(
        (manifest.get("total_rows") or {}).get("required")
        or str(manifest.get("table_contract_version") or "") >= "1.2"
    )
    total_rows = {
        "origin": [row for row in origin if row.get("module") == "__layer_total__"],
        "overview": [row for row in overview_all if is_total_row(row)],
        "core": [row for row in core_all if is_total_row(row)],
        "auxiliary": [row for row in auxiliary_all if is_total_row(row)],
        "classes": [row for row in classes_all if is_total_row(row)],
        "stages": [row for row in stages_all if is_total_row(row)],
    }
    if total_rows_required:
        for name, rows in total_rows.items():
            if len(rows) != 1:
                errors.append(f"{name} table needs exactly one total row")
        final_markers = {
            "origin": bool(origin) and origin[-1].get("module") == "__layer_total__",
            "overview": bool(overview_all) and is_total_row(overview_all[-1]),
            "core": bool(core_all) and is_total_row(core_all[-1]),
            "auxiliary": bool(auxiliary_all) and is_total_row(auxiliary_all[-1]),
            "classes": bool(classes_all) and is_total_row(classes_all[-1]),
            "stages": bool(stages_all) and is_total_row(stages_all[-1]),
        }
        for name, is_final in final_markers.items():
            if not is_final:
                errors.append(f"{name} total row must be the final row")

        overview_fields = read_fields(paths["overview"])
        try:
            operator_index = overview_fields.index("算子名称")
        except ValueError:
            operator_index = -1
        if operator_index <= 0 or overview_fields[operator_index - 1] != "module":
            errors.append("overview module must immediately precede 算子名称")

        accumulated_duration = sum(
            number(row.get("算子耗时(us)")) or 0 for row in overview
        )
        core_duration = totals["core"]["duration"]
        auxiliary_duration = totals["auxiliary"]["duration"]
        total_duration = (
            number(total_rows["origin"][0].get("duration_avg_us"))
            or number(total_rows["origin"][0].get("duration_us"))
            if total_rows["origin"] else None
        )

        def check_total(
            name: str, field: str, expected: float, *, count_field: str | None = None,
            expected_count: int | None = None,
        ) -> None:
            rows = total_rows[name]
            if len(rows) != 1:
                return
            actual = number(rows[0].get(field))
            if actual is None or not math.isclose(actual, expected, abs_tol=0.01):
                errors.append(f"{name} total {field} disagrees with data rows")
            if count_field and int(number(rows[0].get(count_field)) or -1) != expected_count:
                errors.append(f"{name} total {count_field} disagrees with data rows")

        check_total("overview", "算子耗时(us)", accumulated_duration)
        check_total("overview", "模块耗时(us)", accumulated_duration)
        check_total("core", "算子耗时(us)", core_duration)
        check_total("auxiliary", "算子耗时(us)", auxiliary_duration)
        check_total(
            "classes", "总耗时(us)", accumulated_duration,
            count_field="算子数量", expected_count=len(overview),
        )
        check_total("stages", "模块耗时(us)", accumulated_duration)
        if total_duration is None or total_duration <= 0:
            errors.append("origin total row needs a positive repeating-unit duration")
        elif not math.isclose(
            number(total_rows["origin"][0].get("duration_avg_pct_of_total")) or -1,
            100.0,
            abs_tol=0.01,
        ):
            errors.append("origin total row percentage must be 100")
    taxonomy = portable_json(
        metadata_root,
        args.taxonomy or manifest.get("architecture_taxonomy"),
        f"{args.prefix}_architecture_taxonomy.json",
    )
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
        required_origin = {"unit_position", "unit_id", "unit_variant"}
        required_view = {"单元位置", "单元ID", "单元类型"}
        if origin_ops and not required_origin.issubset(origin_ops[0]):
            errors.append("heterogeneous repeating unit needs unit fields in origin table")
        if overview and not required_view.issubset(overview[0]):
            errors.append("heterogeneous repeating unit needs unit fields in overview table")
        if stages and not required_view.issubset(stages[0]):
            errors.append("heterogeneous repeating unit needs unit fields in stage table")
        if any(
            not all(row.get(field) for field in required_origin)
            for row in origin_ops
        ):
            errors.append("heterogeneous repeating unit has unscoped origin rows")
        table_variants = {row.get("单元类型") for row in overview if row.get("单元类型")}
        if table_variants != set(variants):
            errors.append(
                f"overview variants {sorted(table_variants)} do not match manifest {sorted(variants)}"
            )
        stage_variants = {row.get("单元类型") for row in stages if row.get("单元类型")}
        if stage_variants != set(variants):
            errors.append(
                f"stage variants {sorted(stage_variants)} do not match manifest {sorted(variants)}"
            )
        positions = {
            str(item.get("position")): item
            for item in composition
            if isinstance(item, dict) and item.get("position") is not None
        }
        if not positions:
            positions = {
                str(index): item
                for index, item in enumerate(composition, 1)
                if isinstance(item, dict)
            }
        for position, item in positions.items():
            rows = [row for row in overview if row.get("单元位置") == position]
            if not rows:
                errors.append(f"composite position {position} has no overview rows")
                continue
            expected_variant = item.get("unit_variant") or item.get("variant")
            if expected_variant and {row.get("单元类型") for row in rows} != {expected_variant}:
                errors.append(f"composite position {position} changes its variant")

    if taxonomy:
        taxonomy_positions = {
            str(item.get("position")): item
            for item in (taxonomy.get("repeating_unit") or {}).get("positions", [])
            if isinstance(item, dict) and item.get("position") is not None
        }
        emitted_positions = {
            row.get("单元位置") for row in overview if row.get("单元位置")
        }
        if emitted_positions != set(taxonomy_positions):
            errors.append(
                "overview positions do not exactly match architecture taxonomy: "
                f"{sorted(emitted_positions)} != {sorted(taxonomy_positions)}"
            )
        for position, definition in taxonomy_positions.items():
            expected_id = str(definition.get("unit_id") or "")
            expected_variant = str(definition.get("unit_variant") or "")
            origin_rows = [
                row for row in origin_ops if row.get("unit_position") == position
            ]
            overview_rows = [
                row for row in overview if row.get("单元位置") == position
            ]
            stage_rows = [
                row for row in stages if row.get("单元位置") == position
            ]
            if not origin_rows or not overview_rows or not stage_rows:
                errors.append(
                    f"taxonomy position {position} must appear in origin, overview and stage tables"
                )
                continue
            if {
                (row.get("unit_id"), row.get("unit_variant"))
                for row in origin_rows
            } != {(expected_id, expected_variant)}:
                errors.append(f"origin identity changes at taxonomy position {position}")
            if {
                (row.get("单元ID"), row.get("单元类型"))
                for row in [*overview_rows, *stage_rows]
            } != {(expected_id, expected_variant)}:
                errors.append(f"human-facing identity changes at taxonomy position {position}")

        taxonomy_variants = {
            item.get("name"): item
            for item in taxonomy.get("variants", [])
            if isinstance(item, dict) and item.get("name")
        }
        for variant, definition in taxonomy_variants.items():
            present_modules = {
                row.get("功能模块")
                for row in stages
                if row.get("单元类型") == variant
            }
            expected_modules = set(definition.get("ordered_functional_modules") or [])
            missing_modules = expected_modules - present_modules
            if missing_modules:
                errors.append(
                    f"variant {variant} misses functional modules: {sorted(missing_modules)}"
                )
            distinctive = set(definition.get("distinctive_functional_modules") or [])
            if len(taxonomy_variants) > 1 and distinctive and not distinctive & present_modules:
                errors.append(f"variant {variant} has no distinguishing module in stage output")

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
        if isinstance(variants, list) and len(variants) > 1:
            frontend_variants = {
                row.get("unitVariant")
                for row in analysis.get("operators", [])
                if row.get("unitVariant")
            }
            if frontend_variants != set(variants):
                errors.append("analysis.json loses heterogeneous unit variants")
            if analysis.get("summary", {}).get("durationLabel") in {
                "单层耗时", "平均单层耗时",
            }:
                errors.append("heterogeneous cycle must not be presented as single-layer duration")

    report = {
        "schema_version": "1.1",
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
