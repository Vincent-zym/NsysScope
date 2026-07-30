#!/usr/bin/env python3
"""Validate a model-specific architecture taxonomy before table generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("taxonomy must be a JSON object")
    return value


def validate(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if data.get("schema_version") != "1.0":
        errors.append("schema_version must be 1.0")

    evidence = data.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append("evidence must contain current-model design/config/source paths")
    else:
        for index, item in enumerate(evidence, 1):
            if not isinstance(item, dict) or not item.get("kind") or not item.get("path"):
                errors.append(f"evidence[{index}] needs kind and path")

    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        errors.append("variants must be a non-empty list")
        variants = []
    names = [
        item.get("name") for item in variants
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    ]
    if len(names) != len(variants) or len(set(names)) != len(names):
        errors.append("variant names must be non-empty and unique")
    declared = set(names)

    repeating = data.get("repeating_unit")
    positions = repeating.get("positions") if isinstance(repeating, dict) else None
    if not isinstance(positions, list) or not positions:
        errors.append("repeating_unit.positions must be a non-empty list")
        positions = []
    seen: set[int] = set()
    seen_unit_ids: set[str] = set()
    for index, item in enumerate(positions, 1):
        if not isinstance(item, dict):
            errors.append(f"position[{index}] must be an object")
            continue
        position = item.get("position")
        if not isinstance(position, int) or position <= 0 or position in seen:
            errors.append(f"position[{index}] needs a unique positive position")
        else:
            seen.add(position)
        unit_id = item.get("unit_id")
        if not unit_id:
            errors.append(f"position[{index}] needs unit_id")
        elif str(unit_id) in seen_unit_ids:
            errors.append(f"position[{index}] unit_id must be unique")
        else:
            seen_unit_ids.add(str(unit_id))
        if item.get("unit_variant") not in declared:
            errors.append(f"position[{index}] references an undeclared unit_variant")
        if item.get("layer_id") in (None, "") and not item.get("module_regex"):
            errors.append(f"position[{index}] needs layer_id or module_regex")
    if seen and seen != set(range(1, len(positions) + 1)):
        errors.append("positions must be contiguous from 1")

    heterogeneous = len(declared) > 1
    policy = data.get("functional_module_policy") or {}
    if not isinstance(policy, dict):
        errors.append("functional_module_policy must be an object")
        policy = {}
    target_min = policy.get("target_min", 5)
    target_max = policy.get("target_max", 8)
    if (
        not isinstance(target_min, int)
        or not isinstance(target_max, int)
        or target_min <= 0
        or target_max < target_min
        or target_max > 8
    ):
        errors.append(
            "functional_module_policy target_min/target_max must be positive "
            "integers with target_max >= target_min and target_max <= 8"
        )
        target_max = 8
    for index, item in enumerate(variants, 1):
        if not isinstance(item, dict):
            continue
        if not item.get("source_evidence"):
            errors.append(f"variant[{index}] needs source_evidence")
        modules = item.get("ordered_functional_modules")
        if not isinstance(modules, list) or not modules:
            errors.append(f"variant[{index}] needs ordered_functional_modules")
        elif (
            any(not isinstance(module, str) or not module.strip() for module in modules)
            or len(set(modules)) != len(modules)
        ):
            errors.append(
                f"variant[{index}] ordered_functional_modules must be unique non-empty strings"
            )
        elif len(modules) > target_max and not str(
            item.get("granularity_exception") or ""
        ).strip():
            errors.append(
                f"variant[{index}] has {len(modules)} functional modules; more than "
                f"{target_max} requires a current-model granularity_exception"
            )
        discriminators = item.get("discriminators")
        if heterogeneous and (
            not isinstance(discriminators, list) or not discriminators
        ):
            errors.append(f"variant[{index}] needs discriminators in a heterogeneous cycle")

    declared_modules = {
        module
        for item in variants
        if isinstance(item, dict)
        for module in (item.get("ordered_functional_modules") or [])
        if isinstance(module, str)
    }
    for index, group in enumerate(data.get("fusion_groups") or [], 1):
        if not isinstance(group, dict):
            errors.append(f"fusion_group[{index}] must be an object")
            continue
        if not group.get("name") or len(group.get("logical_owners") or []) < 2:
            errors.append(f"fusion_group[{index}] needs name and at least two logical_owners")
        if group.get("attribution_policy") != "indivisible":
            errors.append(
                f"fusion_group[{index}] attribution_policy must be indivisible"
            )
        functional_module = group.get("functional_module") or group.get("name")
        if functional_module and functional_module not in declared_modules:
            errors.append(
                f"fusion_group[{index}] functional_module must be an ordered "
                "functional module"
            )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("taxonomy", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = load(args.taxonomy)
    errors = validate(data)
    report = {
        "schema_version": "1.0",
        "status": "failed" if errors else "passed",
        "errors": errors,
        "variant_count": len(data.get("variants") or []),
        "position_count": len((data.get("repeating_unit") or {}).get("positions") or []),
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise SystemExit("\n".join(errors))
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
