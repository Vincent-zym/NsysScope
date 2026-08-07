#!/usr/bin/env python3
"""Validate optimization.json against the sglang-operator-fusion-advisor schema."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CONFIDENCE_VALUES = {"high", "medium", "low"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("optimization_json", type=Path)
    parser.add_argument("--analysis-json", type=Path, default=None)
    return parser.parse_args()


def fail(errors: list[str], message: str) -> None:
    errors.append(message)


def validate(payload: dict[str, Any], analysis: dict[str, Any] | None) -> list[str]:
    errors: list[str] = []
    if payload.get("schemaVersion") != "1.0":
        fail(errors, "schemaVersion must be '1.0'")
    if not isinstance(payload.get("scope"), dict):
        fail(errors, "scope must be an object")
    suggestions = payload.get("suggestions")
    if not isinstance(suggestions, list):
        fail(errors, "suggestions must be a list")
        return errors

    valid_indices: set[int] | None = None
    if analysis is not None:
        operators = analysis.get("operators") or []
        valid_indices = {
            int(op["index"]) for op in operators if "index" in op
        }

    seen_ids: set[str] = set()
    for i, suggestion in enumerate(suggestions):
        label = f"suggestions[{i}]"
        if not isinstance(suggestion, dict):
            fail(errors, f"{label} must be an object")
            continue
        sid = suggestion.get("id")
        if not sid or not isinstance(sid, str):
            fail(errors, f"{label}.id must be a non-empty string")
        elif sid in seen_ids:
            fail(errors, f"{label}.id '{sid}' is not unique")
        else:
            seen_ids.add(sid)

        targets = suggestion.get("targetOperators")
        if not isinstance(targets, list) or not targets:
            fail(errors, f"{label}.targetOperators must be a non-empty list")
        elif valid_indices is not None:
            missing = [t for t in targets if int(t) not in valid_indices]
            if missing:
                fail(
                    errors,
                    f"{label}.targetOperators references indices not in "
                    f"analysis.json operators: {missing}",
                )

        options = suggestion.get("options")
        if not isinstance(options, list) or not (1 <= len(options) <= 3):
            fail(errors, f"{label}.options must contain between 1 and 3 entries")
            continue

        gains: list[float] = []
        group_duration = suggestion.get("groupDurationUs")
        for j, option in enumerate(options):
            olabel = f"{label}.options[{j}]"
            if not isinstance(option, dict):
                fail(errors, f"{olabel} must be an object")
                continue
            if not option.get("approach"):
                fail(errors, f"{olabel}.approach must be non-empty")
            if not option.get("rationale"):
                fail(errors, f"{olabel}.rationale must be non-empty")
            if not option.get("estimatedGainBasis"):
                fail(errors, f"{olabel}.estimatedGainBasis must be non-empty")
            gain = option.get("estimatedGainPct")
            if not isinstance(gain, (int, float)) or not (0 <= gain <= 100):
                fail(errors, f"{olabel}.estimatedGainPct must be a number in [0, 100]")
            else:
                gains.append(float(gain))
            confidence = option.get("confidence")
            if confidence not in CONFIDENCE_VALUES:
                fail(
                    errors,
                    f"{olabel}.confidence must be one of {sorted(CONFIDENCE_VALUES)}",
                )
            if not isinstance(option.get("referenceLinks", []), list):
                fail(errors, f"{olabel}.referenceLinks must be a list")

        if gains != sorted(gains, reverse=True):
            fail(errors, f"{label}.options must be sorted by estimatedGainPct descending")

        if (
            isinstance(group_duration, (int, float))
            and group_duration > 0
            and any(g > 100 for g in gains)
        ):
            fail(errors, f"{label}: estimatedGainPct cannot exceed 100")

    return errors


def main() -> None:
    args = parse_args()
    payload = json.loads(args.optimization_json.read_text())
    analysis = None
    if args.analysis_json and args.analysis_json.exists():
        analysis = json.loads(args.analysis_json.read_text())
    errors = validate(payload, analysis)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
    print("optimization.json is valid")


if __name__ == "__main__":
    main()
