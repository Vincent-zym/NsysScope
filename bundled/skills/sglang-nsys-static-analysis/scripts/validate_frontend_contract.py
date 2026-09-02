#!/usr/bin/env python3
"""Check `analysis.json` against what the NsysScope frontend requires.

`validate_analysis_package.py` cross-checks the numbers -- do the frontend's
classification totals and sample count still match the tables. This checks the
other half: whether the document is *shaped* the way the dashboard needs, which is
what decides between "renders" and "renders wrong". They are separate because a
package can be arithmetically perfect and still, say, collapse a heterogeneous
cycle into one nameless layer, which no total would reveal.

Used both ways: `validate_analysis_package.py --analysis-json` runs these checks
too, and the NsysScope service runs this script as its last gate before marking a
job succeeded. Run it standalone with:

    python scripts/validate_frontend_contract.py /path/result/analysis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


CATEGORIES = {"core", "communication", "auxiliary"}
# Labels that claim "one layer". A cycle holding several layer variants may not use
# them: the number is a multi-layer span and the reader would divide by the wrong
# thing.
SINGLE_LAYER_LABELS = {"单层耗时", "平均单层耗时"}


def contract_errors(payload: Any) -> list[str]:
    """Every way `payload` breaks the frontend contract, in reading order."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["analysis.json must be a JSON object"]
    operators = payload.get("operators")
    if payload.get("schemaVersion") != "1.0" or not isinstance(operators, list) or not operators:
        return ["analysis.json schema or operators are invalid"]
    if any(operator.get("category") not in CATEGORIES for operator in operators):
        errors.append("analysis.json contains an invalid operator category")
    summary = payload.get("summary") or {}
    if summary.get("operatorCount") != len(operators):
        errors.append("analysis.json operatorCount disagrees with operators")
    if not summary.get("devices") or int(summary.get("stableSamples", 0)) < 1:
        errors.append("analysis.json stable sample/device scope is missing")
    if summary.get("heterogeneous"):
        errors.extend(heterogeneous_errors(payload, summary, operators))
    return errors


def heterogeneous_errors(
    payload: dict, summary: dict, operators: list,
) -> list[str]:
    """Checks that only apply to a cycle with more than one layer variant.

    A heterogeneous cycle is the case every simplification breaks: drop the variant
    and the dashboard shows one averaged layer that exists nowhere in the trace.
    """
    errors: list[str] = []
    variants = set(summary.get("distinctUnitVariants") or [])
    operator_variants = {
        operator.get("unitVariant") for operator in operators
        if operator.get("unitVariant")
    }
    if len(variants) < 2 or operator_variants != variants:
        errors.append("analysis.json loses heterogeneous unit variants")
    if any(
        operator.get("unitPosition") is None or not operator.get("unitId")
        for operator in operators
    ):
        errors.append("heterogeneous analysis has unscoped operators")
    if summary.get("durationLabel") in SINGLE_LAYER_LABELS:
        errors.append("heterogeneous cycle is mislabeled as single-layer duration")
    stages = payload.get("stages") or []
    stage_variants = {
        stage.get("unitVariant") for stage in stages if stage.get("unitVariant")
    }
    # A pattern-level rollup is the one legitimate way to drop identity from the
    # stage view: every row is explicitly the whole pattern, not a layer.
    pattern_rollup = bool(stages) and all(
        stage.get("unitId") == "__pattern_total__"
        and stage.get("unitPosition") is None
        and not stage.get("unitVariant")
        for stage in stages
    )
    if not pattern_rollup and (
        stage_variants != variants
        or any(
            stage.get("unitPosition") is None or not stage.get("unitId")
            for stage in stages
        )
    ):
        errors.append("heterogeneous stage view loses structural-unit identity")
    units = payload.get("units") or []
    if len(units) < 2 or {
        unit.get("variant") for unit in units if unit.get("variant")
    } != variants:
        errors.append("heterogeneous analysis needs an explicit structural-unit index")
    positions = [unit.get("position") for unit in units]
    if positions != list(range(1, len(units) + 1)):
        errors.append("structural-unit positions must be contiguous and ordered")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_json", type=Path)
    args = parser.parse_args()
    try:
        payload = json.loads(args.analysis_json.read_text())
    except (OSError, ValueError) as exc:
        raise SystemExit(f"cannot read {args.analysis_json}: {exc}") from exc
    errors = contract_errors(payload)
    for error in errors:
        print(error, file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print(f"frontend contract ok: {args.analysis_json}")


if __name__ == "__main__":
    main()
