#!/usr/bin/env python3
"""Validate seven-table semantics, MFU evidence, manifest, and frontend parity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_frontend_contract import contract_errors  # noqa: E402


SUFFIXES = (
    "_operator_origin_table.csv",
    "_opreator_table.csv",
    "_core_compute_table.csv",
    "_auxiliary_operator_table.csv",
    "_op_classification_table.csv",
    "_stage_table.csv",
)
# The forward-pipeline table is a valuable seventh table when the capture supports
# it (it is the only place the package says what fraction of a forward step the
# measured unit is), but some captures genuinely cannot produce it -- a single
# forward step, no usable step marker, or a schema this analyzer version does not
# yet handle. Treat it as optional so one flaky table does not fail an otherwise
# complete package; validate_forward_pipeline() below still checks it when present.
OPTIONAL_SUFFIXES = (
    "_forward_pipeline_table.csv",
)
CATEGORY_LABELS = {
    "核心计算": "core",
    "通信": "communication",
    "辅助算子": "auxiliary",
}
VOCABULARY_PATH = Path(__file__).resolve().parents[1] / "references/functional-modules.json"
IDENTIFIER_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def load_vocabulary(path: Path) -> set[str]:
    """Every canonical 功能模块 name, across blocks and families.

    Accepts both layouts: schema 2.0 keys stages by architectural block (plus
    cross-block and communication stages), schema 1.x listed them per family.
    """
    if not path.is_file():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return set()
    names = {
        str(stage["name"])
        for block in (data.get("blocks") or {}).values()
        if isinstance(block, dict)
        for stage in (block.get("stages") or [])
        if isinstance(stage, dict) and stage.get("name")
    }
    for group in ("cross_block", "communication"):
        for item in data.get(group) or []:
            if not isinstance(item, dict):
                continue
            if item.get("name"):
                names.add(str(item["name"]))
            names |= {str(alias) for alias in (item.get("deprecated_aliases") or [])}
    # A corrected name keeps its predecessor accepted, so packages produced
    # before the correction do not start reporting drift warnings.
    for block in (data.get("blocks") or {}).values():
        if not isinstance(block, dict):
            continue
        for stage in block.get("stages") or []:
            if isinstance(stage, dict):
                names |= {
                    str(alias) for alias in (stage.get("deprecated_aliases") or [])
                }
    names |= {
        str(name)
        for family in (data.get("families") or {}).values()
        if isinstance(family, dict)
        for name in (family.get("modules") or [])
    }
    return names


def load_vocabulary_index(path: Path) -> dict[str, Any]:
    """Canonical name -> block, plus alias -> canonical and the family lists.

    load_vocabulary() answers "is this name allowed"; this answers the two
    questions that decide whether two runs are comparable: which block a stage
    belongs to, and which stages the family says a layer should have.
    """
    empty: dict[str, Any] = {"block_of": {}, "alias_to_canonical": {}, "families": {}}
    if not path.is_file():
        return empty
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return empty
    block_of: dict[str, str] = {}
    alias_to_canonical: dict[str, str] = {}
    for block, body in (data.get("blocks") or {}).items():
        if not isinstance(body, dict):
            continue
        for stage in body.get("stages") or []:
            if not isinstance(stage, dict) or not stage.get("name"):
                continue
            name = str(stage["name"])
            block_of[name] = str(block)
            for alias in stage.get("deprecated_aliases") or []:
                alias_to_canonical[str(alias)] = name
                block_of.setdefault(str(alias), str(block))
    for group in ("cross_block", "communication"):
        for item in data.get(group) or []:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            name = str(item["name"])
            block_of[name] = str(group)
            for alias in item.get("deprecated_aliases") or []:
                alias_to_canonical[str(alias)] = name
                block_of.setdefault(str(alias), str(group))
    families = {
        str(fid): [str(n) for n in (body.get("typical_layer") or [])]
        for fid, body in (data.get("families") or {}).items()
        if isinstance(body, dict)
    }
    return {
        "block_of": block_of,
        "alias_to_canonical": alias_to_canonical,
        "families": families,
    }


def check_naming_stability(
    origin_ops: list[dict[str, str]],
    stages: list[dict[str, str]],
    vocabulary: set[str],
    warnings: list[str],
) -> None:
    """Report names that only one run of the analysis would ever produce.

    Both checks are warnings, not errors: a package whose numbers are right should
    ship. But invented names make two analyses of the same model incomparable --
    per-module timing cannot be diffed if the modules are renamed every run -- so
    the drift has to be visible rather than silent.
    """
    if vocabulary:
        used = {row.get("功能模块") for row in stages if row.get("功能模块")}
        unknown = sorted(name for name in used if name not in vocabulary)
        if unknown:
            warnings.append(
                "功能模块 outside references/functional-modules.json: "
                f"{unknown}. Map them onto canonical names, or record the new names "
                "and the reason in the taxonomy's functional_module_policy."
            )
    # A `module` leaf is supposed to be lifted from the source -- an attribute or the
    # dispatching function -- not composed as prose. When none of the leaf's words
    # appear anywhere in the row's own evidence, it was probably invented, and the
    # next run will invent a different one. Matching is per word and by substring on
    # purpose: `pre_norm_quant` legitimately abbreviates
    # `fused_add_rmsnorm_per_token_quant`, and demanding a whole-identifier match
    # would flag every such abbreviation.
    unsupported = []
    for row in origin_ops:
        module = row.get("module") or ""
        leaf = module.rsplit("/", 1)[-1]
        words = [part for part in re.split(r"[^A-Za-z0-9]+", leaf) if len(part) > 2]
        if not words:
            continue
        evidence = " ".join((
            row.get("python_function") or "",
            row.get("dispatch_code_snippet") or "",
            row.get("operator_name") or "",
        )).lower()
        if not any(word.lower() in evidence for word in words):
            unsupported.append(module)
    if unsupported:
        sample = sorted(set(unsupported))[:8]
        warnings.append(
            f"{len(unsupported)} rows have a module leaf that appears in no cited "
            f"evidence (call chain, dispatch statement, kernel name), e.g. {sample}. "
            "Derive the leaf from the source symbol so the label is reproducible."
        )


def check_stage_composition(
    stages: list[dict[str, str]],
    index: dict[str, Any],
    taxonomy: dict[str, Any] | None,
    warnings: list[str],
) -> None:
    """Report drift in *which* stages were emitted, not just how they are spelled.

    Two runs can both use only canonical names and still be incomparable: one
    splits MoE into 计算 + 输出合并, the other emits the fused 计算与输出合并.
    Spelling is checked elsewhere; here the emitted set is compared against the
    family's typical_layer, the per-variant stage count against the policy
    window, and each stage against the ~2%-of-its-block fold rule.
    """
    block_of = index.get("block_of") or {}
    if not block_of:
        return
    alias_to_canonical = index.get("alias_to_canonical") or {}
    policy = (taxonomy or {}).get("functional_module_policy") or {}

    rows = [
        row for row in stages
        if str(row.get("单元位置") or "").isdigit() and row.get("功能模块")
    ]
    used = {str(row["功能模块"]) for row in rows}
    stale = sorted(name for name in used if name in alias_to_canonical)
    if stale:
        pairs = ", ".join(f"{name} -> {alias_to_canonical[name]}" for name in stale)
        warnings.append(
            f"功能模块 uses deprecated names: {pairs}. They still validate so old "
            "packages keep working, but a new package should emit the canonical name."
        )

    per_position: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        per_position[str(row["单元位置"])].append(row)
    lo = number(policy.get("target_min")) or 5
    hi = number(policy.get("target_max")) or 8
    off_target = {
        pos: len(items) for pos, items in per_position.items()
        if not lo <= len(items) <= hi
    }
    if off_target:
        warnings.append(
            f"stage count per 单元位置 outside the {int(lo)}-{int(hi)} target: "
            f"{dict(sorted(off_target.items()))}. Fold or split per the vocabulary "
            "rules so two runs of this model land on the same granularity."
        )

    thin = []
    for pos, items in sorted(per_position.items()):
        by_block: dict[str, float] = defaultdict(float)
        for row in items:
            block = block_of.get(str(row["功能模块"]))
            if block is None:
                continue  # unknown name: already reported, block share is meaningless
            by_block[block] += number(row.get("模块耗时(us)")) or 0.0
        for row in items:
            block = block_of.get(str(row["功能模块"]))
            if block is None or block in {"cross_block", "communication"}:
                continue
            total = by_block[block]
            duration = number(row.get("模块耗时(us)")) or 0.0
            if total > 0 and duration / total < 0.02:
                thin.append(
                    f"{row['功能模块']}@{pos} ({duration / total * 100:.1f}% of {block})"
                )
    if thin:
        warnings.append(
            f"stages below ~2% of their block: {thin[:8]}. The vocabulary folds "
            "these into the block's 计算 stage -- a 0.4% stage is a `module`, not a "
            "functional module."
        )

    families = index.get("families") or {}
    blob = json.dumps(policy, ensure_ascii=False)
    declared = [fid for fid in families if fid and fid in blob]
    if len(declared) == 1:
        expected = set(families[declared[0]])
        canonical_used = {alias_to_canonical.get(name, name) for name in used}
        missing = sorted(expected - canonical_used)
        extra = sorted(canonical_used - expected)
        if missing or extra:
            warnings.append(
                f"emitted 功能模块 set differs from family '{declared[0]}' "
                f"typical_layer: missing={missing} extra={extra}. Either emit the "
                "family's stages, or update the family's typical_layer and say why "
                "in functional_module_policy -- an undocumented difference means the "
                "next run may pick either shape."
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("package", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--analysis-json", type=Path)
    parser.add_argument("--taxonomy", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--vocabulary", type=Path, default=VOCABULARY_PATH,
        help="canonical 功能模块 list; naming drift is reported as a warning",
    )
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


FORWARD_PIPELINE_FIELDS = (
    "环节", "环节类型", "层数", "子步数", "单次耗时(us)", "总耗时(us)",
    "占forward步(%)", "占父环节(%)", "样本数", "min_us", "max_us", "备注",
)
FORWARD_PIPELINE_TYPES = {"total", "phase", "variant", "stage", "other", "gap"}


def validate_forward_pipeline(path: Path, errors: list[str]) -> None:
    """Check the optional forward-pipeline table's structure and closures.

    The closures are the correctness test for step segmentation: a boundary marker
    that fires at the wrong place shows up here as a sum that does not add up.
    """
    if not path.is_file():
        return
    rows = read_csv(path)
    fields = read_fields(path)
    if list(fields) != list(FORWARD_PIPELINE_FIELDS):
        errors.append(
            f"forward pipeline columns must be exactly {list(FORWARD_PIPELINE_FIELDS)}"
        )
        return
    if not rows:
        errors.append("forward pipeline table has no rows")
        return

    bad = sorted({r.get("环节类型", "") for r in rows} - FORWARD_PIPELINE_TYPES)
    if bad:
        errors.append(f"forward pipeline has unknown 环节类型: {bad}")
        return

    totals = [r for r in rows if r["环节类型"] == "total"]
    if len(totals) != 1:
        errors.append("forward pipeline needs exactly one total row")
        return
    step_us = number(totals[0].get("总耗时(us)"))
    if not step_us or step_us <= 0:
        errors.append("forward pipeline total row needs a positive 总耗时(us)")
        return
    tol = step_us * 0.005

    # Nesting is positional: rows after a phase row belong to it until the next one.
    phases: list[tuple[dict[str, str], list[dict[str, str]]]] = []
    for row in rows:
        kind = row["环节类型"]
        if kind == "phase":
            phases.append((row, []))
        elif kind in {"variant", "stage", "other"}:
            if not phases:
                errors.append(f"forward pipeline row '{row['环节']}' has no parent phase")
                return
            phases[-1][1].append(row)
    if not phases:
        errors.append("forward pipeline needs at least one phase row")
        return

    phase_sum = 0.0
    for phase, children in phases:
        value = number(phase.get("总耗时(us)"))
        if value is None:
            errors.append(f"forward pipeline phase '{phase['环节']}' has no 总耗时(us)")
            return
        phase_sum += value
        if not children:
            continue
        child_sum = 0.0
        for child in children:
            cv = number(child.get("总耗时(us)"))
            if cv is None:
                errors.append(
                    f"forward pipeline row '{child['环节']}' has no 总耗时(us)"
                )
                return
            child_sum += cv
        if abs(child_sum - value) > tol:
            errors.append(
                f"forward pipeline phase '{phase['环节']}' does not close: "
                f"children {child_sum:.1f}us vs phase {value:.1f}us "
                f"(tolerance {tol:.1f}us) -- re-derive the boundaries instead of "
                f"adjusting the 其他 row"
            )

    # A step is its phases plus the idle gap between them:
    #     forward step = target [+ draft] + 步间间隙
    # The gap is a sibling of the phases, not a shadow number excluded from the sum.
    # It used to be carved out of the target phase's prep window *and* folded into it,
    # which made that phase equal the whole step at 100% and hid the gap from closure.
    gap_rows = [r for r in rows if r["环节类型"] == "gap"]
    gap_sum = 0.0
    for row in gap_rows:
        gap = number(row.get("总耗时(us)"))
        if gap is None:
            errors.append("forward pipeline gap row has no 总耗时(us)")
            continue
        if gap < 0:
            errors.append(f"forward pipeline gap row '{row['环节']}' is negative: {gap:.1f}us")
        gap_sum += gap
        if not row.get("备注"):
            errors.append(
                "forward pipeline gap row needs 备注 stating the threshold and hole "
                "count, so a 0.0 result is distinguishable from 'not measured'"
            )

    if abs(phase_sum + gap_sum - step_us) > tol:
        errors.append(
            f"forward pipeline does not close: phases {phase_sum:.1f}us + gap "
            f"{gap_sum:.1f}us = {phase_sum + gap_sum:.1f}us vs step {step_us:.1f}us "
            f"(tolerance {tol:.1f}us) -- a step must equal target [+ draft] + gap"
        )


CJK = re.compile(r"[\u4e00-\u9fff]")
CALL_SITE = re.compile(r"\.py:\d+")
# PP activation handoff: a rank's wait for its neighbour, never part of a layer
PIPELINE_HANDOFF = re.compile(r"SendRecv|Recv|Send", re.IGNORECASE)


def validate_evidence_depth(
    origin_ops: list[dict[str, str]], core: list[dict[str, str]],
    errors: list[str],
) -> None:
    """Reject a package whose evidence columns were filled with boilerplate.

    Failing to match the supplied source tree to the captured build commit is not a
    licence to drop call sites, code snippets and GEMM shapes: the commit only
    decides whether source *defaults* may be quoted as runtime truth. A real package
    cites `file:line` for ~80-100% of rows; the run that triggered this gate cited 0.
    """
    if not origin_ops:
        return
    total = len(origin_ops)

    def ratio(count: int) -> str:
        return f"{count}/{total}"

    cited = sum(
        1 for row in origin_ops if CALL_SITE.search(row.get("python_function", ""))
    )
    if cited < total * 0.6:
        errors.append(
            f"only {ratio(cited)} origin rows cite a file:line call site; an unverified "
            "source commit does not remove the call-site requirement"
        )
    described = sum(
        1 for row in origin_ops if CJK.search(row.get("function_introduction", ""))
    )
    if described < total * 0.6:
        errors.append(
            f"only {ratio(described)} origin rows have a Chinese function_introduction; "
            "a single English noun phrase is not a functional description"
        )
    if "dispatch_code_snippet" in origin_ops[0]:
        coded = sum(
            1 for row in origin_ops if "(" in row.get("dispatch_code_snippet", "")
        )
        if coded < total * 0.6:
            errors.append(
                f"only {ratio(coded)} origin rows carry a real dispatch code snippet "
                "(prose in this column is not evidence)"
            )
    if core and not any(row.get("shape") for row in core):
        errors.append(
            "no core-compute row has a shape: GEMM shape/mfu/mbu cannot all be blank "
            "when config and launch material provide token count and dimensions"
        )
    validate_efficiency_bounds(core, errors)


def validate_efficiency_bounds(
    core: list[dict[str, str]], errors: list[str],
) -> None:
    """Reject MFU/MBU percentages that claim a kernel beat the hardware.

    MFU already had this guard by convention; MBU never did, and a package
    shipped 134% MBU on a skinny GEMM whose bf16 activation had been costed as
    fp32. Counting every operand element once is a *lower* bound on HBM traffic
    -- tiling only re-reads, and cache reuse pushes real traffic down toward
    that bound -- so above 100% is arithmetically impossible, not a
    cache-friendly shape. Blank is the correct output when the inputs are
    unknown; an impossible number is not.
    """
    for column in ("mfu", "mbu"):
        offenders = []
        for row in core:
            raw = str(row.get(column) or "").strip().rstrip("%")
            if not raw:
                continue
            try:
                value = float(raw)
            except ValueError:
                errors.append(
                    f"core-compute {column} is not a percentage: {row.get(column)!r}"
                )
                continue
            if value > 100.0 or value < 0.0:
                offenders.append(
                    f"{row.get('算子名称', '?')} {row.get('shape', '')} = {value:.2f}%"
                )
        if offenders:
            errors.append(
                f"{len(offenders)} core-compute row(s) report an impossible {column} "
                f"(outside 0-100%): {'; '.join(offenders[:3])}"
                + (" …" if len(offenders) > 3 else "")
                + f"; leave {column} blank and record the reason instead"
            )


def validate_unit_attribution(
    origin_ops: list[dict[str, str]], unit_duration: float | None,
    errors: list[str],
) -> None:
    """Catch operators attributed to a layer they cannot belong to.

    Two impossibilities were observed on a package that passed every other check: a
    400ms pipeline-parallel `SendRecv` wait counted inside a 102ms layer window, and
    a first position truncated to 32 kernels while its siblings of the same variant
    had 41-42. Both mean the unit window is misaligned with the real layer boundary.

    Communication kernels are not banned from a layer -- TP all-reduce and DCP
    all-to-all legitimately run inside one, at a few percent of its span. What cannot
    belong to a layer is a rank-level wait that dwarfs it.
    """
    if not origin_ops:
        return
    for row in origin_ops:
        duration = number(row.get("duration_us"))
        if not (unit_duration and duration):
            continue
        if duration > unit_duration * 1.001:
            errors.append(
                f"operator {row.get('operator_name', '')[:60]} lasts {duration:.1f}us, "
                f"longer than the whole repeating unit ({unit_duration:.1f}us): it is "
                "not inside this unit"
            )
        elif (
            row.get("unit_position")
            and PIPELINE_HANDOFF.search(row.get("operator_name", ""))
            and duration > unit_duration * 0.2
        ):
            errors.append(
                f"handoff kernel {row.get('operator_name', '')[:60]} takes "
                f"{duration / unit_duration:.0%} of unit position "
                f"{row.get('unit_position')}; that is a rank-level wait, not layer work"
            )

    per_position: Counter[tuple[str, str]] = Counter(
        (row.get("unit_variant", ""), row.get("unit_position", ""))
        for row in origin_ops
        if row.get("unit_position")
    )
    by_variant: dict[str, list[int]] = {}
    for (variant, _), count in per_position.items():
        by_variant.setdefault(variant, []).append(count)
    for variant, counts in by_variant.items():
        if len(counts) > 1 and max(counts) > min(counts) * 1.15:
            errors.append(
                f"variant {variant} positions hold {min(counts)}-{max(counts)} "
                "operators; positions of one variant execute the same kernels, so the "
                "unit window is phase-shifted against the layer boundary"
            )


def validate_sampling(manifest: dict[str, Any], errors: list[str]) -> None:
    """A single occurrence cannot represent a repeating unit."""
    if not manifest:
        errors.append(
            "<prefix>_analysis_manifest.json was not found next to the tables or "
            "in metadata/: sampling, unit composition and device selection cannot be "
            "verified, and the frontend then falls back to 1 sample / 1 unit"
        )
        return
    stable = manifest.get("stable_statistics") or manifest.get("stable_stats") or {}
    if not isinstance(stable, dict):
        return
    if stable.get("single_sample_fallback"):
        errors.append(
            "stable_statistics.single_sample_fallback is set: one occurrence cannot "
            "represent the repeating unit (per-step spans vary by up to 2x on a "
            "serving capture)"
        )
    accepted = (
        stable.get("accepted_sample_count")
        or stable.get("accepted_full_template_sample_count")
    )
    if accepted is not None and int(accepted) < 3:
        errors.append(
            f"only {accepted} unit occurrence(s) accepted; sample at least 3 from the "
            "steady state, and never the capture's first forward"
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
    missing_optional = [
        f"{args.prefix}{suffix}" for suffix in OPTIONAL_SUFFIXES
        if not (root / f"{args.prefix}{suffix}").is_file()
    ]
    if missing_optional:
        print(f"[validate] optional tables absent, continuing: {missing_optional}")

    paths = {
        "origin": root / f"{args.prefix}_operator_origin_table.csv",
        "overview": root / f"{args.prefix}_opreator_table.csv",
        "core": root / f"{args.prefix}_core_compute_table.csv",
        "auxiliary": root / f"{args.prefix}_auxiliary_operator_table.csv",
        "classes": root / f"{args.prefix}_op_classification_table.csv",
        "stages": root / f"{args.prefix}_stage_table.csv",
    }
    # The seventh table is required, and its presence is already enforced above.
    validate_forward_pipeline(
        root / f"{args.prefix}_forward_pipeline_table.csv", errors,
    )
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
        row_count = number(row.get("算子数量"))
        if row_count is None or int(row_count) != totals[category]["count"]:
            errors.append(f"{row['算子类型']} count disagrees with operator membership")
        row_duration = number(row.get("总耗时(us)"))
        if row_duration is None or not math.isclose(
            row_duration,
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

    # The tables can sit in csv/, in result/ or in the job dir itself -- the layout is
    # the agent's choice -- so look for the metadata dir instead of inferring it from
    # the table dir's name, which made every metadata-derived check silently vacuous
    # for any layout that was not csv/.
    metadata_candidates = (root / "metadata", root.parent / "metadata")
    metadata_root = next(
        (path for path in metadata_candidates if (path / "context.json").is_file()),
        next((path for path in metadata_candidates if path.is_dir()), root),
    )
    manifest_name = f"{args.prefix}_analysis_manifest.json"
    # Accept the manifest either in metadata/ or beside the tables; the converter
    # reads it the same way, and an unread manifest silently drops the unit
    # composition and the sample count from the frontend.
    manifest_path = next(
        (path for path in (metadata_root / manifest_name, root / manifest_name)
         if path.exists()),
        metadata_root / manifest_name,
    )
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
            if count_field:
                actual_count = number(rows[0].get(count_field))
                if actual_count is None or int(actual_count) != expected_count:
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

    validate_evidence_depth(origin_ops, core, errors)
    warnings: list[str] = []
    # A forward-pipeline table whose layer counts disagree with config.json is still a
    # valid package -- the step total, the phase split and the gap are measured the same
    # way. Surface it as a warning so the caveat travels with the package instead of
    # costing it the seventh table.
    forward_sidecar = metadata_root / f"{args.prefix}_forward_pipeline.json"
    if forward_sidecar.is_file():
        try:
            fragment = json.loads(forward_sidecar.read_text()).get("forward_pipeline") or {}
        except (json.JSONDecodeError, OSError):
            fragment = {}
        for conflict in fragment.get("declaration_conflicts") or []:
            warnings.append(f"forward pipeline vs config/launch declaration: {conflict}")
    check_naming_stability(
        origin_ops, stages, load_vocabulary(args.vocabulary), warnings,
    )
    check_stage_composition(
        stages, load_vocabulary_index(args.vocabulary), taxonomy, warnings,
    )
    unit_duration = (
        number(total_rows["origin"][0].get("duration_avg_us"))
        or number(total_rows["origin"][0].get("duration_us"))
        if total_rows["origin"] else None
    )
    validate_unit_attribution(origin_ops, unit_duration, errors)
    validate_sampling(manifest, errors)

    analysis = None
    if args.analysis_json and args.analysis_json.exists():
        analysis = json.loads(args.analysis_json.read_text())
        # Shape first: the numeric cross-checks below assume a document the frontend
        # can actually render.
        errors.extend(contract_errors(analysis))
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

    # The frontend-contract check and the taxonomy cross-check can reach the same
    # conclusion from different evidence; report it once.
    errors = list(dict.fromkeys(errors))
    report = {
        "schema_version": "1.1",
        "status": "failed" if errors else "passed",
        "errors": errors,
        "warnings": warnings,
        "operator_count": len(overview),
        "category_totals": totals,
        "analysis_json_checked": analysis is not None,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if errors:
        raise SystemExit("\n".join(errors))
    for warning in warnings:
        print(f"[validate] warning: {warning}")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
