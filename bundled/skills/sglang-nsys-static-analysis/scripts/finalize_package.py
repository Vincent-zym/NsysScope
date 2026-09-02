#!/usr/bin/env python3
"""Lay a finished analysis out as a portable NsysScope package.

The analysis itself is easier to write flat: one directory, tables and sidecars
side by side. What a reader (and the NsysScope frontend) wants is the canonical
layout, with the seven tables in `csv/`, workbooks in `xlsx/`, every sidecar in
`metadata/`, the trace in `trace/` and a manifest at the root:

    result/
    |-- analysis.json
    |-- final_report.md
    |-- nsysscope-package.json
    |-- csv/       <prefix>_*_table.csv
    |-- xlsx/      one workbook per table
    |-- metadata/  taxonomy, manifest, semantic map, statistics, validation
    |-- trace/     the exported .sqlite
    `-- logs/      job.log, when a job produced one

This script performs that last step, and it is the only implementation: the
NsysScope service calls it instead of packaging on its own, so a Skill run done
by hand and a job run through the tool end in the same directory. Idempotent by
construction -- running it twice, or on an already-canonical package, changes
nothing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from csv_to_xlsx import convert_directory  # noqa: E402
from validate_frontend_contract import contract_errors  # noqa: E402


AGENT_TABLE_SUFFIXES = (
    "_operator_origin_table.csv",
    "_opreator_table.csv",
    "_core_compute_table.csv",
    "_auxiliary_operator_table.csv",
    "_op_classification_table.csv",
    "_stage_table.csv",
)
# The seventh table relates the measured unit to a whole forward step. It is
# optional on purpose: a capture with a single forward step cannot produce it, and
# that must not cost the package its other six tables.
FORWARD_PIPELINE_SUFFIX = "_forward_pipeline_table.csv"
TABLE_SUFFIXES = AGENT_TABLE_SUFFIXES + (FORWARD_PIPELINE_SUFFIX,)

# Written by the analysis, read by the frontend: they belong at the package root,
# not in metadata/.
ROOT_JSON = {"analysis.json", "nsysscope-package.json"}


def complete(directory: Path, prefix: str) -> bool:
    """Whether `directory` holds all six mandatory tables for `prefix`."""
    return all(
        (directory / f"{prefix}{suffix}").is_file() for suffix in AGENT_TABLE_SUFFIXES
    )


def locate_tables(result_dir: Path, prefix: str) -> Path:
    """The directory the six tables are in right now.

    Checked in the order a package is likely to be in: already canonical, flat in
    the result directory, or somewhere below it (an agent picks its own working
    directory and only the tables identify it).
    """
    for candidate in (result_dir / "csv", result_dir):
        if complete(candidate, prefix):
            return candidate
    for marker in sorted(result_dir.rglob(f"{prefix}_stage_table.csv")):
        if complete(marker.parent, prefix):
            return marker.parent
    raise SystemExit(
        f"cannot find the six tables for prefix {prefix!r} under {result_dir}"
    )


def move_into(source: Path, target_dir: Path) -> None:
    """Move one file into `target_dir`, tolerating "already there"."""
    target = target_dir / source.name
    if source.resolve() == target.resolve():
        return
    shutil.move(str(source), target)


def collect_tables(tables_dir: Path, csv_dir: Path, prefix: str) -> list[str]:
    names = []
    for suffix in TABLE_SUFFIXES:
        source = tables_dir / f"{prefix}{suffix}"
        target = csv_dir / source.name
        if not source.is_file() and not target.is_file():
            if suffix == FORWARD_PIPELINE_SUFFIX:
                continue
            raise SystemExit(f"package is missing {source.name}")
        if source.is_file():
            move_into(source, csv_dir)
        names.append(target.name)
    return names


def sweep_sidecars(
    search_dirs: list[Path], csv_dir: Path, metadata_dir: Path,
) -> None:
    """Everything that is not a canonical table or a root JSON goes to metadata/.

    Scratch CSVs an analysis leaves behind are evidence worth keeping, but they
    must not sit next to the seven tables: a reader cannot tell them apart, and
    `csv/` is a contract. The canonical tables have already been moved out by
    then, so whatever is left is by definition not one of them.
    """
    for source_dir in dict.fromkeys(search_dirs):
        if source_dir.resolve() == csv_dir.resolve():
            continue
        if not source_dir.is_dir():
            continue
        for extra in sorted(source_dir.glob("*.csv")):
            move_into(extra, metadata_dir)
        for sidecar in sorted(source_dir.glob("*.json")):
            if sidecar.name in ROOT_JSON:
                continue
            move_into(sidecar, metadata_dir)


def place_trace(trace: Path, result_dir: Path, trace_dir: Path) -> Path:
    """Put the exported trace under trace/, moving it when it is already ours."""
    target = trace_dir / trace.name
    if trace.resolve() == target.resolve():
        return target
    trace_dir.mkdir(exist_ok=True)
    if trace.is_relative_to(result_dir):
        shutil.move(str(trace), target)
    else:
        shutil.copy2(trace, target)
    return target


def write_manifest(
    result_dir: Path, prefix: str, tables: list[str], trace: Path | None,
) -> Path:
    manifest = {
        "schemaVersion": "1.0",
        "kind": "nsysscope-analysis-package",
        "analysis": "analysis.json",
        "csvDirectory": "csv",
        "xlsxDirectory": "xlsx",
    }
    if trace is not None:
        manifest["trace"] = f"trace/{trace.name}"
    # Only a job run has a log; a Skill run by hand must not advertise one that
    # does not exist.
    if (result_dir / "logs" / "job.log").is_file():
        manifest["log"] = "logs/job.log"
    manifest["metadataDirectory"] = "metadata"
    manifest["prefix"] = prefix
    manifest["tables"] = tables
    path = result_dir / "nsysscope-package.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    return path


def describe(result_dir: Path) -> dict[str, str]:
    """Model/stage/hardware for a rebuild, from whatever recorded them.

    `analysis.json` carries them in `metadata`; a job also writes them into
    `metadata/context.json` under the request's own names. Missing values are
    passed as empty, exactly as a first conversion without them would.
    """
    for path, keys in (
        (result_dir / "analysis.json", ("model", "stage", "hardware")),
        (result_dir / "metadata" / "context.json", ("model_name", "stage", "hardware")),
    ):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        source = payload.get("metadata") if "metadata" in payload else payload
        if not isinstance(source, dict):
            continue
        found = {
            name: str(source.get(key) or "")
            for name, key in zip(("model", "stage", "hardware"), keys)
        }
        if any(found.values()):
            return found
    return {"model": "", "stage": "", "hardware": ""}


def rebuild_analysis_json(result_dir: Path, csv_dir: Path, prefix: str) -> list[str]:
    """Regenerate `analysis.json` from the tables. Returns the errors that remain.

    `analysis.json` is a derived view, so a document that fails the contract is
    almost always a stale or half-written conversion over tables that are fine --
    and that is repairable without judgement, by deriving it again. Reaching the end
    of an analysis and then throwing it away over a derived file would be the worst
    possible trade. The rejected document is kept as evidence rather than deleted.
    """
    target = result_dir / "analysis.json"
    described = describe(result_dir)
    if target.is_file():
        rejected = result_dir / "metadata" / "analysis.rejected.json"
        rejected.parent.mkdir(exist_ok=True)
        shutil.copy2(target, rejected)
    command = [
        sys.executable, str(Path(__file__).resolve().parent / "build_analysis_json.py"),
        str(csv_dir), str(target), "--prefix", prefix,
        "--model", described["model"],
        "--stage", described["stage"],
        "--hardware", described["hardware"],
    ]
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "").strip()
        return [f"could not rebuild analysis.json from {csv_dir}: {detail[:500]}"]
    try:
        return contract_errors(json.loads(target.read_text()))
    except (OSError, ValueError) as exc:
        return [f"rebuilt analysis.json is unreadable: {exc}"]


def check_contract(result_dir: Path, csv_dir: Path, prefix: str) -> None:
    """Refuse to package an analysis the frontend cannot render -- after repairing.

    The gate lives here, at the one step every package must pass through, rather
    than in a command someone has to remember: a malformed `analysis.json` that
    reaches a reader looks like a broken tool, and by then the trace and the
    reasoning are gone. But failing is the last resort, not the first: the document
    is derived from the tables, so it is rebuilt and re-checked first, and only a
    violation that survives that is a real defect in the tables themselves.
    """
    path = result_dir / "analysis.json"
    errors = []
    if path.is_file():
        try:
            errors = contract_errors(json.loads(path.read_text()))
        except ValueError as exc:
            errors = [f"analysis.json is not valid JSON: {exc}"]
    else:
        errors = ["analysis.json is missing"]
    if not errors:
        return
    print(
        "analysis.json does not meet the frontend contract, rebuilding it from "
        f"{csv_dir}:",
        file=sys.stderr,
    )
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    remaining = rebuild_analysis_json(result_dir, csv_dir, prefix)
    if not remaining:
        print("rebuilt analysis.json meets the frontend contract", file=sys.stderr)
        return
    joined = "\n".join(f"  - {error}" for error in remaining)
    raise SystemExit(
        "analysis.json still does not meet the NsysScope frontend contract after "
        f"rebuilding it from the tables:\n{joined}\n"
        "The tables themselves are missing what the frontend needs (most often the "
        "structural position/id/variant of a heterogeneous cycle) -- fix the table "
        "and package again."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--prefix", required=True)
    parser.add_argument(
        "--tables", type=Path,
        help="directory the tables are in now; located automatically by default",
    )
    parser.add_argument(
        "--trace", type=Path,
        help="exported .sqlite to place under trace/",
    )
    parser.add_argument(
        "--no-xlsx", action="store_true",
        help="skip the workbooks (they are regenerated from csv/ at any time)",
    )
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    if not result_dir.is_dir():
        raise SystemExit(f"result directory does not exist: {result_dir}")
    tables_dir = (args.tables.resolve() if args.tables
                  else locate_tables(result_dir, args.prefix))

    csv_dir = result_dir / "csv"
    xlsx_dir = result_dir / "xlsx"
    metadata_dir = result_dir / "metadata"
    for directory in (csv_dir, xlsx_dir, metadata_dir):
        directory.mkdir(exist_ok=True)

    tables = collect_tables(tables_dir, csv_dir, args.prefix)
    sweep_sidecars([tables_dir, result_dir], csv_dir, metadata_dir)
    check_contract(result_dir, csv_dir, args.prefix)
    if not args.no_xlsx:
        convert_directory(csv_dir, xlsx_dir)
    trace = (place_trace(args.trace.resolve(), result_dir, result_dir / "trace")
             if args.trace else None)
    manifest = write_manifest(result_dir, args.prefix, tables, trace)
    print(manifest)


if __name__ == "__main__":
    main()
