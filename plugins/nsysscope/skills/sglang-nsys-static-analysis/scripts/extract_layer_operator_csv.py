#!/usr/bin/env python3
"""Extract module-tagged per-kernel CSV files from an Nsight Systems report.

The helper accepts either a pre-exported .sqlite file or an .nsys-rep file that
must first be exported to sqlite. The caller supplies one or more layer kernel
index windows and JSON module maps after analyzing model source and the timeline.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
import subprocess
from pathlib import Path
from typing import Any


DEFAULT_COLUMNS = [
    "module",
    "operator_name",
    "duration_us",
    "start_ns",
    "end_ns",
    "device",
    "stream",
    "python_function",
    "mapping_reason",
]


def table_columns(cur: sqlite3.Cursor, table: str) -> set[str]:
    return {row[1] for row in cur.execute(f"PRAGMA table_info({table})")}


def require_tables(cur: sqlite3.Cursor) -> None:
    tables = {
        row[0]
        for row in cur.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
        )
    }
    missing = {"CUPTI_ACTIVITY_KIND_KERNEL", "StringIds"} - tables
    if missing:
        raise RuntimeError(f"Nsight sqlite is missing required tables: {sorted(missing)}")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_module_map(path: Path | None) -> dict[int, Any]:
    if path is None:
        return {}
    data: Any = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("module map must be a JSON object mapping kernel indices to labels")
    return {int(key): value for key, value in data.items()}


def resolve_module_entry(
    module_map: dict[int, Any],
    index: int,
    relative_offset: int,
    unknown_prefix: str,
) -> tuple[str, str]:
    raw = module_map.get(index)
    if raw is None:
        raw = module_map.get(relative_offset)
    if raw is None:
        module = f"{unknown_prefix}/{index}"
        return module, module
    if isinstance(raw, str):
        return raw, raw
    if isinstance(raw, dict):
        module = str(raw.get("module", f"{unknown_prefix}/{index}"))
        occurrence = raw.get("occurrence", raw.get("group", module))
        return module, f"{module}#{occurrence}"
    module = str(raw)
    return module, module


def load_python_function_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data: Any = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(
            "python function map must be a JSON object mapping kernel indices or module labels to call-site strings"
        )
    return {str(key): str(value) for key, value in data.items()}


def load_mapping_reason_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    data: Any = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(
            "mapping reason map must be a JSON object mapping kernel indices, module labels, or __layer_total__ to explanation strings"
        )
    return {str(key): str(value) for key, value in data.items()}


def resolve_python_function(
    python_function_map: dict[str, str],
    index: int | str,
    relative_offset: int | str,
    module: str,
) -> str:
    index_key = str(index)
    relative_key = str(relative_offset)
    if index_key in python_function_map:
        return python_function_map[index_key]
    if relative_key in python_function_map:
        return python_function_map[relative_key]
    if module in python_function_map:
        return python_function_map[module]
    return ""


def resolve_mapping_reason(
    mapping_reason_map: dict[str, str],
    index: int | str,
    relative_offset: int | str,
    module: str,
) -> str:
    index_key = str(index)
    relative_key = str(relative_offset)
    if index_key in mapping_reason_map:
        return mapping_reason_map[index_key]
    if relative_key in mapping_reason_map:
        return mapping_reason_map[relative_key]
    if module in mapping_reason_map:
        return mapping_reason_map[module]
    if "*" in mapping_reason_map:
        return mapping_reason_map["*"]
    return ""


def resolve_nsys(nsys: str | None) -> str:
    if nsys:
        return nsys
    resolved = shutil.which("nsys")
    if resolved is None:
        raise RuntimeError("input is .nsys-rep but no --nsys was provided and nsys is not in PATH")
    return resolved


def export_nsys_rep(
    nsys_rep: Path,
    sqlite_path: Path | None,
    nsys: str | None,
    force_export: bool,
) -> Path:
    if sqlite_path is None:
        sqlite_path = nsys_rep.with_suffix(".sqlite")

    if sqlite_path.exists() and not force_export:
        return sqlite_path

    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    nsys_bin = resolve_nsys(nsys)
    command = [
        nsys_bin,
        "export",
        "--type",
        "sqlite",
        "--output",
        str(sqlite_path),
        str(nsys_rep),
    ]
    try:
        subprocess.run(command, check=True, text=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "failed to export .nsys-rep to sqlite\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{exc.stdout}\n"
            f"stderr:\n{exc.stderr}"
        ) from exc

    if not sqlite_path.exists():
        raise RuntimeError(f"nsys export completed but sqlite was not created: {sqlite_path}")
    return sqlite_path


def build_query(cur: sqlite3.Cursor, name_mode: str) -> str:
    cols = table_columns(cur, "CUPTI_ACTIVITY_KIND_KERNEL")
    if "deviceId" not in cols:
        raise RuntimeError("CUPTI_ACTIVITY_KIND_KERNEL.deviceId is required")

    if name_mode == "demangled" and "demangledName" in cols:
        name_column = "demangledName"
    elif name_mode == "short" and "shortName" in cols:
        name_column = "shortName"
    elif "demangledName" in cols:
        name_column = "demangledName"
    elif "shortName" in cols:
        name_column = "shortName"
    else:
        raise RuntimeError("kernel table has neither demangledName nor shortName")

    stream_expr = "k.streamId" if "streamId" in cols else "NULL"
    return f"""
        SELECT
            k.start,
            k.end,
            s.value AS operator_name,
            k.deviceId AS device,
            {stream_expr} AS stream
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        LEFT JOIN StringIds s ON k.{name_column} = s.id
        WHERE k.deviceId = ?
        ORDER BY k.start
    """


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract one or more layer-mode CUDA kernel CSVs from an Nsight Systems report."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--nsys-rep", type=Path, help="Nsight Systems .nsys-rep path")
    input_group.add_argument("--sqlite-input", type=Path, help="Pre-exported Nsight Systems sqlite path")
    parser.add_argument(
        "--sqlite",
        type=Path,
        help="sqlite export path for --nsys-rep",
    )
    parser.add_argument("--nsys", help="nsys binary path used when exporting --nsys-rep")
    parser.add_argument(
        "--force-export",
        action="store_true",
        help="re-export --nsys-rep even if the sqlite path already exists",
    )
    parser.add_argument(
        "--runs-json",
        type=Path,
        help="JSON list of extraction runs for multi-mode output",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="manifest JSON path for --runs-json; defaults to runs_json parent/operator_csv_manifest.json",
    )
    parser.add_argument("--output", type=Path, help="single CSV output path")
    parser.add_argument("--device", type=int, help="deviceId to extract for single output")
    parser.add_argument(
        "--start-index",
        type=int,
        help="inclusive kernel index within the selected device ordered by start time",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="inclusive kernel index within the selected device ordered by start time",
    )
    parser.add_argument(
        "--module-map",
        type=Path,
        help="JSON object mapping selected kernel indices to module labels for single output",
    )
    parser.add_argument(
        "--python-function-map",
        type=Path,
        help="JSON object mapping kernel indices or module labels to '<function> @ <path>:<line>' values for the python_function CSV column",
    )
    parser.add_argument(
        "--mapping-reason-map",
        type=Path,
        help="JSON object mapping kernel indices, module labels, or '*' to explanations for the final mapping_reason CSV column",
    )
    parser.add_argument(
        "--name-mode",
        choices=["demangled", "short"],
        default="demangled",
        help="kernel name field to prefer; default uses full demangled names",
    )
    parser.add_argument(
        "--unknown-prefix",
        default="unknown",
        help="module label prefix when an index is absent from module map",
    )
    return parser.parse_args()


def get_sqlite_path(args: argparse.Namespace) -> Path:
    if args.sqlite_input is not None:
        return args.sqlite_input
    if args.nsys_rep is not None:
        return export_nsys_rep(args.nsys_rep, args.sqlite, args.nsys, args.force_export)
    raise RuntimeError("either --nsys-rep or --sqlite-input is required")


def normalize_run(raw: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    try:
        output = Path(raw["output"])
        device = int(raw["device"])
        start_index = int(raw["start_index"])
        end_index = int(raw["end_index"])
    except KeyError as exc:
        raise ValueError(f"run is missing required field: {exc}") from exc
    if end_index < start_index:
        raise ValueError(f"run {raw.get('name', output)} has end_index < start_index")
    layer_start_ns = raw.get("layer_start_ns")
    layer_end_ns = raw.get("layer_end_ns")
    if layer_start_ns is not None:
        layer_start_ns = int(layer_start_ns)
    if layer_end_ns is not None:
        layer_end_ns = int(layer_end_ns)
    if layer_start_ns is not None and layer_end_ns is not None and layer_end_ns < layer_start_ns:
        raise ValueError(f"run {raw.get('name', output)} has layer_end_ns < layer_start_ns")
    return {
        "name": str(raw.get("name", output.stem)),
        "output": output,
        "device": device,
        "start_index": start_index,
        "end_index": end_index,
        "module_map": Path(raw["module_map"]) if raw.get("module_map") else None,
        "python_function_map": Path(raw["python_function_map"])
        if raw.get("python_function_map")
        else None,
        "mapping_reason_map": Path(raw["mapping_reason_map"])
        if raw.get("mapping_reason_map")
        else None,
        "discriminator": str(raw.get("discriminator", "")),
        "unknown_prefix": str(raw.get("unknown_prefix", args.unknown_prefix)),
        "layer_start_ns": layer_start_ns,
        "layer_end_ns": layer_end_ns,
        "total_source": str(raw.get("total_source", "selected window span")),
    }


def build_single_run(args: argparse.Namespace) -> dict[str, Any]:
    missing = [
        name
        for name, value in {
            "--output": args.output,
            "--device": args.device,
            "--start-index": args.start_index,
            "--end-index": args.end_index,
        }.items()
        if value is None
    ]
    if missing:
        raise ValueError("single-output mode requires " + ", ".join(missing))
    return normalize_run(
        {
            "name": args.output.stem,
            "output": str(args.output),
            "device": args.device,
            "start_index": args.start_index,
            "end_index": args.end_index,
            "module_map": str(args.module_map) if args.module_map else None,
            "python_function_map": str(args.python_function_map)
            if args.python_function_map
            else None,
            "mapping_reason_map": str(args.mapping_reason_map)
            if args.mapping_reason_map
            else None,
        },
        args,
    )


def load_runs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.runs_json is None:
        return [build_single_run(args)]
    data = load_json(args.runs_json)
    if not isinstance(data, list):
        raise ValueError("--runs-json must be a JSON list")
    return [normalize_run(raw, args) for raw in data]


def fetch_device_rows(cur: sqlite3.Cursor, query: str, device: int) -> list[tuple[Any, ...]]:
    return list(cur.execute(query, (device,)))


def assert_contiguous_groups(csv_rows: list[dict[str, Any]], run_name: str) -> None:
    seen_closed: set[str] = set()
    active_group: str | None = None
    for item in csv_rows:
        group_key = item["group_key"]
        if group_key == active_group:
            continue
        if group_key in seen_closed:
            raise ValueError(
                f"run {run_name} produced non-contiguous rows for module group {group_key!r}. "
                "Rows from the same submodule occurrence must be emitted as one contiguous block; "
                "check module_map occurrence/group assignments."
            )
        if active_group is not None:
            seen_closed.add(active_group)
        active_group = group_key


def assert_single_stream_groups(csv_rows: list[dict[str, Any]], run_name: str) -> None:
    streams_by_group: dict[str, set[Any]] = {}
    for item in csv_rows:
        streams_by_group.setdefault(item["group_key"], set()).add(item["stream"])
    for group_key, streams in streams_by_group.items():
        if len(streams) > 1:
            raise ValueError(
                f"run {run_name} assigned module group {group_key!r} to multiple streams: "
                f"{sorted(streams)}. One submodule occurrence must map to exactly one stream; "
                "split this module into separate occurrence/group entries in module_map."
            )


def write_run_csv(
    run: dict[str, Any],
    rows: list[tuple[Any, ...]],
) -> dict[str, Any]:
    start_index = run["start_index"]
    end_index = run["end_index"]
    if start_index >= len(rows) or end_index >= len(rows):
        raise IndexError(
            f"run {run['name']} requested index window [{start_index}, {end_index}] but "
            f"device {run['device']} has only {len(rows)} kernels"
        )

    module_map = load_module_map(run["module_map"])
    python_function_map = load_python_function_map(run["python_function_map"])
    mapping_reason_map = load_mapping_reason_map(run["mapping_reason_map"])
    first_start = rows[start_index][0]
    last_end = rows[end_index][1]
    layer_start_ns = run["layer_start_ns"] if run["layer_start_ns"] is not None else first_start
    layer_end_ns = run["layer_end_ns"] if run["layer_end_ns"] is not None else last_end
    layer_duration_us = (layer_end_ns - layer_start_ns) / 1000.0

    csv_rows = []
    group_start_ns: dict[str, int] = {}
    group_first_index: dict[str, int] = {}
    missing_module_indices: list[int] = []
    for index in range(start_index, end_index + 1):
        relative_offset = index - start_index
        start_ns, end_ns, operator_name, device, stream = rows[index]
        duration_us = (end_ns - start_ns) / 1000.0
        module, group_key = resolve_module_entry(
            module_map,
            index,
            relative_offset,
            run["unknown_prefix"],
        )
        if index not in module_map and relative_offset not in module_map:
            missing_module_indices.append(index)
        python_function = resolve_python_function(
            python_function_map,
            index,
            relative_offset,
            module,
        )
        mapping_reason = resolve_mapping_reason(
            mapping_reason_map,
            index,
            relative_offset,
            module,
        )
        group_start_ns[group_key] = min(group_start_ns.get(group_key, start_ns), start_ns)
        group_first_index[group_key] = min(group_first_index.get(group_key, index), index)
        csv_rows.append(
            {
                "index": index,
                "relative_offset": relative_offset,
                "group_key": group_key,
                "module": module,
                "operator_name": operator_name or "",
                "duration_us": duration_us,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "device": device,
                "stream": stream,
                "python_function": python_function,
                "mapping_reason": mapping_reason,
            }
        )

    if missing_module_indices:
        preview = ", ".join(str(index) for index in missing_module_indices[:10])
        raise ValueError(
            f"run {run['name']} is missing module-map entries for "
            f"{len(missing_module_indices)} kernels, starting with global indices: {preview}. "
            "Provide module_map keys as either global kernel indices or relative offsets "
            "inside the selected window so CSV ordering can group by submodule."
        )

    assert_single_stream_groups(csv_rows, run["name"])
    csv_rows.sort(
        key=lambda item: (
            group_start_ns[item["group_key"]],
            group_first_index[item["group_key"]],
            item["start_ns"],
            item["index"],
        )
    )
    assert_contiguous_groups(csv_rows, run["name"])

    output = run["output"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(DEFAULT_COLUMNS)
        for item in csv_rows:
            writer.writerow(
                [
                    item["module"],
                    item["operator_name"],
                    f"{item['duration_us']:.3f}",
                    item["start_ns"],
                    item["end_ns"],
                    item["device"],
                    item["stream"] if item["stream"] is not None else "",
                    item["python_function"],
                    item["mapping_reason"],
                ]
            )
        layer_total_python_function = resolve_python_function(
            python_function_map,
            "__layer_total__",
            "__layer_total__",
            "__layer_total__",
        )
        layer_total_mapping_reason = resolve_mapping_reason(
            mapping_reason_map,
            "__layer_total__",
            "__layer_total__",
            "__layer_total__",
        )
        writer.writerow(
            [
                "__layer_total__",
                run["name"],
                f"{layer_duration_us:.3f}",
                layer_start_ns,
                layer_end_ns,
                run["device"],
                "",
                layer_total_python_function,
                layer_total_mapping_reason,
            ]
        )

    return {
        "name": run["name"],
        "csv": str(output),
        "device": run["device"],
        "start_index": start_index,
        "end_index": end_index,
        "start_ns": first_start,
        "end_ns": last_end,
        "layer_start_ns": layer_start_ns,
        "layer_end_ns": layer_end_ns,
        "layer_duration_us": round(layer_duration_us, 3),
        "total_source": run["total_source"],
        "kernel_row_count": end_index - start_index + 1,
        "csv_row_count": end_index - start_index + 2,
        "csv_ordering": "module_group_start_ns_then_kernel_start_ns",
        "mapping_reason_map": str(run["mapping_reason_map"]) if run["mapping_reason_map"] else None,
        "discriminator": run["discriminator"],
    }


def main() -> None:
    args = parse_args()
    sqlite_path = get_sqlite_path(args)
    runs = load_runs(args)

    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    require_tables(cur)
    query = build_query(cur, args.name_mode)

    rows_by_device: dict[int, list[tuple[Any, ...]]] = {}
    manifest_runs = []
    for run in runs:
        device = run["device"]
        if device not in rows_by_device:
            rows_by_device[device] = fetch_device_rows(cur, query, device)
        manifest_runs.append(write_run_csv(run, rows_by_device[device]))

    print(f"sqlite={sqlite_path}")
    if args.runs_json is not None:
        manifest_path = args.manifest or args.runs_json.parent / "operator_csv_manifest.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest = {
            "sqlite": str(sqlite_path),
            "name_mode": args.name_mode,
            "runs": manifest_runs,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
        print(f"manifest={manifest_path}")
        for item in manifest_runs:
            print(f"csv[{item['name']}]={item['csv']}")
    else:
        print(f"csv={manifest_runs[0]['csv']}")


if __name__ == "__main__":
    main()
