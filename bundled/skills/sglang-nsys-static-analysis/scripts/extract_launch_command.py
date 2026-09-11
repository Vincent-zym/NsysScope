#!/usr/bin/env python3
"""Recover the captured launch command from an exported nsys SQLite trace.

The command line nsys wrapped is stored in the trace itself, in
`META_DATA_CAPTURE`, as `PROCESS_<n>:COMMAND` plus `PROCESS_<n>:ARGUMENT_0..N`
-- one row per argv item, which is why an argument containing spaces survives
intact. This is the same string Nsight Systems shows as "Session activities /
Launch:" in its Analysis Summary.

Reading it from the trace replaces asking the caller for the deployment
script: the script states intent, this states what actually ran, with every
shell variable and arithmetic expression already resolved. Measured on the 86
SQLite traces on this machine: 86/86 carry it, across nsys export versions
2025.1.3.0 / 2026.2.1.210 / 2026.4.1.191 and 60 different command shapes.

The captured `server_args=ServerArgs(...)` log line that
`audit_runtime_evidence.py` reads is still the authority on what the runtime
resolved each flag to -- it is just not always present (2 of 7 packages here
have no such line), while argv always is.
"""
from __future__ import annotations

import argparse
import shlex
import sqlite3
import sys
from pathlib import Path


def process_prefixes(rows: dict[str, str]) -> list[str]:
    """`PROCESS_0`, `PROCESS_1`, ... in numeric order.

    A capture that attached to more than one process records each of them; the
    first is the one nsys launched, and the others are kept so a multi-process
    capture is not silently reported as single-process.
    """
    prefixes = {name.split(":", 1)[0] for name in rows if name.startswith("PROCESS_")}
    return sorted(prefixes, key=lambda name: int(name.rsplit("_", 1)[1]))


def argv_of(rows: dict[str, str], prefix: str) -> list[str]:
    command = rows.get(f"{prefix}:COMMAND")
    if not command:
        return []
    argv = [command]
    index = 0
    while f"{prefix}:ARGUMENT_{index}" in rows:
        argv.append(rows[f"{prefix}:ARGUMENT_{index}"])
        index += 1
    return argv


def launch_commands(sqlite_path: Path) -> list[tuple[str, list[str]]]:
    connection = sqlite3.connect(f"file:{sqlite_path}?mode=ro", uri=True)
    try:
        tables = {
            row[0] for row in
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        if "META_DATA_CAPTURE" not in tables:
            return []
        rows = dict(connection.execute("SELECT name, value FROM META_DATA_CAPTURE"))
    finally:
        connection.close()
    found = []
    for prefix in process_prefixes(rows):
        argv = argv_of(rows, prefix)
        if argv:
            found.append((prefix, argv))
    return found


def render(commands: list[tuple[str, list[str]]], rows_hint: str | None = None) -> str:
    """One command per line, shell-quoted so the file can be read or re-run.

    Quoting matters: `--model-loader-extra-config` is stored as bare JSON in a
    single argv slot, and a consumer that splits the line on whitespace would
    otherwise turn it into three broken arguments.
    """
    lines = []
    for prefix, argv in commands:
        if len(commands) > 1:
            lines.append(f"# {prefix}")
        lines.append(" ".join(shlex.quote(item) for item in argv))
    if rows_hint:
        lines.append(f"# {rows_hint}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sqlite", type=Path, help="exported nsys SQLite trace")
    parser.add_argument("--output", type=Path, default=None,
                        help="write the command here instead of stdout")
    args = parser.parse_args()

    if not args.sqlite.is_file():
        raise SystemExit(f"[launch-command] 找不到 trace：{args.sqlite}")
    commands = launch_commands(args.sqlite)
    if not commands:
        raise SystemExit(
            f"[launch-command] {args.sqlite} 的 META_DATA_CAPTURE 里没有 "
            "PROCESS_*:COMMAND，无法还原启动命令"
        )
    text = render(commands)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
        argc = len(commands[0][1])
        print(f"[launch-command] {args.output}（{len(commands)} 个进程，argc={argc}）")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
