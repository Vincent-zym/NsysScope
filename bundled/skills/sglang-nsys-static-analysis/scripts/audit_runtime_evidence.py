#!/usr/bin/env python3
"""Extract captured SGLang runtime facts and report material conflicts."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import sqlite3
import subprocess
from pathlib import Path
from typing import Any


RUNTIME_KEYS = (
    "model_path",
    "json_model_override_args",
    "tp_size",
    "dcp_size",
    "dp_size",
    "enable_dp_attention",
    "enable_dp_lm_head",
    "moe_a2a_backend",
    "attention_backend",
    "decode_attention_backend",
    "prefill_attention_backend",
    "speculative_algorithm",
    "speculative_num_draft_tokens",
    "speculative_eagle_topk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlite", type=Path, required=True)
    parser.add_argument("--launch", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def table_exists(connection: sqlite3.Connection, name: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,),
    ).fetchone() is not None


def captured_environment(connection: sqlite3.Connection) -> dict[str, str]:
    if not table_exists(connection, "TARGET_INFO_SYSTEM_ENV"):
        return {}
    rows = connection.execute(
        "SELECT name,value FROM TARGET_INFO_SYSTEM_ENV",
    ).fetchall()
    environment: dict[str, str] = {}
    for name, value in rows:
        if name not in {"DeviceEnvironment", "Environment"}:
            continue
        for item in str(value).split(";"):
            key, separator, entry = item.partition("=")
            if separator and key:
                environment[key] = entry
    return environment


def captured_server_args(connection: sqlite3.Connection) -> tuple[dict[str, Any], str | None]:
    if not table_exists(connection, "StringIds"):
        return {}, None
    rows = connection.execute(
        "SELECT value FROM StringIds WHERE value LIKE '%server_args=ServerArgs(%'",
    ).fetchall()
    candidates: list[str] = []
    for (blob,) in rows:
        candidates.extend(
            line for line in str(blob).splitlines()
            if "server_args=ServerArgs(" in line
        )
    if not candidates:
        return {}, None
    line = candidates[0]
    values: dict[str, Any] = {}
    for key in RUNTIME_KEYS:
        match = re.search(
            rf"(?<![A-Za-z0-9_]){re.escape(key)}="
            r"('(?:\\.|[^'])*'|\"(?:\\.|[^\"])*\"|[^,]+)",
            line,
        )
        if not match:
            continue
        raw = match.group(1).strip()
        if raw in {"True", "False"}:
            values[key] = raw == "True"
        elif raw == "None":
            values[key] = None
        elif re.fullmatch(r"-?\d+", raw):
            values[key] = int(raw)
        else:
            values[key] = raw.strip("'\"")
    return values, line


def launch_facts(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(errors="replace")
    facts: dict[str, Any] = {"path": str(path)}
    match = re.search(r"(?:export\s+)?PYTHONPATH=[\"']?([^\"'\n]+)", text)
    if match:
        facts["PYTHONPATH"] = match.group(1)
    try:
        tokens = shlex.split(text.replace("\\\n", " "), comments=True)
    except ValueError:
        tokens = text.replace("\\\n", " ").split()
    arguments: dict[str, Any] = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if not token.startswith("--"):
            index += 1
            continue
        flag, separator, inline = token[2:].partition("=")
        if separator:
            arguments[flag] = inline
        elif index + 1 < len(tokens) and not tokens[index + 1].startswith("--"):
            arguments[flag] = tokens[index + 1]
            index += 1
        else:
            arguments[flag] = True
        index += 1
    for key in RUNTIME_KEYS:
        flag = key.replace("_", "-")
        if flag in arguments:
            value = arguments[flag]
            if isinstance(value, str) and re.fullmatch(r"-?\d+", value):
                value = int(value)
            elif value in {"true", "True"}:
                value = True
            elif value in {"false", "False"}:
                value = False
            facts[key] = value
        disabled = f"disable-{flag.removeprefix('enable-')}"
        if disabled in arguments:
            facts[key] = False
    return facts


def source_identity(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    result: dict[str, Any] = {"path": str(path.resolve()), "git_commit": None}
    try:
        completed = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            text=True, capture_output=True, timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return result
    if completed.returncode == 0:
        result["git_commit"] = completed.stdout.strip()
    return result


def main() -> None:
    args = parse_args()
    with sqlite3.connect(args.sqlite) as connection:
        environment = captured_environment(connection)
        runtime, raw_server_args = captured_server_args(connection)
    launch = launch_facts(args.launch)
    source = source_identity(args.source)
    conflicts: list[dict[str, Any]] = []
    for key in RUNTIME_KEYS:
        if key in runtime and key in launch and runtime[key] != launch[key]:
            conflicts.append({
                "field": key,
                "captured_runtime": runtime[key],
                "launch_material": launch[key],
                "resolution": "captured_runtime",
            })
    captured_commit = environment.get("SGLANG_BUILD_COMMIT")
    source_commit = source.get("git_commit")
    if captured_commit and source_commit and captured_commit != source_commit:
        conflicts.append({
            "field": "source_commit",
            "captured_runtime": captured_commit,
            "supplied_source": source_commit,
            "resolution": "captured runtime branch is authoritative; supplied source cannot prove call sites",
        })
    elif captured_commit and not source_commit:
        conflicts.append({
            "field": "source_commit",
            "captured_runtime": captured_commit,
            "supplied_source": None,
            "resolution": "source identity is unverified; do not use source-only defaults as runtime truth",
        })
    payload = {
        "schema_version": "1.0",
        "sqlite": str(args.sqlite.resolve()),
        "captured_runtime": runtime,
        "captured_environment": {
            key: environment.get(key)
            for key in ("PYTHONPATH", "SGLANG_BUILD_COMMIT", "SGLANG_IMAGE_TAG")
            if environment.get(key) is not None
        },
        "launch_material": launch,
        "source_material": source,
        "conflicts": conflicts,
        "status": "conflicts_detected" if conflicts else "consistent",
        "raw_server_args_present": raw_server_args is not None,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
