#!/usr/bin/env python3
"""Build a small self-extracting NsysScope Linux executable."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import os
import stat
import tarfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
VERSION = "0.3.3"
INCLUDE = (
    "nsysscope",
    "README.md",
    "ARCHITECTURE.md",
    "backend",
    "scripts/skill_manager.py",
    "bundled",
)
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_NAMES = {"test_service.py", ".env.example", "run.sh"}


def source_files() -> list[Path]:
    files: list[Path] = []
    for relative in INCLUDE:
        path = PROJECT / relative
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(item for item in path.rglob("*") if item.is_file())
    return sorted({
        path for path in files
        if not any(part in EXCLUDED_PARTS for part in path.relative_to(PROJECT).parts)
        and path.suffix not in EXCLUDED_SUFFIXES
        and path.name not in EXCLUDED_NAMES
    })


def archive_bytes() -> bytes:
    raw = io.BytesIO()
    with gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=9, mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for path in source_files():
                relative = path.relative_to(PROJECT)
                info = archive.gettarinfo(str(path), arcname=str(relative))
                info.uid = info.gid = 0
                info.uname = info.gname = ""
                info.mtime = 0
                with path.open("rb") as handle:
                    archive.addfile(info, handle)
    return raw.getvalue()


def launcher_stub(version: str, runtime_id: str) -> bytes:
    text = f"""#!/usr/bin/env bash
set -euo pipefail

RUNTIME_BASE="${{XDG_DATA_HOME:-${{HOME}}/.local/share}}/nsysscope"
CACHE_BASE="${{XDG_CACHE_HOME:-${{HOME}}/.cache}}/nsysscope"
INSTALL_DIR="$RUNTIME_BASE/runtime-{runtime_id}"
PAYLOAD_LINE="$(awk '/^__NSYSSCOPE_ARCHIVE_BELOW__$/ {{ print NR + 1; exit }}' "$0")"
MATERIAL_ROOT="$(pwd -P)"

# `-x nsysscope` became true halfway through extraction, so an interrupted first
# run left a partial runtime that every later run happily reused (and failed in,
# with "Skill 无法解析"). Extract to a staging dir, stamp it, then move it into
# place, and treat a missing stamp as "extract again".
if [[ ! -f "$INSTALL_DIR/.nsysscope-complete" ]]; then
  mkdir -p "$RUNTIME_BASE" "$CACHE_BASE"
  STAGE_DIR="$(mktemp -d "$RUNTIME_BASE/.stage-XXXXXX")"
  trap 'rm -rf -- "$STAGE_DIR"' EXIT
  tail -n +"$PAYLOAD_LINE" "$0" | tar -xzf - -C "$STAGE_DIR"
  chmod +x "$STAGE_DIR/nsysscope"
  touch "$STAGE_DIR/.nsysscope-complete"
  if [[ ! -f "$INSTALL_DIR/.nsysscope-complete" ]]; then
    rm -rf -- "$INSTALL_DIR"
    mv "$STAGE_DIR" "$INSTALL_DIR"
  fi
  rm -rf -- "$STAGE_DIR"
  trap - EXIT
fi

export NSYSSCOPE_VENV_DIR="${{NSYSSCOPE_VENV_DIR:-$CACHE_BASE/venv-{version}}}"
export NSYSSCOPE_ALLOWED_ROOTS="${{NSYSSCOPE_ALLOWED_ROOTS:-$MATERIAL_ROOT}}"
exec "$INSTALL_DIR/nsysscope" "$@"

__NSYSSCOPE_ARCHIVE_BELOW__
"""
    return text.encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "release" / "nsysscope-linux-x86_64.run",
    )
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = archive_bytes()
    runtime_id = f"{VERSION}-{hashlib.sha256(payload).hexdigest()[:12]}"
    args.output.write_bytes(launcher_stub(VERSION, runtime_id) + payload)
    args.output.chmod(args.output.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    print(args.output)
    print(f"{args.output.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
