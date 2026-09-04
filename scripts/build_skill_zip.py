#!/usr/bin/env python3
"""Package the analysis Skill as a standalone, distributable zip.

The Skill is usable without NsysScope: hand it to any Agent CLI that understands
the skill mechanism and it produces a complete result package on its own. This
builds the artifact for that use, so nobody has to clone the repo to get it.

The archive keeps the Skill's own directory as its top level, so unpacking it
anywhere yields `sglang-nsys-static-analysis/SKILL.md` -- which is what a skills
directory expects. Bit-for-bit reproducible: entries are sorted and every
timestamp is zeroed, so the same commit always produces the same bytes whether
it is built here or in CI.
"""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SKILL = PROJECT / "bundled" / "skills" / "sglang-nsys-static-analysis"

# Caches and test scratch never belong in a distributed artifact. `evals/` does:
# it is how someone verifies a Skill copy still honours its contracts, which
# matters most for the people running it outside this repo.
EXCLUDED_PARTS = {"__pycache__", ".pytest_cache"}
EXCLUDED_SUFFIXES = {".pyc", ".pyo"}


def skill_files(root: Path) -> list[Path]:
    return sorted(
        path for path in root.rglob("*")
        if path.is_file()
        and not any(part in EXCLUDED_PARTS for part in path.relative_to(root).parts)
        and path.suffix not in EXCLUDED_SUFFIXES
    )


def build(root: Path, output: Path) -> Path:
    if not (root / "SKILL.md").is_file():
        raise SystemExit(f"not a Skill directory (no SKILL.md): {root}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in skill_files(root):
            arcname = Path(root.name) / path.relative_to(root)
            # A fixed date_time is what makes the output reproducible; ZIP cannot
            # store a zero timestamp, so use the format's own epoch (1980-01-01).
            info = zipfile.ZipInfo(str(arcname), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, path.read_bytes())
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skill-dir", type=Path, default=SKILL)
    parser.add_argument(
        "--output", type=Path,
        default=PROJECT / "release" / "sglang-nsys-static-analysis.zip",
    )
    args = parser.parse_args()
    built = build(args.skill_dir.resolve(), args.output)
    digest = hashlib.sha256(built.read_bytes()).hexdigest()
    print(built)
    print(f"{built.stat().st_size / 1024:.1f} KiB")
    print(f"sha256 {digest}")


if __name__ == "__main__":
    main()
