#!/usr/bin/env python3
"""Resolve and validate NsysScope's replaceable analysis skill."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any


SKILL_NAME = "sglang-nsys-static-analysis"
PROJECT = Path(__file__).resolve().parents[1]
BUNDLED_SKILL = PROJECT / "bundled" / "skills" / SKILL_NAME
CONFIG_DIR = Path(
    os.getenv("NSYSSCOPE_CONFIG_DIR", Path.home() / ".config" / "nsysscope")
).expanduser()
CONFIG_PATH = CONFIG_DIR / "config.json"
IGNORED_PARTS = {"result", "evals", "__pycache__"}
IGNORED_SUFFIXES = {".pyc", ".pyo"}


def read_config() -> dict[str, Any]:
    try:
        data = json.loads(CONFIG_PATH.read_text())
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid config file {CONFIG_PATH}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"invalid config file {CONFIG_PATH}: expected an object")
    return data


def write_config(data: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(CONFIG_PATH)


def validate_skill(path: Path) -> list[str]:
    path = path.expanduser().resolve()
    errors: list[str] = []
    skill_md = path / "SKILL.md"
    if not path.is_dir():
        return [f"目录不存在：{path}"]
    if not skill_md.is_file():
        return [f"缺少文件：{skill_md}"]
    text = skill_md.read_text(errors="replace")
    if not text.startswith("---\n"):
        errors.append("SKILL.md 缺少 YAML frontmatter")
    if f"name: {SKILL_NAME}" not in text:
        errors.append(f"SKILL.md 的 name 必须是 {SKILL_NAME}")
    for relative in (
        "scripts/build_static_analysis_tables.py",
        "scripts/extract_layer_operator_csv.py",
        "scripts/audit_runtime_evidence.py",
        "scripts/validate_analysis_package.py",
        # The frontend contract and the workbooks are generated from the Skill, so an
        # external Skill without them cannot serve a job end to end.
        "scripts/build_analysis_json.py",
        "scripts/csv_to_xlsx.py",
        "references/output-spec.md",
        "references/hardware-peaks.json",
    ):
        if not (path / relative).is_file():
            errors.append(f"缺少运行文件：{relative}")
    return errors


def skill_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = [
        item for item in path.rglob("*")
        if item.is_file()
        and not any(part in IGNORED_PARTS for part in item.relative_to(path).parts)
        and item.suffix not in IGNORED_SUFFIXES
    ]
    for item in sorted(files):
        relative = item.relative_to(path).as_posix()
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(item.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def configured_skill() -> Path | None:
    value = read_config().get("skill_dir")
    return Path(str(value)).expanduser().resolve() if value else None


def resolve_skill() -> tuple[Path, str]:
    environment = os.getenv("NSYSSCOPE_SKILL_DIR")
    if environment:
        path = Path(environment).expanduser().resolve()
        errors = validate_skill(path)
        if errors:
            raise RuntimeError("环境变量指定的 Skill 无效：" + "; ".join(errors))
        return path, "environment"
    configured = configured_skill()
    if configured:
        errors = validate_skill(configured)
        if errors:
            raise RuntimeError(
                "已配置的外部 Skill 无效，请修复或执行 skill reset："
                + "; ".join(errors)
            )
        return configured, "external"
    candidates = [
        (Path.home() / ".codex" / "skills" / SKILL_NAME, "codex"),
        (BUNDLED_SKILL, "bundled"),
    ]
    messages = []
    for path, source in candidates:
        errors = validate_skill(path)
        if not errors:
            return path.resolve(), source
        messages.append(f"{source}: {'; '.join(errors)}")
    raise RuntimeError("找不到可用的分析 Skill：" + " | ".join(messages))


def provenance(path: Path, source: str) -> dict[str, str]:
    return {
        "name": SKILL_NAME,
        "source": source,
        "path": str(path),
        "sha256": skill_hash(path),
    }


def command_status(_: argparse.Namespace) -> int:
    path, source = resolve_skill()
    info = provenance(path, source)
    print(f"Skill:   {info['name']}")
    print(f"来源:    {info['source']}")
    print(f"目录:    {info['path']}")
    print(f"SHA256:  {info['sha256']}")
    print("状态:    验证通过")
    return 0


def command_resolve(_: argparse.Namespace) -> int:
    path, _ = resolve_skill()
    print(path)
    return 0


def command_info(_: argparse.Namespace) -> int:
    path, source = resolve_skill()
    print(json.dumps(provenance(path, source), ensure_ascii=False))
    return 0


def command_use(args: argparse.Namespace) -> int:
    path = Path(args.path).expanduser().resolve()
    errors = validate_skill(path)
    if errors:
        for error in errors:
            print(f"✗ {error}", file=sys.stderr)
        return 1
    data = read_config()
    data["skill_dir"] = str(path)
    data["skill_sha256"] = skill_hash(path)
    write_config(data)
    print(f"✓ 已切换到外部 Skill：{path}")
    print("  外部目录后续更新会在下次启动时自动生效。")
    return 0


def command_sync(_: argparse.Namespace) -> int:
    path = configured_skill()
    if path is None:
        print("当前没有配置外部 Skill；正在使用 Codex 安装版或内置版本。")
        return 0
    errors = validate_skill(path)
    if errors:
        for error in errors:
            print(f"✗ {error}", file=sys.stderr)
        return 1
    digest = skill_hash(path)
    data = read_config()
    changed = data.get("skill_sha256") != digest
    data["skill_sha256"] = digest
    write_config(data)
    print(f"✓ 外部 Skill 验证通过：{path}")
    print(f"  SHA256: {digest}")
    print("  状态: 已检测到更新" if changed else "  状态: 无变化")
    return 0


def command_reset(_: argparse.Namespace) -> int:
    data = read_config()
    data.pop("skill_dir", None)
    data.pop("skill_sha256", None)
    if data:
        write_config(data)
    elif CONFIG_PATH.exists():
        CONFIG_PATH.unlink()
    path, source = resolve_skill()
    print(f"✓ 已取消外部 Skill，当前使用 {source}：{path}")
    return 0


def command_validate(args: argparse.Namespace) -> int:
    if args.path:
        path = Path(args.path).expanduser().resolve()
    else:
        path, _ = resolve_skill()
    errors = validate_skill(path)
    if errors:
        for error in errors:
            print(f"✗ {error}", file=sys.stderr)
        return 1
    print(f"✓ Skill 验证通过：{path}")
    print(f"  SHA256: {skill_hash(path)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status").set_defaults(handler=command_status)
    subparsers.add_parser("resolve").set_defaults(handler=command_resolve)
    subparsers.add_parser("info").set_defaults(handler=command_info)
    use_parser = subparsers.add_parser("use")
    use_parser.add_argument("path")
    use_parser.set_defaults(handler=command_use)
    sync_parser = subparsers.add_parser("sync")
    sync_parser.set_defaults(handler=command_sync)
    reset_parser = subparsers.add_parser("reset")
    reset_parser.set_defaults(handler=command_reset)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("path", nargs="?")
    validate_parser.set_defaults(handler=command_validate)
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
