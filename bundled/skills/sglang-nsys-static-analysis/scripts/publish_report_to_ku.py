#!/usr/bin/env python3
"""Publish a finished `final_report.md` into an existing 如流知识库 page.

Optional last step: the report in the result package is the deliverable, and
this only mirrors it onto a wiki page when the caller supplies one. The package
is opened read-only -- nothing here rewrites the report, so a failed publish
never damages the analysis output.

Three things about 如流's open API decide how this works, all established by
probing a live document rather than from its docs:

1. Inline styles do survive. `border:1px solid #999` becomes the cell's
   `borderColor`/`borderIndex`, `vertical-align` becomes `verticalAlign`, and
   `text-align` lands as `textAlign` on the cell's inner paragraph. `<b>`
   becomes a bold text run and `<code>` an `inline-code` node.
2. Cell background is the one exception: the API reads the `background-color`
   longhand and silently drops the `background` shorthand (and a `bgcolor`
   attribute). `build_final_report.py` emits the longhand; older reports are
   normalised here so they publish with their header tints intact.
3. An edit lands in the document's edit state, not its preview state. Without a
   following `publish-doc` the API reports success and readers still see the old
   page, so publishing is part of writing, not an optional follow-up.

The write itself is `cover` with a placeholder paragraph (which clears the page
and gives mdsl something to anchor on), then a single mdsl `replace_range` that
swaps that placeholder for the whole report. Anchoring on a placeholder we just
wrote is what keeps this independent of whatever the page held before.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

PLACEHOLDER = "NSYSSCOPE-REPORT-PLACEHOLDER"


def resolve_ku_binary(explicit: str | None) -> Path:
    """Locate the `ku` CLI from the ku-doc-manage Skill."""
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    for variable in ("KU_BIN", "KU_DOC_MANAGE_DIR", "COMATE_SKILL_DIR"):
        value = os.getenv(variable)
        if not value:
            continue
        candidate = Path(value)
        candidates.extend([candidate, candidate / "bin" / "ku"])
    found = shutil.which("ku")
    if found:
        candidates.append(Path(found))
    candidates.append(Path.home() / "ku-doc-manage" / "bin" / "ku")
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    raise SystemExit(
        "找不到 ku CLI：请用 --ku-bin 指定 ku-doc-manage/bin/ku，"
        "或设置 KU_DOC_MANAGE_DIR / COMATE_SKILL_DIR 环境变量"
    )


def document_id(url: str | None, doc_id: str | None) -> str:
    """A knowledge-base URL's 4th path segment is the document id."""
    if doc_id:
        return doc_id
    if not url:
        raise SystemExit("必须提供 --url 或 --doc-id")
    parts = [part for part in url.split("?")[0].rstrip("/").split("/") if part]
    try:
        knowledge = parts.index("knowledge")
    except ValueError:
        raise SystemExit(f"不是知识库文档链接：{url}") from None
    tail = parts[knowledge + 1:]
    if len(tail) < 4:
        raise SystemExit(
            f"链接缺少文档段（需要 4 段 path，实际 {len(tail)} 段）：{url}"
        )
    return tail[3]


def run(ku: Path, *args: str) -> dict:
    """One ku call; raises on transport failure or an API-level failure.

    `--username` is passed by the caller, not added here: the write commands
    require it and `query-content` rejects it outright, so appending it
    unconditionally turns the read-back into a usage error.
    """
    process = subprocess.run(
        [str(ku), *args],
        capture_output=True, text=True,
    )
    payload = process.stdout[process.stdout.find("{"):] if "{" in process.stdout else ""
    if not payload:
        raise SystemExit(
            f"ku {args[0]} 无返回：\n{process.stdout.strip()}\n{process.stderr.strip()}"
        )
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"ku {args[0]} 返回不是 JSON：{exc}") from exc
    if not data.get("success"):
        raise SystemExit(f"ku {args[0]} 失败：{data.get('returnMessage') or data}")
    return data


def normalise(html: str) -> tuple[str, int]:
    """Rewrite the `background` shorthand the API drops into the longhand."""
    shorthand = re.compile(r"background:(?!-)")
    return shorthand.sub("background-color:", html), len(shorthand.findall(html))


def verify(ku: Path, doc_id: str, html: str) -> None:
    """Read the published page back and compare it with what we sent.

    Reporting "published" on the strength of a 200 would hide the two failure
    modes that actually happen: content parsed into nothing (an unsupported tag
    silently drops its whole block) and an edit stuck in the edit state. Both
    show up as a table or cell count that does not match the source.
    """
    data = run(ku, "query-content", "--doc-id", doc_id, "--protocol", "json")
    content = (data.get("result") or {}).get("content")
    tables: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "table":
                tables.append(node)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(content)
    cells = [
        cell
        for table in tables for row in table.get("children", [])
        for cell in row.get("children", [])
    ]
    expected_tables = html.count("<table")
    expected_cells = html.count("<td") + html.count("<th")
    expected_tinted = html.count("background-color:")
    tinted = sum(1 for cell in cells if "backgroundColor" in (cell.get("data") or {}))
    problems = []
    if len(tables) != expected_tables:
        problems.append(f"表格数 {len(tables)}，源文件 {expected_tables}")
    if len(cells) != expected_cells:
        problems.append(f"单元格数 {len(cells)}，源文件 {expected_cells}")
    if tinted != expected_tinted:
        problems.append(f"带背景色单元格 {tinted}，源文件声明 {expected_tinted}")
    if problems:
        raise SystemExit("回读校验不通过：" + "；".join(problems))
    bordered = sum(1 for cell in cells if "borderColor" in (cell.get("data") or {}))
    print(f"[ku] 回读校验通过：{len(tables)} 张表、{len(cells)} 个单元格、"
          f"{bordered} 个带边框、{tinted} 个带背景色")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="result package directory (holds final_report.md)")
    parser.add_argument("--url", help="知识库文档链接，从中解析 doc id")
    parser.add_argument("--doc-id", help="文档 ID，与 --url 二选一")
    parser.add_argument("--username", default=os.getenv("BAIDU_CC_USERNAME")
                        or os.getenv("SANDBOX_USERNAME"),
                        help="编辑者用户名，默认取 BAIDU_CC_USERNAME")
    parser.add_argument("--ku-bin", help="ku-doc-manage/bin/ku 路径")
    parser.add_argument("--report", help="报告路径，默认 <package>/final_report.md")
    args = parser.parse_args()

    if not args.username:
        raise SystemExit("缺少用户名：用 --username 指定或设置 BAIDU_CC_USERNAME")
    report = Path(args.report) if args.report else Path(args.package) / "final_report.md"
    if not report.is_file():
        raise SystemExit(f"找不到报告：{report}")
    # Read-only: the package's own report stays exactly as generated.
    html, rewritten = normalise(report.read_text())
    if "<!-- TODO" in html:
        raise SystemExit(
            f"{report.name} 仍有未填写的 <!-- TODO --> 标记，先补齐再发布"
        )
    if rewritten:
        print(f"[ku] 已将 {rewritten} 处 background 简写改写为 background-color（仅影响发布内容）")

    ku = resolve_ku_binary(args.ku_bin)
    doc_id = document_id(args.url, args.doc_id)
    print(f"[ku] 目标文档 {doc_id}，使用 {ku}")

    cover = json.dumps([{
        "mode": "cover",
        "json": [{"type": "paragraph",
                  "children": [{"type": "text", "text": PLACEHOLDER}]}],
    }], ensure_ascii=False)
    run(ku, "edit-content", "--doc-id", doc_id, "--username", args.username,
        "--editor-mode", "cover", "--operations", cover)
    operation = json.dumps({
        "mode": "replace_range",
        "selectionWithEllipsis": PLACEHOLDER,
        "markdown": html,
    }, ensure_ascii=False)
    run(ku, "edit-mdsl-content", "--doc-id", doc_id, "--username", args.username,
        "--operation", operation)
    # Without this the API has reported success but readers still see the old page.
    run(ku, "publish-doc", "--doc-id", doc_id, "--username", args.username)
    verify(ku, doc_id, html)
    if args.url:
        print(f"[ku] 已发布：{args.url}")
    else:
        print(f"[ku] 已发布文档 {doc_id}")


if __name__ == "__main__":
    main()
