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

Attachments are a second pass, because mdsl cannot express one: only the editor
JSON has an `attachment` node. So the trace and the packed result directory are
uploaded, the just-written page is read back as JSON (a lossless round trip --
the read-back already carries every style the API kept), the two 输入配置/输出物料
cells get their contents swapped for attachment nodes, and the whole thing goes
back through `cover`. Upload is capped server-side at 64 MiB (70 MiB already
answers 413): the `.nsys-rep` goes up as-is and its cell is left blank when the
file is missing or over the cap, while the result-directory zip falls back to
repacking without `trace/`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

PLACEHOLDER = "NSYSSCOPE-REPORT-PLACEHOLDER"
# Probed against the live API: 64 MiB uploads, 70 MiB answers HTTP 413.
UPLOAD_LIMIT = 64 * 1024 * 1024
TRACE_CELL = "nsys 文件"
ARTIFACT_CELL = "工具产物"


def resolve_ku_binary(explicit: str | None) -> Path:
    """Locate the `ku` CLI from the ku-doc-manage Skill.

    `Path.home()` is not enough on its own: this script often runs as a different
    user than the one whose home holds the checkout (a service, or a root shell),
    so the installed Comate Skill directory and the checkout next to this project
    are searched too.
    """
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
    roots = [Path.home()]
    username = os.getenv("BAIDU_CC_USERNAME")
    if username:
        roots.append(Path("/home/users") / username)
    # The project checkout's own parent: NsysScope and ku-doc-manage sit side by side.
    roots.append(Path(__file__).resolve().parents[4].parent)
    for root in roots:
        candidates.append(root / "ku-doc-manage" / "bin" / "ku")
        candidates.append(root / ".comate" / "skills" / ".system" / "ku-doc-manage" / "bin" / "ku")
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


def set_title(ku: Path, doc_id: str, username: str, title: str,
              attempts: int = 3) -> None:
    """Name the page after the report, checking the name that actually stuck.

    A rename can come back with a `(1)` suffix appended -- the platform
    de-duplicates against the name the page is being renamed away from, so the
    first attempt on a freshly written page can land as `…结果(1)`. Renaming
    again with the same string then takes cleanly, so the name is read back and
    retried rather than assumed.
    """
    for attempt in range(1, attempts + 1):
        run(ku, "rename-doc", "--doc-id", doc_id, "--username", username,
            "--new-name", title)
        data = run(ku, "query-content", "--doc-id", doc_id,
                   "--protocol", "markdown", "--show-doc-info")
        actual = ((data.get("result") or {}).get("docInfo") or {}).get("name")
        if actual == title:
            print(f"[ku] 文档标题已设为「{title}」")
            return
        if attempt < attempts:
            print(f"[ku] 标题落为「{actual}」，重试一次")
            time.sleep(1)
    print(f"[ku] 标题最终为「{actual}」，与报告标题「{title}」不一致，请手动确认")


def normalise(html: str) -> tuple[str, int]:
    """Rewrite the `background` shorthand the API drops into the longhand."""
    shorthand = re.compile(r"background:(?!-)")
    return shorthand.sub("background-color:", html), len(shorthand.findall(html))


def split_title(html: str) -> tuple[str | None, str]:
    """Take the report's own outer heading off the body, to use as the page title.

    A 如流 page already renders its document name as the heading at the top, so
    keeping the report's first `<h1>` in the body would show the same line twice.
    The heading is the report's title by construction (build_final_report.py emits
    the `{model}-{stage}-{规模}` banner, e.g. `DeepSeek-V4.1-Flash-decode-bs256`,
    first), so it becomes the
    document name and is dropped from the content.
    """
    match = re.match(r'\s*<h1[^>]*>(.*?)</h1>\s*\n?', html, re.S)
    if not match:
        return None, html
    title = re.sub(r"<[^>]+>", "", match.group(1)).strip()
    if not title:
        return None, html
    return title, html[match.end():]


def find_nsys_rep(package: Path) -> Path | None:
    """The capture as 如流 should carry it: the original `.nsys-rep`, uncompressed.

    Looked up in the package's own `trace/` first, then beside the report path
    the job recorded (a package usually keeps the exported `.sqlite`, while the
    `.nsys-rep` stays where it was captured). Returns None when there is no
    `.nsys-rep` or it is over the upload cap -- the cell then keeps the file name
    the report already shows, rather than a compressed or substituted stand-in.
    """
    candidates = sorted((package / "trace").glob("*.nsys-rep")) \
        if (package / "trace").is_dir() else []
    try:
        context = json.loads((package / "metadata" / "context.json").read_text())
    except (OSError, json.JSONDecodeError):
        context = {}
    recorded = context.get("report_path") or context.get("report")
    if recorded:
        beside = Path(recorded)
        for candidate in (beside.with_suffix(".nsys-rep"), beside):
            if candidate.suffix == ".nsys-rep" and candidate.is_file():
                candidates.append(candidate)
    for candidate in candidates:
        size = candidate.stat().st_size
        if size <= UPLOAD_LIMIT:
            return candidate
        print(f"[ku] {candidate.name} {size / 1e6:.0f} MB 超过上传上限 "
              f"{UPLOAD_LIMIT / 1e6:.0f} MB，且 nsys 文件不做压缩，该格保持原文件名")
    if not candidates:
        print("[ku] 未找到 .nsys-rep（包内 trace/ 与采集路径旁都没有），nsys 文件格保持原文件名")
    return None


def pack_package(package: Path, workdir: Path, skip_trace: bool = False) -> Path:
    """Zip the result directory; drops trace/ when the archive would be too big.

    Dropping trace/ is the right fallback rather than giving up on the artifact:
    the capture is attached separately in the same page, so the only thing lost
    from the archive is a copy of a file the reader already has.
    """
    archive = workdir / f"{package.name}{'-no-trace' if skip_trace else ''}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in sorted(package.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(package)
            if skip_trace and relative.parts and relative.parts[0] == "trace":
                continue
            bundle.write(path, Path(package.name) / relative)
    if archive.stat().st_size > UPLOAD_LIMIT and not skip_trace:
        archive.unlink()
        return pack_package(package, workdir, skip_trace=True)
    print(f"[ku] 结果目录已打包 {archive.name} "
          f"{archive.stat().st_size / 1e6:.0f} MB"
          + ("（已排除 trace/，capture 单独作为附件）" if skip_trace else ""))
    return archive


def upload(ku: Path, doc_id: str, path: Path) -> dict:
    """Upload one attachment and return the node payload for it."""
    if path.stat().st_size > UPLOAD_LIMIT:
        raise SystemExit(
            f"{path.name} {path.stat().st_size / 1e6:.0f} MB 超过上传上限 "
            f"{UPLOAD_LIMIT / 1e6:.0f} MB，无法作为附件"
        )
    data = run(ku, "upload-attachment", "--doc-id", doc_id, "--file", str(path))
    result = data.get("result") or {}
    attach_id = result.get("attachId")
    if not attach_id:
        raise SystemExit(f"上传 {path.name} 未返回 attachId：{data}")
    print(f"[ku] 已上传附件 {result.get('name')} "
          f"{(result.get('size') or 0) / 1e6:.1f} MB")
    return {
        "type": "attachment",
        "fileId": attach_id,
        "fileInfo": {
            "name": result.get("name") or path.name,
            "size": result.get("size") or path.stat().st_size,
            "extension": result.get("extension") or path.suffix.lstrip("."),
            "type": "application/octet-stream",
        },
        "viewType": "card",
        # Every other cell in these tables is centred, and the API keeps
        # textAlign on an attachment node just as it does on a paragraph.
        "textAlign": "center",
        "invalid": False,
        "children": [{"text": ""}],
        "docId": doc_id,
        "url": "",
    }


def cell_text(cell: dict) -> str:
    """Flatten a table cell to plain text so a label can be matched."""
    parts: list[str] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if isinstance(node.get("text"), str):
                parts.append(node["text"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(cell)
    return "".join(parts).strip()


def attach_into_cells(blocks: list, replacements: dict[str, dict]) -> list[str]:
    """Swap a labelled row's value cell for an attachment node.

    Sections 1 and 3 are vertical fact tables: each row is `label | value`, so
    the row whose first cell reads `nsys 文件` or `工具产物` is the one to patch.
    Only cells with a file to offer are listed in `replacements`; a cell with no
    attachment is left exactly as generated, keeping the file name it already
    shows.

    The replacement inherits the cell's own alignment instead of hardcoding
    centre: section 1 is centred but section 3 is left-aligned, so reading it
    off the paragraph being replaced keeps each attachment lined up with the
    table it sits in.
    """
    filled: list[str] = []

    def alignment(cell: dict) -> str:
        # The API stores textAlign only when it differs from the editor default,
        # so section 1's centred cells carry "center" while section 3's
        # left-aligned ones carry nothing. An absent value therefore means left,
        # not centre -- defaulting to centre put a centred attachment into a
        # left-aligned table.
        for child in cell.get("children") or []:
            if isinstance(child, dict) and child.get("textAlign"):
                return child["textAlign"]
        return "left"

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "table-row":
                cells = node.get("children") or []
                if len(cells) == 2:
                    label = cell_text(cells[0])
                    if label in replacements and label not in filled:
                        fresh = json.loads(json.dumps(replacements[label]))
                        fresh["textAlign"] = alignment(cells[1])
                        cells[1]["children"] = [fresh]
                        filled.append(label)
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk(blocks)
    return filled


def inner_blocks(content: object) -> list:
    """The editable blocks inside the platform's own title/card wrapper."""
    if not isinstance(content, list):
        raise SystemExit("读回的文档内容不是块数组")
    for node in content:
        if isinstance(node, dict) and node.get("type") == "card":
            children = node.get("children") or []
            if children and isinstance(children[0], dict):
                return children[0].get("children") or []
    raise SystemExit("读回的文档里找不到 card 容器")


# Linux caps a single argv entry at MAX_ARG_STRLEN (128 KiB) regardless of the
# much larger ARG_MAX, and `ku` only takes the operation JSON as an argument --
# no @file form. A read-back document carries a blockId/diffId/renderKey on every
# node, which for a 350-cell report is over half the payload and all of it
# regenerated server-side, so dropping those ids is what keeps `cover` under the
# limit instead of failing with E2BIG.
VOLATILE_KEYS = {"blockId", "diffId", "renderKey", "id"}
ARG_LIMIT = 128 * 1024


def strip_volatile(node: object) -> object:
    if isinstance(node, dict):
        return {key: strip_volatile(value) for key, value in node.items()
                if key not in VOLATILE_KEYS}
    if isinstance(node, list):
        return [strip_volatile(item) for item in node]
    return node


def cover_operations(blocks: list) -> str:
    payload = json.dumps([{"mode": "cover", "json": strip_volatile(blocks)}],
                         ensure_ascii=False, separators=(",", ":"))
    size = len(payload.encode())
    if size > ARG_LIMIT:
        raise SystemExit(
            f"cover 负载 {size} 字节超过单参数上限 {ARG_LIMIT}，"
            "无法通过 ku CLI 写入（报告正文已发布，仅附件未写入）"
        )
    return payload


def verify(ku: Path, doc_id: str, html: str, expected_attachments: int = 0,
           attempts: int = 6, pause: float = 2.0) -> None:
    """Read the published page back and compare it with what we sent.

    Reporting "published" on the strength of a 200 would hide the two failure
    modes that actually happen: content parsed into nothing (an unsupported tag
    silently drops its whole block) and an edit stuck in the edit state. Both
    show up as a table or cell count that does not match the source.

    `publish-doc` answers before the preview state has caught up (its response
    even carries a `delayMs`), so a mismatch is retried a few times before it
    counts as a failure -- otherwise a correct publish reads as broken.
    """
    for attempt in range(1, attempts + 1):
        problems, summary = inspect_page(ku, doc_id, html, expected_attachments)
        if not problems:
            print(summary)
            return
        if attempt < attempts:
            time.sleep(pause)
    raise SystemExit("回读校验不通过：" + "；".join(problems))


def inspect_page(ku: Path, doc_id: str, html: str,
                 expected_attachments: int) -> tuple[list[str], str]:
    data = run(ku, "query-content", "--doc-id", doc_id, "--protocol", "json")
    content = (data.get("result") or {}).get("content")
    tables: list[dict] = []
    attachments: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, dict):
            if node.get("type") == "table":
                tables.append(node)
            if node.get("type") == "attachment":
                attachments.append(node)
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
    if len(attachments) != expected_attachments:
        problems.append(f"附件数 {len(attachments)}，期望 {expected_attachments}")
    bordered = sum(1 for cell in cells if "borderColor" in (cell.get("data") or {}))
    summary = (f"[ku] 回读校验通过：{len(tables)} 张表、{len(cells)} 个单元格、"
               f"{bordered} 个带边框、{tinted} 个带背景色")
    if expected_attachments:
        summary += f"、{len(attachments)} 个附件"
    return problems, summary


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
    parser.add_argument("--no-attach", dest="attach", action="store_false",
                        help="只写报告正文，不上传 nsys 与结果目录附件")
    parser.set_defaults(attach=True)
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
    title, html = split_title(html)

    ku = resolve_ku_binary(args.ku_bin)
    doc_id = document_id(args.url, args.doc_id)
    print(f"[ku] 目标文档 {doc_id}，使用 {ku}")
    if title:
        set_title(ku, doc_id, args.username, title)

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

    if args.attach:
        publish_attachments(ku, doc_id, args.username, Path(args.package), html)
    if args.url:
        print(f"[ku] 已发布：{args.url}")
    else:
        print(f"[ku] 已发布文档 {doc_id}")


def apply_attachments(ku: Path, doc_id: str, username: str,
                      nodes: dict[str, dict], html: str,
                      expected: int, attempts: int = 5,
                      pause: float = 8.0) -> list[str]:
    """Patch the labelled cells with attachment nodes, retrying until it sticks.

    A `cover` issued soon after the mdsl write's `publish-doc` is accepted (200,
    "完成编辑") and then silently lost: the publish is still settling
    server-side, so the cover lands against the pre-mdsl revision and is
    discarded. A fixed delay works but picks a magic number; re-reading the page
    and re-applying until the attachments are actually visible is
    self-correcting, and each round is idempotent because the patch is derived
    from the page as it currently reads.
    """
    problems: list[str] = []
    for attempt in range(1, attempts + 1):
        data = run(ku, "query-content", "--doc-id", doc_id, "--protocol", "json")
        blocks = inner_blocks((data.get("result") or {}).get("content"))
        filled = attach_into_cells(blocks, nodes)
        missing = [label for label in nodes if label not in filled]
        if missing:
            raise SystemExit(
                "报告里找不到这些字段所在的表格行，附件未写入：" + "、".join(missing)
            )
        run(ku, "edit-content", "--doc-id", doc_id, "--username", username,
            "--editor-mode", "cover", "--operations", cover_operations(blocks))
        run(ku, "publish-doc", "--doc-id", doc_id, "--username", username)
        problems, _ = inspect_page(ku, doc_id, html, expected)
        if not problems:
            return filled
        if attempt < attempts:
            print(f"[ku] 附件尚未生效（{'；'.join(problems)}），{pause:.0f}s 后重试"
                  f"（{attempt}/{attempts - 1}）")
            time.sleep(pause)
    raise SystemExit("附件多次写入后仍未生效：" + "；".join(problems))


def publish_attachments(ku: Path, doc_id: str, username: str, package: Path,
                        html: str) -> None:
    """Put the capture and the packed result directory into their own cells."""
    with tempfile.TemporaryDirectory(prefix="nsysscope-ku-") as scratch:
        workdir = Path(scratch)
        nodes: dict[str, dict] = {}
        report = find_nsys_rep(package)
        if report is not None:
            nodes[TRACE_CELL] = upload(ku, doc_id, report)
        nodes[ARTIFACT_CELL] = upload(ku, doc_id, pack_package(package, workdir))
        filled = apply_attachments(ku, doc_id, username, nodes, html, len(nodes))
    verify(ku, doc_id, html, expected_attachments=len(nodes))
    print(f"[ku] 已把 {'、'.join(filled)} 替换为附件"
          + ("" if TRACE_CELL in nodes else f"；{TRACE_CELL} 保持原文件名不变"))


if __name__ == "__main__":
    main()
