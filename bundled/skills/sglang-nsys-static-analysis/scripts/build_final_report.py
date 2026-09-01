#!/usr/bin/env python3
"""Write the human-readable `final_report.md` for a finished analysis package.

Every number in the report already exists in the package's tables, so the
mechanical half is generated here instead of being retyped by hand: the four
tables, their totals and percentages. The judgement half -- 结论, 潜在优化点,
分析思路, the model-structure line -- is left as `<!-- TODO ... -->` markers for
the agent to fill in.

The tables are emitted as inline-styled HTML rather than markdown pipe tables
because the report is pasted into 如流知识库, whose editor keeps cell-level
styling (borders, background, alignment) but drops table/column widths and
`<caption>` entirely, and adds a blank line wherever the preceding block has a
bottom margin. Hence: captions as a `<p style="margin:0">` right above the
table, `margin:0` on every block, no blank line before a table, and no width
declarations at all -- column widths come from the header wording.
"""
from __future__ import annotations

import argparse
import csv
import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

TABLE_OPEN = (
    '<table border="1" cellspacing="0" cellpadding="6" '
    'style="border-collapse:collapse;border:1px solid #999;text-align:center;margin:0">'
)
CELL = "border:1px solid #999;text-align:center;vertical-align:middle"
HEAD_BG = "#b4c7e7"   # header row
LABEL_BG = "#d9e2f3"  # first column
SPACER = '<p style="margin:0">&nbsp;</p>'


def th(text: str, background: str) -> str:
    return f'<th style="{CELL};background:{background}">{text}</th>'


def td(text: str) -> str:
    return f'<td style="{CELL}">{text}</td>'


def table(title: str, header: list[str], rows: list[tuple[str, list[str]]], *,
          bold_title: bool = True, note: str | None = None) -> str:
    """One report table: optional note line, bold title line, then the table.

    `header` is the full first row including its leading label cell; each entry in
    `rows` is `(row label, cells)`.
    """
    lines: list[str] = []
    if note:
        lines.append(f'<p style="margin:0">{note}</p>')
    heading = f"<b>{title}</b>" if bold_title else title
    lines.append(f'<p style="margin:0">{heading}</p>')
    lines.append(TABLE_OPEN)
    lines.append("<tr>")
    lines.extend(th(cell, HEAD_BG) for cell in header)
    lines.append("</tr>")
    for label, cells in rows:
        lines.append("<tr>")
        lines.append(th(label, LABEL_BG))
        lines.extend(td(cell) for cell in cells)
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def round_half_up(value: float, digits: int) -> str:
    """Half-up rounding, so 24.325 ms reads 24.33 like the CSV-derived numbers.

    Python's format uses round-half-even, which reports 24.32 for the same value
    and makes the report disagree with hand checks against the tables.
    """
    quantum = Decimal(1).scaleb(-digits)
    return str(Decimal(str(value)).quantize(quantum, rounding=ROUND_HALF_UP))

def ms(value: str | float, digits: int = 2) -> str:
    try:
        return round_half_up(float(value) / 1000, digits)
    except (TypeError, ValueError):
        return "—"


def pct(value: str | float, digits: int = 2) -> str:
    try:
        return round_half_up(float(value), digits) + "%"
    except (TypeError, ValueError):
        return "—"


def forward_tables(rows: list[dict[str, str]]) -> list[str]:
    """Tables 1 and 2: the forward step's split, then target's own children.

    The pipeline table lists target's children between the target phase row and
    the draft phase row (or the gap row), the same positional split the frontend
    uses -- the rows carry no parent column.
    """
    if not rows:
        return []
    kinds = [row.get("环节类型", "") for row in rows]
    total = next((row for row, kind in zip(rows, kinds) if kind == "total"), None)
    phases = [row for row, kind in zip(rows, kinds) if kind == "phase"]
    gap = next((row for row, kind in zip(rows, kinds) if kind == "gap"), None)
    if total is None or not phases:
        return []
    target, draft = phases[0], (phases[1] if len(phases) > 1 else None)
    draft_index = rows.index(draft) if draft is not None else len(rows)
    children = [
        row for row in rows[rows.index(target) + 1:draft_index]
        if row.get("环节类型") in {"stage", "variant", "other"}
    ]

    step_us = float(total.get("总耗时(us)") or 0) or 1.0
    draft_time = ms(draft.get("总耗时(us)")) if draft else "—（未启用投机）"
    draft_pct = pct(draft.get("占forward步(%)")) if draft else "—"
    first = table(
        "Token 链路耗时",
        ["阶段", "Forward 耗时", "Target 耗时", "Draft 耗时", "Token间间隙"],
        [
            ("耗时(ms)", [
                ms(total.get("总耗时(us)")), ms(target.get("总耗时(us)")),
                draft_time, ms(gap.get("总耗时(us)")) if gap else "—",
            ]),
            ("耗时百分比", [
                "100%", pct(target.get("占forward步(%)")), draft_pct,
                pct(gap.get("占forward步(%)")) if gap else "—",
            ]),
        ],
        note=None,
    )

    names, times, shares, per_layer = [], [], [], []
    for row in children:
        layers = (row.get("层数") or "").strip()
        name = row.get("环节", "")
        names.append(f"{name} × {layers}" if layers else name)
        times.append(ms(row.get("总耗时(us)")))
        shares.append(pct(row.get("占forward步(%)")))
        per_layer.append(ms(row.get("单次耗时(us)")) if layers else "—")
    second = table(
        "Target 内部构成",
        ["环节", *names],
        [("耗时(ms)", times), ("占 forward step", shares), ("单层耗时(ms)", per_layer)],
    )
    return [first, second]


def module_table(stage_rows: list[dict[str, str]], operator_rows: list[dict[str, str]],
                 pattern_us: float) -> str | None:
    """Table 3: functional modules summed over the whole repeating unit.

    Column order is each module's first appearance in the operator table, which is
    emitted in execution order -- the stage table is sorted by duration.
    """
    totals: dict[str, float] = {}
    for row in stage_rows:
        if not (row.get("序号") or "").strip().isdigit():
            continue  # the trailing 总计 row
        name = row.get("功能模块") or ""
        if name:
            totals[name] = totals.get(name, 0.0) + float(row.get("模块耗时(us)") or 0)
    if not totals:
        return None
    order: list[str] = []
    for row in operator_rows:
        name = row.get("功能模块") or ""
        if name in totals and name not in order:
            order.append(name)
    order += [name for name in totals if name not in order]

    header = ["功能模块", "pattern", *order]
    times = [ms(pattern_us), *(ms(totals[name]) for name in order)]
    # Percentages come from the durations, not from summing the stage table's
    # already-rounded per-row percentages: a module split across four layers would
    # otherwise accumulate its rounding (6.735 -> 6.74 instead of 6.73).
    shares = ["100%", *(pct(totals[name] / pattern_us * 100) for name in order)]
    return table(
        "按功能模块划分（按执行顺序）",
        header,
        [("耗时(ms)", times), ("耗时百分比", shares)],
    )


def category_table(rows: list[dict[str, str]]) -> str | None:
    """Table 4: core / communication / auxiliary counts and time."""
    data = [row for row in rows if (row.get("序号") or "").strip().isdigit()]
    if not data:
        return None
    names = [row.get("算子类型", "") for row in data]
    names = [("小算子（辅助算子）" if name == "辅助算子" else name) for name in names]
    return table(
        "按算子类型划分",
        ["算子类型", *names],
        [
            ("算子数量", [row.get("算子数量", "") for row in data]),
            ("耗时(ms)", [ms(row.get("总耗时(us)")) for row in data]),
            ("耗时百分比", [pct(row.get("耗时占比(%)")) for row in data]),
        ],
    )


def build(package: Path, prefix: str) -> str:
    csv_dir = package / "csv" if (package / "csv").is_dir() else package
    metadata_dir = package / "metadata"
    if not metadata_dir.is_dir() and (csv_dir.parent / "metadata").is_dir():
        metadata_dir = csv_dir.parent / "metadata"

    manifest = read_json(metadata_dir / f"{prefix}_analysis_manifest.json")
    context = read_json(metadata_dir / "context.json")
    pipeline = read_json(metadata_dir / f"{prefix}_forward_pipeline.json").get(
        "forward_pipeline", {})
    stage_rows = read_csv(csv_dir / f"{prefix}_stage_table.csv")
    operator_rows = read_csv(csv_dir / f"{prefix}_opreator_table.csv")
    pattern_us = float(manifest.get("total_duration_us") or 0) or 1.0

    trace = Path(context.get("sqlite_path") or "")
    stage = manifest.get("stage") or context.get("stage") or "—"
    devices = pipeline.get("device")
    model = context.get("model_name") or prefix
    head = [
        f"# {model} {stage} 性能分析报告",
        "",
        f"- 结果包：`{package}`",
        f"- 硬件：{manifest.get('hardware') or '—'}"
        + (f"；采样 rank：device {devices}" if devices is not None else ""),
        f"- 阶段：{stage}；trace：`{trace.name or '—'}`",
        "- 模型结构：<!-- TODO 层数与变体构成，例如 45 层 = 11 × DSA-MoE + 34 × KDA-MoE；是否启用 MTP -->",
        "",
        "# 1. 结论",
        "",
        "<!-- TODO 3~5 条结论，每条一句话给出事实 + 数字 + 影响，按重要性排序 -->",
        "",
        "#### 潜在优化点",
        "",
        "<!-- TODO 按收益排序的优化项，只写有数据支撑的 -->",
        "",
        "",
    ]

    body = ['<h1 style="margin:0">2. 链路与算子耗时分析</h1>']
    body.append('<p style="margin:0"><b>结论</b>：<!-- TODO 本节一句话结论 --></p>')
    body.append(SPACER)
    body.append('<p style="margin:0">以下是具体分析过程：</p>')
    body.append('<p style="margin:0"><b>分析思路</b>：<!-- TODO 重复单元的选取依据与单元耗时 --></p>')

    forward = forward_tables(read_csv(csv_dir / f"{prefix}_forward_pipeline_table.csv"))
    if forward:
        body.append('<h2 style="margin:0">2.1 forward 链路耗时</h2>')
        config = [
            f"chunked-prefill-size = {manifest['chunk_size']}"
            if manifest.get("chunk_size") else None,
            f"batch size = {manifest['batch_size']}" if manifest.get("batch_size") else None,
        ]
        note = "当前配置：" + "、".join(item for item in config if item) if any(config) else None
        body.append(
            (f'<p style="margin:0">{note}<!-- TODO TP/EP/PP、CUDA graph 等 --></p>\n'
             if note else "") + forward[0])
        body.append(forward[1])

    body.append('<h2 style="margin:0">2.2 算子耗时分析</h2>')
    covered = sum(
        float(row.get("模块耗时(us)") or 0) for row in stage_rows
        if (row.get("序号") or "").strip().isdigit()
    )
    # Modules can sum above the unit wall span when variants run on their own
    # streams, and calling that excess "unclassified leftovers" -- as this line
    # used to unconditionally -- is both self-contradictory and wrong.
    if covered > pattern_us:
        body.append(
            '<p style="margin:0">以下口径为<b>一个重复单元</b>内、稳定样本逐算子平均耗时之和。'
            f"单元墙钟 {ms(pattern_us)} ms，各模块累计 {ms(covered)} ms，"
            f"因部分模块在独立 CUDA 流上与主流并行而超出墙钟 "
            f"{covered / pattern_us * 100 - 100:.1f}%，按实测原样呈现、不归一化到 100%。</p>"
        )
    else:
        body.append(
            '<p style="margin:0">以下口径为<b>一个重复单元</b>内、稳定样本逐算子平均耗时之和，'
            f"单元合计 {ms(pattern_us)} ms，下表覆盖其中 {ms(covered)} ms"
            f"（{covered / pattern_us * 100:.1f}%，余量为未归类的零散算子）。</p>"
        )
    modules = module_table(stage_rows, operator_rows, pattern_us)
    if modules:
        body.append(modules)
    body.append("<!-- TODO 模块层面的补充说明，例如各变体的出现层数、单层最贵的 kernel -->")
    body.append(SPACER)
    categories = category_table(read_csv(csv_dir / f"{prefix}_op_classification_table.csv"))
    if categories:
        body.append(categories)
    body.append(SPACER)
    # A list, not a comma-run: five kernel names with two numbers each is what a
    # reviewer scans to pick a fusion target.
    body.append('<p style="margin:0"><b>小算子 Top 5</b>（单元内合计 / 启动次数）</p>')
    body.append('<ul style="margin:0;padding-left:22px">')
    body.append("<!-- TODO 每行一个：<li><code>kernel</code> X.XX ms / N 次</li> -->")
    body.append("</ul>")
    body.append("<!-- TODO 效率上界参考：本包内 MFU / MBU 最高值，以及占比最大算子的实测值 -->")

    tail = [
        "",
        "# 3. 算子分析工具数据",
        "",
        "popo 发布页面链接：待补（人工发布后填入）",
        "",
        "# 4. 物料",
        "",
        f"nsys 文件：`{trace or '—'}`"
        + (f"（包内副本 `trace/{trace.name}`）" if trace.name else ""),
        "",
    ]
    conflicts = pipeline.get("declaration_conflicts") or []
    if conflicts:
        tail[1:1] = ["", "> ⚠ 与 config/启动命令声明不一致：" + "；".join(conflicts)]
    return "\n".join(head) + "\n".join(body) + "\n".join(tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="result package directory (holds csv/ and metadata/)")
    parser.add_argument("--prefix", default="analysis")
    parser.add_argument("--output", default=None, help="defaults to <package>/final_report.md")
    args = parser.parse_args()

    package = Path(args.package).resolve()
    output = Path(args.output) if args.output else package / "final_report.md"
    output.write_text(build(package, args.prefix))
    print(f"[final-report] wrote {output}")


if __name__ == "__main__":
    main()
