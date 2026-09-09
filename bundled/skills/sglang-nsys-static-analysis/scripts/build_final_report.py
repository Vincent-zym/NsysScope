#!/usr/bin/env python3
"""Write the human-readable `final_report.md` for a finished analysis package.

Every number in the report already exists in the package's tables, so the
mechanical half is generated here instead of being retyped by hand: the input
configuration facts already on disk, and the timing/operator tables with their
totals and percentages. The judgement half -- 分析思路, engine config the manifest
doesn't carry structurally, the model-structure line -- is left as
`<!-- TODO ... -->` markers for the agent to fill in.

This is a pure timeline/operator-timing report: no 结论 section, no 潜在优化点, no
prose interpretation anywhere. Every line is a fact with a number behind it or a
`<!-- TODO -->` for a fact the manifest doesn't carry structurally.

The whole document is inline-styled HTML, including every heading, because the
report is pasted into 如流知识库, whose editor keeps cell-level styling (borders,
background, alignment) but drops table/column widths and `<caption>` entirely,
and adds a blank line wherever the preceding block (including a markdown `#`
heading, which carries its own default margin the editor does not let this
file override) has a bottom margin. Hence: every heading and paragraph is
`<h1|h2|h3 style="margin:0">`/`<p style="margin:0">`, never markdown `#`; no
blank line anywhere in the source between one block and the next; captions as
a `<p style="margin:0">` right above their table; and no width declarations at
all -- column widths come from the header wording, and every table (including
the vertical fact tables in sections 1 and 3) is centred (`text-align:center`)
so a column is never narrower than its own content.

The same HTML also goes into 如流 through its open API (see
`publish_report_to_ku.py`), which parses these inline styles into the editor's
own cell attributes. That path is stricter than pasting on exactly one point:
it reads `background-color` and ignores the `background` shorthand, so this
file always emits the longhand.
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


def config_table(rows: list[tuple[str, str]]) -> str:
    """A vertical fact table: one row per field, label column then value column.

    Unlike `table()`, which lays entities out as columns for comparison, this is
    for facts that only ever have one value each -- forcing them into `table()`'s
    shape would need a table with one column and be unreadable. Centred and with
    no width declaration, same as every other table in this report, so a column
    is never narrower than its own content.
    """
    lines = [
        '<table border="1" cellspacing="0" cellpadding="6" '
        'style="border-collapse:collapse;border:1px solid #999;'
        'text-align:center;margin:0">'
    ]
    for label, value in rows:
        lines.append("<tr>")
        lines.append(th(label, LABEL_BG))
        lines.append(td(value))
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def th(text: str, background: str) -> str:
    # `background-color`, not the `background` shorthand: 如流's open API parses the
    # longhand into the cell's backgroundColor and silently drops the shorthand, so
    # a report written through publish_report_to_ku.py would lose every header tint.
    # Manual pasting accepts either, so the longhand is the one that works for both.
    return f'<th style="{CELL};background-color:{background}">{text}</th>'


def td(text: str) -> str:
    return f'<td style="{CELL}">{text}</td>'


def table(title: str | None, header: list[str], rows: list[tuple[str, list[str]]], *,
          bold_title: bool = True, note: str | None = None) -> str:
    """One report table: optional note line, optional bold title line, then the table.

    `header` is the full first row including its leading label cell; each entry in
    `rows` is `(row label, cells)`. `title=None` omits the caption line entirely --
    the report's brevity rule treats a table's own header row as sufficient context,
    so 2.1/2.2.1/2.2.3's captions ("Token 链路耗时" / "Target 内部构成" /
    "按算子类型划分") were dropped as redundant prose.
    """
    lines: list[str] = []
    if note:
        lines.append(f'<p style="margin:0">{note}</p>')
    if title:
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


def forward_tables(rows: list[dict[str, str]]) -> list[str | None]:
    """Tables for 2.1/2.2.1/2.3.1: the forward step's split, target's own
    children, then draft's own children when the capture has a draft phase.

    The pipeline table lists target's children between the target phase row and
    the draft phase row, and draft's children (if any) between the draft phase
    row and the gap row -- the same positional split the frontend uses, since
    the rows carry no parent column.

    Draft's children come from one of two upstream code paths and are not
    always broken down the same way: a CUDA-graph capture can only place the
    draft window's boundary, not segment inside it, so draft_rows there is a
    single "draft N 层 forward" stage row plus an 其他 residual; an eager
    capture with a detected MTP/NextN layer segments draft by variant exactly
    like target. Either shape renders through the same table -- this function
    does not need to know which one it got.
    """
    if not rows:
        return [None, None, None]
    kinds = [row.get("环节类型", "") for row in rows]
    total = next((row for row, kind in zip(rows, kinds) if kind == "total"), None)
    phases = [row for row, kind in zip(rows, kinds) if kind == "phase"]
    gap = next((row for row, kind in zip(rows, kinds) if kind == "gap"), None)
    if total is None or not phases:
        return [None, None, None]
    target, draft = phases[0], (phases[1] if len(phases) > 1 else None)
    draft_index = rows.index(draft) if draft is not None else len(rows)
    gap_index = rows.index(gap) if gap is not None else len(rows)
    target_children = [
        row for row in rows[rows.index(target) + 1:draft_index]
        if row.get("环节类型") in {"stage", "variant", "other"}
    ]
    draft_children = [
        row for row in rows[draft_index + 1:gap_index]
        if draft is not None and row.get("环节类型") in {"stage", "variant", "other"}
    ]

    step_us = float(total.get("总耗时(us)") or 0) or 1.0
    draft_time = ms(draft.get("总耗时(us)")) if draft else "—（未启用投机）"
    draft_pct = pct(draft.get("占forward步(%)")) if draft else "—"
    first = table(
        None,
        ["阶段", "Forward step", "Target 主模型", "Draft 模型", "Token间间隙"],
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

    def children_table(children: list[dict[str, str]], phase: dict[str, str],
                       total_label: str) -> str:
        """One phase's children, with the phase's own total as the first column.

        The leading total column is what lets a reader check the split without
        scrolling back to 2.1: the children sum to it (plus 其他), so a column
        that does not add up is visible in place.
        """
        names = [total_label]
        times = [ms(phase.get("总耗时(us)"))]
        shares = [pct(phase.get("占forward步(%)"))]
        per_layer = ["—"]
        for row in children:
            layers = (row.get("层数") or "").strip()
            name = row.get("环节", "")
            names.append(f"{name} × {layers}" if layers else name)
            times.append(ms(row.get("总耗时(us)")))
            shares.append(pct(row.get("占forward步(%)")))
            per_layer.append(ms(row.get("单次耗时(us)")) if layers else "—")
        return table(
            None,
            ["环节", *names],
            [("耗时(ms)", times), ("占 forward step", shares), ("单层耗时(ms)", per_layer)],
        )

    second = children_table(target_children, target, "Target 主模型耗时")
    third = (children_table(draft_children, draft, "Draft 模型耗时")
             if draft_children and draft is not None else None)
    return [first, second, third]


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

    header = ["功能模块", "pattern总耗时", *order]
    times = [ms(pattern_us), *(ms(totals[name]) for name in order)]
    # Percentages come from the durations, not from summing the stage table's
    # already-rounded per-row percentages: a module split across four layers would
    # otherwise accumulate its rounding (6.735 -> 6.74 instead of 6.73).
    shares = ["100%", *(pct(totals[name] / pattern_us * 100) for name in order)]
    return table(
        None,
        header,
        [("耗时(ms)", times), ("耗时百分比", shares)],
    )


def category_table(rows: list[dict[str, str]], pattern_us: float) -> str | None:
    """Table: core / communication / auxiliary counts and time.

    The leading pattern总耗时 column carries the unit's wall span and the total
    operator count, so the three category columns can be read as a share of the
    unit rather than of each other. The categories sum below the wall span (or
    above it, under multi-stream overlap) -- that gap is the point of showing
    the total, not an error to hide.
    """
    data = [row for row in rows if (row.get("序号") or "").strip().isdigit()]
    if not data:
        return None
    names = [row.get("算子类型", "") for row in data]
    names = [("小算子（辅助算子）" if name == "辅助算子" else name) for name in names]
    total_row = next(
        (row for row in rows if (row.get("算子类型") or "").strip() == "总计"), None)
    if total_row is not None:
        total_count = total_row.get("算子数量", "")
    else:
        total_count = str(sum(
            int(row.get("算子数量") or 0) for row in data
            if (row.get("算子数量") or "").strip().isdigit()
        ))
    return table(
        None,
        ["算子类型", "pattern总耗时", *names],
        [
            ("算子数量", [total_count, *(row.get("算子数量", "") for row in data)]),
            ("耗时(ms)", [ms(pattern_us), *(ms(row.get("总耗时(us)")) for row in data)]),
            ("耗时百分比", ["100%", *(pct(row.get("耗时占比(%)")) for row in data)]),
        ],
        bold_title=False,
    )


def abbreviate_kernel_name(name: str) -> str:
    """Shorten a demangled kernel symbol so the 算子名称 column stays readable.

    Some symbols carry a long chain of template arguments or a redundant
    dispatcher prefix (e.g. `kernel_cutlass_kernel_TgvGemmCuteExtKernel_...`,
    `fmhaSm100fKernel_QkvE4m3OBfloat16H512PagedKvDenseDynamicTokenSparse...`).
    This keeps the kernel's own identifying name plus its most distinguishing
    parameter(s) and collapses the rest into `…`, rather than truncating
    blindly, so two different specializations don't collide on the same
    abbreviation.

    Every return path escapes `<`/`>` to `&lt;`/`&gt;`, including the
    untouched short-name path: kernel symbols routinely carry their own
    literal template angle brackets (e.g. `gatherTopK<float,uint,2,false>`),
    and this string is placed inside `<code>...</code>` in the report, so an
    unescaped `<...>` gets parsed as an HTML tag and silently disappears.
    """
    def escape(text: str) -> str:
        return text.replace("<", "&lt;").replace(">", "&gt;")

    if len(name) <= 60:
        return escape(name)
    # `kernel_cutlass_kernel_<RealName>_<template args...>`: drop the
    # dispatcher prefix and keep <RealName> plus the first template arg.
    if name.startswith("kernel_cutlass_kernel_"):
        rest = name[len("kernel_cutlass_kernel_"):]
        parts = rest.split("_")
        real_name = parts[0]
        first_arg = next((part for part in parts[1:] if part), "")
        return f"{real_name}&lt;{first_arg},…&gt;" if first_arg else f"{real_name}&lt;…&gt;"
    # `fmhaSm100fKernel_<CamelCaseFlags>`: flags are concatenated in CamelCase
    # without a separator. Keep every digit-bearing token (H512, Q8, Kv128,
    # ...) plus the last plain-word token before the common `AbForGen` tail
    # (Static/Persistent/MultiCtas, ...) -- together these are what actually
    # distinguish one specialization from another; dropping the latter would
    # collide specializations that only differ by scheduling mode into one
    # abbreviated name.
    if name.startswith("fmhaSm100fKernel_"):
        rest = name[len("fmhaSm100fKernel_"):]
        import re
        tokens = re.findall(r"[A-Z][a-z]*[0-9]*", rest)
        distinguishing = [tok for tok in tokens if any(ch.isdigit() for ch in tok)]
        scheduling = next(
            (tok for tok in ("Static", "Persistent", "MultiCtas", "Dynamic")
             if tok in tokens),
            None,
        )
        keep_parts = distinguishing[:3] + ([scheduling] if scheduling else [])
        keep = "、".join(keep_parts) if keep_parts else rest[:20]
        return f"fmhaSm100fKernel&lt;{keep},…&gt;"
    # Long name, no recognized pattern: escape as-is rather than dropping the
    # brackets silently -- still readable, just not shortened.
    return escape(name)


def kernel_table(operator_rows: list[dict[str, str]], pattern_us: float, top_n: int = 15) -> str | None:
    """Table 2.2.4: every distinct kernel, summed across all modules it appears
    in, ranked by total duration -- highest first.

    A kernel is one row here regardless of how many functional modules or unit
    positions it appears under: 所属模块 lists every module it contributed to,
    ordered by that module's own share of the kernel's time (largest first), so
    the single most relevant attribution reads first without losing the rest.
    Splitting the same kernel into one row per module would make 启动次数 and
    耗时(ms) disagree with the totals reported elsewhere in the package.
    """
    if not operator_rows:
        return None
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    per_module: dict[str, dict[str, float]] = {}
    for row in operator_rows:
        if not (row.get("序号") or "").strip().isdigit():
            continue
        name = row.get("算子名称") or ""
        module = row.get("功能模块") or ""
        duration = float(row.get("算子耗时(us)") or 0)
        totals[name] = totals.get(name, 0.0) + duration
        counts[name] = counts.get(name, 0) + 1
        per_module.setdefault(name, {})
        per_module[name][module] = per_module[name].get(module, 0.0) + duration
    if not totals:
        return None
    ranked = sorted(totals.items(), key=lambda item: -item[1])[:top_n]
    rows = []
    for name, duration in ranked:
        modules = sorted(per_module[name].items(), key=lambda item: -item[1])
        module_names = "、".join(module for module, _ in modules if module)
        rows.append([abbreviate_kernel_name(name), module_names, ms(duration),
                     pct(duration / pattern_us * 100), str(counts[name])])
    lines = [
        '<p style="margin:0">按算子合计耗时从高到低排列，Top '
        f'{top_n}；同一 kernel 跨多个模块出现时，耗时/次数为跨模块合计，'
        "所属模块列出全部（按各自贡献从高到低排序）。</p>",
        TABLE_OPEN,
        "<tr>",
        *(th(cell, HEAD_BG) for cell in ("算子名称", "所属模块", "耗时(ms)", "占pattern耗时", "启动次数")),
        "</tr>",
    ]
    for name, module_names, duration_ms, share, count in rows:
        lines.append("<tr>")
        lines.append(td(f"<code>{name}</code>"))
        lines.append(td(module_names))
        lines.append(td(duration_ms))
        lines.append(td(share))
        lines.append(td(count))
        lines.append("</tr>")
    # Two footer rows so the Top N can be read against the whole unit: what the
    # listed kernels add up to, and the unit's own wall span. Without them a
    # reader cannot tell whether Top 15 is most of the unit or a sliver of it.
    top_us = sum(duration for _, duration in ranked)
    top_count = sum(counts[name] for name, _ in ranked)
    for label, value_ms, value_pct, value_count in (
        (f"Top {top_n} 累积耗时", ms(top_us), pct(top_us / pattern_us * 100), str(top_count)),
        ("pattern总耗时", ms(pattern_us), "100%", "—"),
    ):
        lines.append("<tr>")
        lines.append(th(label, LABEL_BG))
        lines.append(td("—"))
        lines.append(td(value_ms))
        lines.append(td(value_pct))
        lines.append(td(value_count))
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def execution_order(origin_rows: list[dict[str, str]]) -> dict[str, int]:
    """Rank each origin `module` by when it first runs inside one unit position.

    The core-compute and operator tables are grouped by functional module, not
    emitted in execution order -- their row order puts an attention output
    projection before the attention core it feeds, which reads as nonsense in a
    table that claims execution order. The origin table is the only one with
    `start_ns`, so the order comes from there: one unit position (so positions
    do not interleave), sorted by start time, taking each module's first
    appearance. `module` is the join key every other table copies from origin.
    """
    positions = [row.get("unit_position") or "" for row in origin_rows]
    first = next((p for p in positions if p), "")
    within = [row for row in origin_rows if (row.get("unit_position") or "") == first]
    try:
        within.sort(key=lambda row: int(row.get("start_ns") or 0))
    except (TypeError, ValueError):
        return {}
    rank: dict[str, int] = {}
    for row in within:
        module = row.get("module") or ""
        if module and module not in rank:
            rank[module] = len(rank)
    return rank


def core_compute_table(core_rows: list[dict[str, str]],
                       origin_rows: list[dict[str, str]],
                       pattern_us: float) -> str | None:
    """Table 2.2.5: core-compute kernels in execution order, with shape/MFU/MBU.

    One row per distinct kernel, summed over its occurrences in the unit. MFU
    and MBU are the mean over those occurrences: the same kernel at four unit
    positions differs by well under a percentage point, so a mean reads cleaner
    than four values or a range, and a kernel with no shape evidence keeps an
    empty MFU rather than a fabricated one.
    """
    data = [row for row in core_rows if (row.get("序号") or "").strip().isdigit()]
    if not data:
        return None
    rank = execution_order(origin_rows)
    unranked = len(rank)
    aggregate: dict[str, dict] = {}
    for row in data:
        name = row.get("算子名称") or ""
        entry = aggregate.setdefault(name, {
            "us": 0.0, "count": 0, "shape": set(), "mfu": [], "mbu": [],
            "modules": {}, "rank": unranked,
        })
        duration = float(row.get("算子耗时(us)") or 0)
        entry["us"] += duration
        entry["count"] += 1
        entry["rank"] = min(entry["rank"], rank.get(row.get("module") or "", unranked))
        shape = (row.get("shape") or "").strip()
        if shape:
            entry["shape"].add(shape)
        for key in ("mfu", "mbu"):
            raw = (row.get(key) or "").strip().rstrip("%")
            if raw:
                try:
                    entry[key].append(float(raw))
                except ValueError:
                    pass
        module = row.get("功能模块") or ""
        entry["modules"][module] = entry["modules"].get(module, 0.0) + duration
    lines = [
        '<p style="margin:0">仅统计核心计算类算子，按执行顺序排列；'
        "MFU/MBU 为该算子各次出现的均值，缺 shape 证据时留空。</p>",
        TABLE_OPEN,
        "<tr>",
        *(th(cell, HEAD_BG) for cell in (
            "算子名称", "所属模块", "shape", "耗时(ms)", "占pattern耗时", "MFU", "MBU", "启动次数")),
        "</tr>",
    ]
    total_us = 0.0
    total_count = 0
    ordered = sorted(aggregate.items(), key=lambda item: (item[1]["rank"], -item[1]["us"]))
    for name, entry in ordered:
        total_us += entry["us"]
        total_count += entry["count"]
        modules = sorted(entry["modules"].items(), key=lambda item: -item[1])
        module_names = "、".join(module for module, _ in modules if module)
        shape = "、".join(sorted(entry["shape"])) if entry["shape"] else "—"
        shape = shape.replace("<", "&lt;").replace(">", "&gt;")
        lines.append("<tr>")
        lines.append(td(f"<code>{abbreviate_kernel_name(name)}</code>"))
        lines.append(td(module_names))
        lines.append(td(f"<code>{shape}</code>"))
        lines.append(td(ms(entry["us"])))
        lines.append(td(pct(entry["us"] / pattern_us * 100)))
        for key in ("mfu", "mbu"):
            values = entry[key]
            lines.append(td(pct(sum(values) / len(values)) if values else "—"))
        lines.append(td(str(entry["count"])))
        lines.append("</tr>")
    for label, value_ms, value_pct, value_count in (
        ("核心计算合计", ms(total_us), pct(total_us / pattern_us * 100), str(total_count)),
        ("pattern总耗时", ms(pattern_us), "100%", "—"),
    ):
        lines.append("<tr>")
        lines.append(th(label, LABEL_BG))
        lines.extend([td("—"), td("—"), td(value_ms), td(value_pct), td("—"), td("—"),
                      td(value_count)])
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def h1(text: str) -> str:
    return f'<h1 style="margin:0">{text}</h1>'


def h2(text: str) -> str:
    return f'<h2 style="margin:0">{text}</h2>'


def h3(text: str) -> str:
    return f'<h3 style="margin:0">{text}</h3>'


def p(text: str) -> str:
    return f'<p style="margin:0">{text}</p>'


def extract_prompt_inputs(prompt_path: Path) -> str:
    """工具启动指令: the task's own input list from prompt.md, not the whole
    prompt. prompt.md is mostly skill-internal instructions (which validator to
    run, how to bucket taxonomy stages, MFU formula, ...) -- none of that is a
    "launch command" a reader would want to reproduce. The one part that is is
    the `- key: value` input block right after "Analyze this task without
    asking follow-up questions:", which lists exactly what was fed into the
    tool (nsys path, model, stage, hardware, config, launch script, source
    root, ...). Falls back to a TODO note when prompt.md is absent or the
    input block cannot be found, rather than guessing at a shape.

    Not every line in that block is a necessary input: `design notes: not
    supplied` and a dispatch-site cache path are tooling/optional-input
    bookkeeping, not something a reader reproducing this run needs to see, so
    both are dropped -- an unsupplied optional input is a non-fact, and the
    dispatch cache is an internal speed-up artefact already implied by
    `model source root` being present. `nsys/sqlite` and `original report`
    are the same path in every observed task (the template always repeats it
    under both keys), so only the first is kept -- printing the identical
    path twice under two different labels would look like two different
    inputs.
    """
    SKIP_PREFIXES = ("design notes", "pre-resolved kernel dispatch sites")
    if not prompt_path.is_file():
        return "<!-- TODO 输入提示词（原始任务指令）暂时未提供 -->"
    text = prompt_path.read_text()
    marker = "Analyze this task without asking follow-up questions:"
    start = text.find(marker)
    if start == -1:
        return "<!-- TODO 输入提示词（原始任务指令）暂时未提供 -->"
    block = text[start + len(marker):]
    lines = []
    seen_values = set()
    for line in block.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith("-"):
            item = stripped[1:].strip()
            if item.lower().startswith(SKIP_PREFIXES):
                continue
            value = item.split(":", 1)[1].strip() if ":" in item else item
            if value in seen_values:
                continue  # same path already printed under an earlier key
            seen_values.add(value)
            lines.append(item)
        else:
            break
    if not lines:
        return "<!-- TODO 输入提示词（原始任务指令）暂时未提供 -->"
    import html
    return "<br>".join(html.escape(line) for line in lines)


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
    model = context.get("model_name") or prefix
    # 硬件字段只写硬件型号本身：采样 rank / device 是工具的取样细节，不是输入条件，
    # 已经在第3节"工具启动指令"里报告，这里不重复。
    hardware = manifest.get("hardware") or "—"
    parallelism = ((manifest.get("job") or {}).get("parallelism")
                   or "<!-- TODO TP/EP/PP/DCP 等 -->")
    shape_parts = [
        f"chunked-prefill-size={manifest['chunk_size']}"
        if manifest.get("chunk_size") else None,
        f"batch_size={manifest['batch_size']}" if manifest.get("batch_size") else None,
    ]
    shape = "、".join(part for part in shape_parts if part) or ""
    shape = (shape + ("、" if shape else "")
              + "<!-- TODO ctx len/mtp 等运行时 shape -->")
    nsys_name = trace.name.replace(".sqlite", ".nsys-rep") if trace.name else "—"
    head = [
        h1(f"{model} {stage} 典型shape Nsys TimeLine分析结果"),
        h1("1. 输入配置"),
        config_table([
            ("模型", model),
            ("硬件", hardware),
            ("阶段", stage),
            ("代码版本", "<!-- TODO 暂时不填 -->"),
            ("引擎配置", parallelism),
            ("运行时 shape", shape),
            ("nsys 文件", f"<code>{nsys_name}</code>"),
        ]),
    ]

    body = [h1("2. 分析结果")]

    forward = forward_tables(read_csv(csv_dir / f"{prefix}_forward_pipeline_table.csv"))
    if forward:
        body.append(h2("2.1 整体耗时统计"))
        body.append(forward[0])

    body.append(h2("2.2 Target部分耗时统计"))
    # 分析思路 belongs here, not under section 2: it states which repeating pattern
    # was selected and its wall time, and that pattern is only the denominator for
    # 2.2's tables -- 2.1 is a forward-step split that does not use it.
    body.append(p(f"<b>分析思路</b>：<!-- TODO 重复 pattern 的选取依据与 pattern 耗时 -->"))
    if forward:
        body.append(h3("2.2.1 整体耗时统计"))
        body.append(forward[1])

    body.append(h3("2.2.2 按功能模块划分统计"))
    covered = sum(
        float(row.get("模块耗时(us)") or 0) for row in stage_rows
        if (row.get("序号") or "").strip().isdigit()
    )
    # Modules can sum above the unit wall span when variants run on their own
    # streams, and calling that excess "unclassified leftovers" -- as this line
    # used to unconditionally -- is both self-contradictory and wrong.
    if covered > pattern_us:
        body.append(p(
            "以下口径为<b>一个重复 pattern</b>内、稳定样本逐算子平均耗时之和。"
            f"pattern 墙钟 {ms(pattern_us)} ms，各模块累计 {ms(covered)} ms，"
            f"因部分模块在独立 CUDA 流上与主流并行而超出墙钟 "
            f"{covered / pattern_us * 100 - 100:.1f}%，按实测原样呈现、不归一化到 100%。"
        ))
    else:
        body.append(p(
            "以下口径为<b>一个重复 pattern</b>内、稳定样本逐算子平均耗时之和，"
            f"pattern 合计 {ms(pattern_us)} ms，下表覆盖其中 {ms(covered)} ms"
            f"（{covered / pattern_us * 100:.1f}%，余量为未归类的零散算子）。"
        ))
    modules = module_table(stage_rows, operator_rows, pattern_us)
    if modules:
        body.append(modules)

    body.append(h3("2.2.3 按算子大类划分统计"))
    categories = category_table(
        read_csv(csv_dir / f"{prefix}_op_classification_table.csv"), pattern_us)
    if categories:
        body.append(categories)

    body.append(h3("2.2.4 按算子小类划分统计"))
    kernels = kernel_table(operator_rows, pattern_us)
    if kernels:
        body.append(kernels)

    core = core_compute_table(
        read_csv(csv_dir / f"{prefix}_core_compute_table.csv"),
        read_csv(csv_dir / f"{prefix}_operator_origin_table.csv"),
        pattern_us,
    )
    if core:
        body.append(h3("2.2.5 按核心计算统计"))
        body.append(core)

    # Only when the capture has a draft phase at all (speculative decoding
    # enabled) -- an ordinary run has no third child table to show, and the
    # section is skipped rather than emitted empty.
    if forward and forward[2]:
        body.append(h2("2.3 Draft部分耗时统计"))
        body.append(h3("2.3.1 整体耗时统计"))
        body.append(forward[2])

    tail: list[str] = []
    conflicts = pipeline.get("declaration_conflicts") or []
    if conflicts:
        tail.append(p("⚠ 与 config/启动命令声明不一致：" + "；".join(conflicts)))

    skill_dir = Path(__file__).resolve().parent.parent
    provenance = read_json(metadata_dir / "skill.json")
    skill_sha256 = provenance.get("sha256") or "<!-- TODO -->"
    launch_command = extract_prompt_inputs(metadata_dir / "prompt.md")
    tail.append(h1("3. 输出物料"))
    tail.append(config_table([
        ("工具版本", f"<code>sglang-nsys-static-analysis</code>，sha256 <code>{skill_sha256}</code>"),
        ("工具启动指令", launch_command),
        (
            "工具产物",
            "<code>analysis.json</code>（前端契约）、<code>final_report.md</code>（本报告）、"
            "<code>nsysscope-package.json</code>（包清单）、<code>csv/</code>（规范化表）、"
            "<code>xlsx/</code>（对应工作簿）"
            + (
                f"、<code>trace/{trace.name}</code>（导出的 SQLite trace，"
                f"原始 nsys 文件：<code>{trace}</code>）"
                if trace.name else ""
            ),
        ),
    ]))
    return "\n".join(head + body + tail)


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
