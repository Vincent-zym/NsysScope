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
from collections.abc import Iterable
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


def config_table(rows: list[tuple[str, str]], align: str = "center") -> str:
    """A vertical fact table: one row per field, label column then value column.

    Unlike `table()`, which lays entities out as columns for comparison, this is
    for facts that only ever have one value each -- forcing them into `table()`'s
    shape would need a table with one column and be unreadable. No width
    declaration, same as every other table in this report.

    `align` is centre by default because a short left-aligned cell renders
    narrower than its own header in 如流 and gets clipped. Section 3's values are
    the exception: a multi-line launch prompt and a file list read as a wall of
    text when centred, and they are long enough that the clipping problem does
    not arise.
    """
    lines = [
        '<table border="1" cellspacing="0" cellpadding="6" '
        'style="border-collapse:collapse;border:1px solid #999;'
        f'text-align:{align};margin:0">'
    ]
    for label, value in rows:
        lines.append("<tr>")
        lines.append(th(label, LABEL_BG, align=align))
        lines.append(td(value, align=align))
        lines.append("</tr>")
    lines.append("</table>")
    return "\n".join(lines)


def th(text: str, background: str, align: str = "center") -> str:
    # `background-color`, not the `background` shorthand: 如流's open API parses the
    # longhand into the cell's backgroundColor and silently drops the shorthand, so
    # a report written through publish_report_to_ku.py would lose every header tint.
    # Manual pasting accepts either, so the longhand is the one that works for both.
    cell = CELL if align == "center" else CELL.replace("text-align:center", f"text-align:{align}")
    return f'<th style="{cell};background-color:{background}">{text}</th>'


def td(text: str, align: str = "center") -> str:
    cell = CELL if align == "center" else CELL.replace("text-align:center", f"text-align:{align}")
    return f'<td style="{cell}">{text}</td>'


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


def overlap_note(numer_us: float, pattern_us: float, subject: str) -> str | None:
    """A note placed **below** a table whose numbers add up past 100%.

    Percentages use the pattern's wall-clock as the denominator, but the numerators
    are per-operator busy times summed up. When kernels run on concurrent CUDA
    streams (communication overlapping compute, dual-stream MoE), each duration is
    counted while the timeline only advances once, so the busy-time sum exceeds the
    wall-clock and the share passes 100%. It is shown as measured, not normalized,
    so the overlap stays visible -- and, per the rule that every >100% carries its
    cause right under it, this explains why.
    """
    if numer_us <= pattern_us:
        return None
    excess = numer_us / pattern_us * 100 - 100
    return p(
        f"⚠ 上表{subject}合计 {ms(numer_us)} ms，超过 pattern 墙钟 {ms(pattern_us)} ms"
        f"（{excess:.1f}%）：通信、MoE 等在独立 CUDA 流上与主流并发执行，各自耗时被分别"
        "计入而时间轴上相互重叠，故忙碌时间之和大于墙钟；按实测原样呈现、不归一化。"
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
    html = table(
        None,
        ["算子类型", "pattern总耗时", *names],
        [
            ("算子数量", [total_count, *(row.get("算子数量", "") for row in data)]),
            ("耗时(ms)", [ms(pattern_us), *(ms(row.get("总耗时(us)")) for row in data)]),
            ("耗时百分比", ["100%", *(pct(row.get("耗时占比(%)")) for row in data)]),
        ],
        bold_title=False,
    )
    # The three categories can sum past the wall-clock under multi-stream overlap;
    # when they do, the cause goes directly below this table.
    category_us = sum(float(row.get("总耗时(us)") or 0) for row in data)
    note = overlap_note(category_us, pattern_us, "各算子大类耗时")
    return html + ("\n" + note if note else "")


def abbreviate_kernel_name(name: str) -> str:
    """Escape a kernel label for the 算子名称 column.

    Shortening itself is decided per package by `kernel_labels()`, from the whole
    name set rather than from prefixes hardcoded here -- an unknown long symbol
    must shorten just as well as a known one. This function only does the last
    step every path needs: escaping `<`/`>` to `&lt;`/`&gt;`. Kernel symbols carry
    literal template angle brackets (`gatherTopK<float,uint,2,false>`) and this
    string is placed inside `<code>...</code>`, so an unescaped `<...>` would be
    parsed as an HTML tag and silently disappear.
    """
    return name.replace("<", "&lt;").replace(">", "&gt;")


def identifier_labels(names: Iterable[str], limit: int = 60) -> dict[str, str]:
    """Shorten over-long symbols whose parameters are baked into the identifier.

    `bmm_MxE4m3_MxE2m1MxE4m3_Fp32_Ab32_..._sm100f` (203 chars),
    `kernel_cutlass_kernel_TgvGemmCuteExtKernel_cta64x16x128_...`,
    `fmhaSm103aKernel_QkvBfloat16OBfloat16HQk192...` and
    `triton_poi_fused__to_copy_arange_..._view_4` all have the same shape: a
    family name followed by a long run of `_`-separated (or CamelCase-run) flags.
    There is no list of families here on purpose -- the previous version special
    cased two prefixes by name, so the next new long symbol was simply not
    shortened at all.

    The rule is "show as much as fits, and never merge two rows": candidate
    labels are built by keeping a run of leading tokens, or only the tokens that
    this member does not share with its siblings, each token clipped to a
    character budget, with `…` marking what was dropped and the last token always
    kept (it is usually the variant index or arch tag). Among the candidates that
    give every name in the family its own label, the most detailed one that still
    fits `limit` wins; if none fits, the shortest unique one does. When nothing
    separates them, the full names are kept.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(name.split("_")[0], []).append(name)
    labels: dict[str, str] = {}
    for members in groups.values():
        members = sorted(set(members))
        if not any(len(name) > limit for name in members):
            labels.update({name: name for name in members})
            continue
        tokens = {name: name.split("_") for name in members}
        widest = max(len(parts) for parts in tokens.values())
        unique: list[dict[str, str]] = []
        for budget in (24, 40, 64, 1024):
            for keep in range(2, widest + 1):
                unique.append(_assemble(tokens, budget, keep=keep))
            for stem in (1, 2, 3):
                for count in range(1, 4):
                    unique.append(_assemble(tokens, budget, stem=stem, count=count))
        candidates = [
            candidate for candidate in unique
            if len(set(candidate.values())) == len(candidate)
        ]
        if not candidates:
            labels.update({name: name for name in members})
            continue
        fitting = [c for c in candidates if max(len(v) for v in c.values()) <= limit]
        if fitting:
            # Most detail that still fits: a label is only useful if it is
            # recognizable, so spend the whole budget rather than the minimum.
            labels.update(max(fitting, key=lambda c: max(len(v) for v in c.values())))
        else:
            labels.update(min(candidates, key=lambda c: max(len(v) for v in c.values())))
    return labels


def _assemble(tokens: dict[str, list[str]], budget: int, keep: int | None = None,
              stem: int = 1, count: int = 0) -> dict[str, str]:
    """One candidate label per name.

    Two shapes: a run of `keep` leading tokens, or a `stem`-token prefix plus the
    first `count` tokens that no sibling has -- the latter is what reaches the
    distinguishing part of a symbol whose siblings share a long common middle.
    """
    labels = {}
    for name, parts in tokens.items():
        if keep is not None:
            head = parts[:keep]
        else:
            others = [other for other in tokens.values() if other is not parts]
            head = parts[:stem] + [
                part for part in parts[stem:-1]
                if all(part not in other for other in others)
            ][:count]
        pieces = [_clip(part, budget) for part in head]
        body = parts[len(head):] if keep is not None else [
            part for part in parts[1:] if part not in head
        ]
        if body:
            tail = parts[-1]
            if len(body) > 1 or body[0] != tail:
                pieces.append("…")
            if tail not in head:
                pieces.append(_clip(tail, budget))
        labels[name] = "_".join(pieces)
    return labels


def _clip(token: str, budget: int) -> str:
    """One token, cut to `budget` characters at a CamelCase boundary when possible."""
    if len(token) <= budget:
        return token
    import re
    words = re.findall(r"[A-Z]?[a-z]+\d*|[A-Z]+\d*|\d+", token)
    kept: list[str] = []
    for word in words:
        if len("".join([*kept, word])) > budget:
            break
        kept.append(word)
    head = "".join(kept) if kept else token[:budget]
    return f"{head}…"


def split_template_args(name: str) -> tuple[str, list[str] | None]:
    """`base<a, b<c,d>, e>` → `("base", ["a", "b<c,d>", "e"])`.

    Splits on top-level commas only, so a nested template argument stays whole.
    Returns `None` for the argument list when the symbol carries no `<...>` at
    all (`nvjet_tst_128x256_...`, `triton_poi_fused_...`): those have nothing to
    drop -- the long middle is the kernel's identity, not a parameter.
    """
    start = name.find("<")
    if start < 0 or not name.endswith(">"):
        return name, None
    base, inner = name[:start], name[start + 1:-1]
    args: list[str] = []
    depth = 0
    current = ""
    for char in inner:
        if char in "<(":
            depth += 1
        elif char in ">)":
            depth -= 1
        if char == "," and depth == 0:
            args.append(current.strip())
            current = ""
        else:
            current += char
    args.append(current.strip())
    return base, args


def kernel_labels(names: Iterable[str], limit: int = 60) -> dict[str, str]:
    """Per-package display labels for kernel symbols: keep only what distinguishes.

    A template argument that is identical across every specialization of the same
    base name in this package carries no information for the reader -- it is the
    same for all rows -- so it collapses into `…`. Only the positions that
    actually differ are kept. `per_token_group_quant_8bit_kernel<__nv_bfloat16,
    __nv_fp8_e4m3, (bool)1, (bool)1, unsigned int>` is the only specialization of
    its base here, so nothing needs distinguishing and it becomes
    `per_token_group_quant_8bit_kernel<…>`; two GEMM specializations that differ
    in one argument keep exactly that argument.

    This is decided from the package's own name set rather than per name, which is
    what makes it consistent: previously only two hardcoded prefixes were
    shortened and every other long symbol passed through, so whether a row was
    abbreviated looked arbitrary.

    A group is only touched when at least one of its names exceeds `limit`: a
    short symbol is already readable, and shortening it would only cost the reader
    information. When collapsing would make two names share one label, the
    dropped positions are added back until they differ again -- a label must never
    merge two different kernels.
    """
    groups: dict[str, list[str]] = {}
    for name in names:
        groups.setdefault(split_template_args(name)[0], []).append(name)
    labels: dict[str, str] = {}
    for base, members in groups.items():
        members = sorted(set(members))
        parsed = {name: split_template_args(name)[1] for name in members}
        if not any(len(name) > limit for name in members):
            labels.update({name: name for name in members})
            continue
        # `base<…>` in the source table is already opaque: it cannot take part in
        # deciding which positions differ, or it would make every position differ.
        concrete = {name: args for name, args in parsed.items()
                    if args is not None and args != ["…"]}
        if not concrete:
            labels.update({name: name for name in members})
            continue
        width = max(len(args) for args in concrete.values())
        differing = [
            index for index in range(width)
            if len({tuple(args[index:index + 1]) for args in concrete.values()}) > 1
        ]
        for extra in range(width + 1):
            keep = sorted(set(differing) | set(range(extra)))
            candidate = {}
            for name, args in parsed.items():
                if args is None:
                    candidate[name] = name
                elif args == ["…"]:
                    candidate[name] = f"{base}<…>"
                else:
                    kept = [args[index] for index in keep if index < len(args)]
                    if len(kept) == len(args):
                        # Nothing was dropped: keep the symbol exactly as the
                        # table spelled it instead of re-joining it and churning
                        # whitespace for no gain.
                        candidate[name] = name
                        continue
                    tail = "…" if len(kept) < len(args) else ""
                    inner = ", ".join([*kept, tail] if tail else kept)
                    candidate[name] = f"{base}<{inner}>"
            if len(set(candidate.values())) == len(candidate):
                labels.update(candidate)
                break
        else:
            # Positions ran out and two names still share a label (the source
            # table wrote the same base both expanded and collapsed): keep the
            # full names rather than merging two kernels into one row label.
            labels.update({name: name for name in members})
    # Second pass for what the template rule cannot reach: symbols whose flags
    # live in the identifier itself, with no `<...>` to thin out.
    plain = [name for name, label in labels.items()
             if len(label) > limit and "<" not in label]
    compressed = identifier_labels(plain, limit)
    for name in plain:
        labels[name] = compressed.get(labels[name], compressed.get(name, labels[name]))
    return labels


def kernel_table(operator_rows: list[dict[str, str]], pattern_us: float,
                 labels: dict[str, str] | None = None, top_n: int = 15) -> str | None:
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
        rows.append([abbreviate_kernel_name((labels or {}).get(name, name)),
                     module_names, ms(duration),
                     pct(duration / pattern_us * 100), str(counts[name])])
    lines = [
        '<p style="margin:0">按算子合计耗时从高到低排列，Top '
        f'{top_n}；同一 kernel 跨多个模块出现时，耗时/次数为跨模块合计，'
        "所属模块列出全部（按各自贡献从高到低排序）。</p>",
        TABLE_OPEN,
        "<tr>",
        *(th(cell, HEAD_BG) for cell in (
            "序号", "算子名称", "所属模块", "耗时(ms)", "占pattern耗时", "启动次数")),
        "</tr>",
    ]
    for order, (name, module_names, duration_ms, share, count) in enumerate(rows, start=1):
        lines.append("<tr>")
        lines.append(td(str(order)))
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
        lines.append(td("—"))
        lines.append(td(value_ms))
        lines.append(td(value_pct))
        lines.append(td(value_count))
        lines.append("</tr>")
    lines.append("</table>")
    # When the Top N busy times already sum past the wall-clock, say why right here.
    note = overlap_note(top_us, pattern_us, f"Top {top_n} 累积耗时")
    if note:
        lines.append(note)
    return "\n".join(lines)


def kernel_base(name: str) -> str:
    """The kernel's own name, without namespace, template arguments or signature.

    The core-compute table already shortens a symbol to
    `sm100_fp8_fp4_gemm_1d1d_impl<…>` while the origin table keeps the full
    `void deep_gemm::sm100_fp8_fp4_gemm_1d1d_impl<(cute::UMMA::Major)0, ...>`,
    so the two only join on the base name.
    """
    head = name.split("<")[0].replace("void ", "").strip()
    return head.split("::")[-1].split("(")[0].strip()


def occurrence_starts(origin_rows: list[dict[str, str]]) -> dict[tuple[str, str, str], list[int]]:
    """`start_ns` queues per (unit position, module, kernel), earliest first.

    The origin table is the only one carrying `start_ns`, and it is what turns
    the core-compute table -- which is grouped by functional module, so an
    attention output projection sits before the attention core that feeds it --
    back into the order the GPU actually ran. A queue rather than a single value
    because one kernel can run several times inside the same module.
    """
    queues: dict[tuple[str, str, str], list[int]] = {}
    for row in sorted(origin_rows, key=lambda item: int(item.get("start_ns") or 0)):
        key = (
            row.get("unit_position") or "",
            row.get("module") or "",
            kernel_base(row.get("operator_name") or ""),
        )
        queues.setdefault(key, []).append(int(row.get("start_ns") or 0))
    return queues


def core_compute_table(core_rows: list[dict[str, str]],
                       origin_rows: list[dict[str, str]],
                       pattern_us: float,
                       labels: dict[str, str] | None = None) -> str | None:
    """Table 2.2.5: every core-compute operator of one pattern, in execution order.

    One row per operator, where an operator is a (dispatch site, kernel, shape)
    triple -- not per kernel name. The name alone is not an identity: a package
    whose core table shortens every deep-gemm specialization to
    `sm100_fp8_fp4_gemm_1d1d_impl<…>` has seven different GEMMs sharing one name,
    and keying on it merged q_b_proj, o_proj, gate_up_proj and the indexer
    projections into a single meaningless row spanning four modules. Keying on the
    triple keeps them as the seven operators they are.

    Rows are aggregated over the pattern's repeated unit positions (启动次数 says
    how many), because the pattern's four layer positions run the same operator
    with the same shape -- four identical rows would be noise, not a timeline.
    MFU/MBU are the mean over those occurrences; they differ by well under a
    percentage point across positions.
    """
    data = [row for row in core_rows if (row.get("序号") or "").strip().isdigit()]
    if not data:
        return None
    queues = occurrence_starts(origin_rows)
    aggregate: dict[tuple[str, str, str], dict] = {}
    for index, row in enumerate(data):
        shape = (row.get("shape") or "").strip()
        key = (row.get("module") or "", row.get("算子名称") or "", shape)
        entry = aggregate.setdefault(key, {
            "us": 0.0, "count": 0, "mfu": [], "mbu": [], "order": (True, 0, index),
            "module": row.get("功能模块") or "", "name": row.get("算子名称") or "",
            "shape": shape,
        })
        entry["us"] += float(row.get("算子耗时(us)") or 0)
        entry["count"] += 1
        queue = queues.get((
            row.get("单元位置") or "",
            row.get("module") or "",
            kernel_base(row.get("算子名称") or ""),
        ))
        start = queue.pop(0) if queue else None
        if start is not None:
            entry["order"] = min(entry["order"], (False, start, index))
        for column in ("mfu", "mbu"):
            raw = (row.get(column) or "").strip().rstrip("%")
            if raw:
                try:
                    entry[column].append(float(raw))
                except ValueError:
                    pass
    lines = [
        '<p style="margin:0">仅统计核心计算类算子，按执行顺序排列；'
        "MFU/MBU 为该算子各次出现的均值，缺 shape 证据时留空。</p>",
        TABLE_OPEN,
        "<tr>",
        *(th(cell, HEAD_BG) for cell in (
            "序号", "算子名称", "所属模块", "shape", "耗时(ms)", "占pattern耗时",
            "MFU", "MBU", "启动次数")),
        "</tr>",
    ]
    total_us = 0.0
    total_count = 0
    ordered = sorted(aggregate.values(), key=lambda item: item["order"])
    for order, entry in enumerate(ordered, start=1):
        total_us += entry["us"]
        total_count += entry["count"]
        shape = (entry["shape"] or "—").replace("<", "&lt;").replace(">", "&gt;")
        lines.append("<tr>")
        lines.append(td(str(order)))
        label = (labels or {}).get(entry["name"], entry["name"])
        lines.append(td(f"<code>{abbreviate_kernel_name(label)}</code>"))
        lines.append(td(entry["module"] or "—"))
        lines.append(td(f"<code>{shape}</code>"))
        lines.append(td(ms(entry["us"])))
        lines.append(td(pct(entry["us"] / pattern_us * 100)))
        for column in ("mfu", "mbu"):
            values = entry[column]
            lines.append(td(pct(sum(values) / len(values)) if values else "—"))
        lines.append(td(str(entry["count"])))
        lines.append("</tr>")
    for label, value_ms, value_pct, value_count in (
        ("核心计算合计", ms(total_us), pct(total_us / pattern_us * 100), str(total_count)),
        ("pattern总耗时", ms(pattern_us), "100%", "—"),
    ):
        lines.append("<tr>")
        lines.append(th(label, LABEL_BG))
        lines.extend([td("—"), td("—"), td("—"), td(value_ms), td(value_pct), td("—"),
                      td("—"), td(value_count)])
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
    # 原始输入文件：任务给的可能就是 .sqlite 导出，这时把名字改写成 .nsys-rep 会写出一个
    # 不存在的文件名，所以优先用 report_path 本身的名字，它是什么后缀就报什么后缀。
    original = Path(context.get("report_path") or context.get("report") or "")
    stage = manifest.get("stage") or context.get("stage") or "—"
    model = context.get("model_name") or prefix
    # 硬件字段只写硬件型号本身：采样 rank / device 是工具的取样细节，不是输入条件，
    # 已经在第3节"工具启动指令"里报告，这里不重复。
    hardware = manifest.get("hardware") or "—"
    parallelism = ((manifest.get("job") or {}).get("parallelism")
                   or "<!-- TODO TP/EP/PP/CP 等，只写短名（TP=8、megamoe、EAGLE 投机解码），"
                      "不要贴 --flag 原文、不要解释 -->")
    shape_parts = [
        f"chunked-prefill-size={manifest['chunk_size']}"
        if manifest.get("chunk_size") else None,
        f"batch_size={manifest['batch_size']}" if manifest.get("batch_size") else None,
    ]
    shape = "、".join(part for part in shape_parts if part) or ""
    shape = (shape + ("、" if shape else "")
              + "<!-- TODO ctx len/mtp 等运行时 shape，只列值，缺的项直接不写、"
                "不要解释为什么缺 -->")
    nsys_name = original.name or trace.name or "—"
    # 报告标题：模型-阶段-规模。decode 用并发 batch、prefill 用 chunk（输入长度）；
    # 规模取不到时省略，不写占位，标题降级为 模型-阶段。
    if str(stage).lower() == "decode" and manifest.get("batch_size"):
        scale = f"-bs{manifest['batch_size']}"
    elif str(stage).lower() == "prefill" and manifest.get("chunk_size"):
        scale = f"-inputlen{manifest['chunk_size']}"
    else:
        scale = ""
    head = [
        h1(f"{model}-{stage}{scale}"),
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
    if forward[0]:
        body.append(h2("2.1 整体耗时统计"))
        body.append(forward[0])

    body.append(h2("2.2 Target部分耗时统计"))
    # 分析思路 belongs here, not under section 2: it states which repeating pattern
    # was selected and its wall time, and that pattern is only the denominator for
    # 2.2's tables -- 2.1 is a forward-step split that does not use it.
    # The hint spells out the length limit and what not to restate, because the
    # failure mode here is not an empty marker but a 200-character paragraph that
    # repeats 2.2.1's 单层耗时 row and adds min/max sampling stats no table asks for.
    body.append(p(
        "<b>分析思路</b>：<!-- TODO 一句话写出重复 pattern 的选取依据与 pattern 耗时，"
        "100 字以内，参照 references/final_report.example.md 的同一句；"
        "不要写采样 min/max、不要重复 2.2.1 已有的单层耗时、不要下结论 -->"
    ))
    if forward[1]:
        body.append(h3("2.2.1 整体耗时统计"))
        body.append(forward[1])

    body.append(h3("2.2.2 按功能模块划分统计"))
    covered = sum(
        float(row.get("模块耗时(us)") or 0) for row in stage_rows
        if (row.get("序号") or "").strip().isdigit()
    )
    classification_rows = read_csv(csv_dir / f"{prefix}_op_classification_table.csv")
    classified_us = next(
        (float(row.get("总耗时(us)") or 0) for row in classification_rows
         if (row.get("算子类型") or "").strip() == "总计"), 0.0)
    # Modules can sum above the unit wall span when variants run on their own
    # streams, and calling that excess "unclassified leftovers" -- as this line
    # used to unconditionally -- is both self-contradictory and wrong.
    if covered > pattern_us:
        # Above the table: only the 口径 and the two numbers. The reason the numbers
        # exceed the wall-clock goes below the table (overlap_note), so every >100%
        # in the report carries its cause directly under the data.
        body.append(p(
            "以下口径为<b>一个重复 pattern</b>内、稳定样本逐算子平均耗时之和，"
            f"pattern 墙钟 {ms(pattern_us)} ms，各模块累计 {ms(covered)} ms。"
        ))
    else:
        # 余量的成因不能一概而论：模块之和已经等于全部算子耗时之和时没有未归类算子，
        # 差额只能是 pattern 内 kernel 之间的 GPU 空隙；只有模块之和还不到算子总和，
        # 余量里才真的有未归类的零散算子。写错了方向就是把一个没有证据的成因塞进报告。
        if classified_us and abs(classified_us - covered) <= 1.0:
            remainder = "余量为 kernel 之间的 GPU 空隙"
        else:
            remainder = "余量为未归类的零散算子"
        body.append(p(
            "以下口径为<b>一个重复 pattern</b>内、稳定样本逐算子平均耗时之和，"
            f"pattern 合计 {ms(pattern_us)} ms，下表覆盖其中 {ms(covered)} ms"
            f"（{covered / pattern_us * 100:.1f}%，{remainder}）。"
        ))
    modules = module_table(stage_rows, operator_rows, pattern_us)
    if modules:
        body.append(modules)
        module_overlap = overlap_note(covered, pattern_us, "各功能模块耗时")
        if module_overlap:
            body.append(module_overlap)

    body.append(h3("2.2.3 按算子大类划分统计"))
    categories = category_table(classification_rows, pattern_us)
    if categories:
        body.append(categories)

    body.append(h3("2.2.4 按算子小类划分统计"))
    core_rows = read_csv(csv_dir / f"{prefix}_core_compute_table.csv")
    # One label map for the whole report, computed from every kernel name the
    # package mentions: which template arguments are worth showing depends on
    # what else is in the table, so 2.2.4 and 2.2.5 must decide it together or
    # the same kernel would appear under two different labels.
    labels = kernel_labels(
        [row.get("算子名称") or "" for row in (*operator_rows, *core_rows)
         if (row.get("序号") or "").strip().isdigit()]
    )
    kernels = kernel_table(operator_rows, pattern_us, labels)
    if kernels:
        body.append(kernels)

    core = core_compute_table(
        core_rows,
        read_csv(csv_dir / f"{prefix}_operator_origin_table.csv"),
        pattern_us,
        labels,
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
    # Left-aligned: a multi-line launch prompt and a file list are unreadable
    # centred, and these values are long enough that the clipping that rules out
    # left-alignment elsewhere does not apply.
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
                f"原始 nsys 文件：<code>{original or trace}</code>）"
                if trace.name else ""
            ),
        ),
    ], align="left"))
    return "\n".join(head + body + tail)


PROSE_LIMIT = 75    # 一句正文的长度上限（分析思路、表说明、冲突说明）
VALUE_LIMIT = 35    # 第1节一个配置字段的上限：值的枚举比正文短，不该变成段落


def strip_tags(line: str) -> str:
    """Visible text of one HTML line, for measuring how long the prose actually is."""
    import html
    import re
    return html.unescape(re.sub(r"<[^>]+>", "", line)).strip()


def text_length(text: str) -> int:
    """长度按"中文字数 + 每段连续英文/数字算 1 个词"计。

    纯字符数会把 `flashinfer_mxfp4`、`41.51 ms` 这类不可压缩的标识符算成十几个
    字，于是一句合规的短句（example.md 的分析思路，146 个字符）反而比一段冗长的
    中文更"长"。按这个口径，那句校准句是 28，被用户驳回的 240 字符版本是 90。
    """
    import re
    cjk = r"\u3000-\u9fff\uff00-\uffef"
    return (len(re.findall(f"[{cjk}]", text))
            + len(re.findall(f"[^{cjk}]+", text)))


def check_report(package: Path, prefix: str, report_path: Path) -> list[str]:
    """Compare a finished report against the skeleton this file would generate.

    The report is written by the generator and then has its `<!-- TODO -->` slots
    filled in by hand, which historically let two kinds of drift through unseen:
    a hand-added row or a rewritten generated note (structure drift), and a
    filled slot that grew into a 240-character paragraph (length drift). Both are
    mechanical to catch -- regenerate the skeleton, require every non-TODO line
    to survive byte-identical, and measure the text of the lines that were filled.
    """
    problems: list[str] = []
    skeleton = build(package, prefix).splitlines()
    report = report_path.read_text().splitlines()
    measured: set[int] = set()
    for index, (want, got) in enumerate(zip(skeleton, report), start=1):
        if "<!-- TODO" not in want:
            if want != got:
                problems.append(
                    f"第 {index} 行与生成骨架不一致（生成的内容不应手改，"
                    f"要改就改 build_final_report.py）：\n  骨架：{want}\n  报告：{got}"
                )
            continue
        head_part, _, rest = want.partition("<!-- TODO")
        tail_part = rest.partition("-->")[2]
        if not (got.startswith(head_part) and got.endswith(tail_part)):
            problems.append(
                f"第 {index} 行的 TODO 槽位以外被改动了：\n  骨架：{want}\n  报告：{got}"
            )
            continue
        if "<!-- TODO" in got:
            continue  # 未填的 marker 是诚实的"尚未分析"，不算错
        measured.add(index)
        filled = strip_tags(got[len(head_part):len(got) - len(tail_part) or None])
        limit = PROSE_LIMIT if got.lstrip().startswith("<p") else VALUE_LIMIT
        if text_length(filled) > limit:
            problems.append(
                f"第 {index} 行填写内容长度 {text_length(filled)}，超过上限 {limit}："
                f"只说结论与数字，采样口径、推导过程、表里已有的数字都不要写\n  {filled}"
            )
    if len(report) > len(skeleton):
        extra = "\n  ".join(report[len(skeleton):][:5])
        problems.append(
            f"报告比生成骨架多 {len(report) - len(skeleton)} 行"
            f"（第1/3节的字段与表格由生成器决定，不要自行增删）：\n  {extra}"
        )
    elif len(report) < len(skeleton):
        problems.append(f"报告比生成骨架少 {len(skeleton) - len(report)} 行")
    for index, line in enumerate(report, start=1):
        if index in measured or "<!-- TODO" in line:
            continue
        if not line.lstrip().startswith("<p"):
            continue
        # The length rule polices hand-written prose (分析思路, filled markers). A
        # line that is byte-identical to the skeleton is generator-authored -- an
        # overlap note, a ranking caption -- and its length is the generator's call,
        # not per-report drift. Only flag prose that deviates from the skeleton.
        if index - 1 < len(skeleton) and line == skeleton[index - 1]:
            continue
        text = strip_tags(line)
        if text_length(text) > PROSE_LIMIT:
            problems.append(
                f"第 {index} 行正文长度 {text_length(text)}，超过上限 {PROSE_LIMIT}：\n  {text}"
            )
    return problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", help="result package directory (holds csv/ and metadata/)")
    parser.add_argument("--prefix", default="analysis")
    parser.add_argument("--output", default=None, help="defaults to <package>/final_report.md")
    parser.add_argument(
        "--check", action="store_true",
        help="不写文件，只检查已有报告：非 TODO 行必须与骨架一致，填写内容不得超长",
    )
    args = parser.parse_args()

    package = Path(args.package).resolve()
    output = Path(args.output) if args.output else package / "final_report.md"
    if args.check:
        if not output.is_file():
            raise SystemExit(f"[final-report] 找不到报告：{output}")
        problems = check_report(package, args.prefix, output)
        if problems:
            print(f"[final-report] {output} 有 {len(problems)} 处问题：")
            for problem in problems:
                print(f"  - {problem}")
            raise SystemExit(1)
        print(f"[final-report] {output} 结构与骨架一致，正文长度合规")
        return
    output.write_text(build(package, args.prefix))
    print(f"[final-report] wrote {output}")


if __name__ == "__main__":
    main()
