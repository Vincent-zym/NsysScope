# final_report.md format

`final_report.md` is the package's human-facing deliverable. It is a pure
timeline/operator-timing report -- no 结论 section, no 潜在优化点, no prose
interpretation anywhere. Every line is a fact with a number behind it, or a
`<!-- TODO -->` for a fact the manifest doesn't carry structurally. It is pasted
into 如流知识库 (the team wiki), which is why the whole document is inline-styled
HTML rather than markdown. `scripts/build_final_report.py` emits the whole
skeleton fully filled from the package's own data; there is normally nothing
left for the agent to write by hand except the handful of `<!-- TODO -->`
markers the manifest genuinely does not carry (代码版本, 引擎配置/运行时 shape
when the manifest lacks them, 分析思路's selection rationale, and the
declaration-conflict note when one exists).

`final_report.example.md` in this directory is a complete filled-in report (a
GLM5.2 prefill analysis). Read it as the reference for the exact skeleton
shape and for how terse 分析思路 is. When in doubt, match it rather than
inventing a new shape.

## Section layout

```text
<h1>{model} {stage} 典型shape Nsys TimeLine分析结果</h1>
<h1>1. 输入配置</h1>              vertical fact table: 模型/硬件/阶段/代码版本/
                                  引擎配置/运行时 shape/nsys 文件
<h1>2. 分析结果</h1>
  <h2>2.1 整体耗时统计</h2>        forward step 拆分（Forward step / Target /
                                     Draft / Token间间隙）
  <h2>2.2 Target部分耗时统计</h2>  分析思路（一句话选取依据，仅此一句，无结论）
    <h3>2.2.1 整体耗时统计</h3>    Target 内部构成，首列为 Target 主模型总耗时
    <h3>2.2.2 按功能模块划分统计</h3>  首列 pattern总耗时，其余按执行顺序
    <h3>2.2.3 按算子大类划分统计</h3>  首列 pattern总耗时，其余为核心计算/通信/
                                       小算子（辅助算子）
    <h3>2.2.4 按算子小类划分统计</h3>  按算子（kernel）合计耗时排序 Top 15，
                                       含所属模块、占pattern耗时、启动次数，
                                       表尾 Top N 累积耗时 + pattern总耗时 两行
    <h3>2.2.5 按核心计算统计</h3>  核心计算算子按执行顺序，含 shape/MFU/MBU，
                                     表尾 核心计算合计 + pattern总耗时 两行
  <h2>2.3 Draft部分耗时统计</h2>   only when the capture has a draft phase
                                     (speculative decoding enabled) -- omitted
                                     entirely otherwise, not emitted empty
    <h3>2.3.1 整体耗时统计</h3>    draft 内部构成, same shape as 2.2.1,
                                     首列为 Draft 模型总耗时
<h1>3. 输出物料</h1>              vertical fact table: 工具版本/工具启动指令/工具产物
```

分析思路 sits under 2.2, not under section 2: it states which repeating unit was
selected and its wall time, and that unit is only the denominator for 2.2's
tables -- 2.1 is a forward-step split that does not use it. There is no
separate popo/算子分析工具数据 section; a declaration-conflict note, when one
exists, goes right before 输出物料.

Every heading in this report is HTML (`<h1|h2|h3 style="margin:0">`), and every
paragraph is `<p style="margin:0">` -- there is no markdown anywhere in the
document, not even the outer title or sections 3/4, and no blank line anywhere
in the source between one block and the next (see paste-fidelity rules below
for why). Section 1 and section 3 are each a single vertical fact table with no
prose around them -- no `结果包`/`模型结构` list items, no narration before or
after the table.

## What the reader wants

- Every sentence of prose in this report -- 分析思路, any filled `<!-- TODO -->`,
  the declaration-conflict note -- must be plain and short: state the fact and
  stop. Target roughly 100 Chinese characters or less per sentence; the 分析思路
  line in final_report.example.md ("GLM5.2 稀疏 MoE 区按 non-shared(full)
  Indexer : shared Indexer = 1 : 3 交替，以 4 层为一个分析pattern（1 × Full-Indexer
  + 3 × Shared-Indexer），pattern耗时 41.51 ms。") is the calibration point -- match that
  register, not a formal-report register. Cut qualifiers, cut restated context
  the reader already saw in a table, and prefer a plain clause over a nested
  one. A sentence that needs a semicolon to hold two separate facts is doing
  the reader's parsing for them -- write two short sentences instead unless the
  facts are genuinely inseparable (as 分析思路's own example is: the unit
  definition and the forward-link caveat are two different scopes of
  statistics, not two clauses of one fact).
- 引擎配置 and 运行时 shape apply the same brevity rule: write the parallelism
  scheme and the handful of flags that actually shape the trace (TP/DP/EP/PP/CP,
  the MoE/attention backend when it changed the kernel selection, whether
  speculative decoding is on), each compressed to its shortest recognizable
  form (`TP=8`, `megamoe`, `EAGLE 投机解码`), not the raw launch-command flags
  pasted verbatim. `--tp-size 8 --dp-size 1 --moe-a2a-backend megamoe
  --attention-backend nsa --nsa-prefill-cp-mode round-robin-split
  --speculative-algorithm EAGLE` is the wrong register; `TP=8、DP=1、CP=8、
  megamoe、EAGLE 投机解码` is the right one -- keep the backend name itself
  (it is what a reader compares runs by) but drop the `--flag-name` and any
  value that does not change between runs of this same package. When the
  manifest's `job.parallelism` field already carries a short form (e.g.
  `tp_size=8, ep_size=8, pp_size=1`), use it as-is; when the agent has to
  derive it from a launch script instead, compress it to this same short form
  rather than quoting the script's flags. If a runtime setting disagrees with
  what the launch script declared (e.g. `--enable-dp-attention` was passed but
  the captured server_args show it is actually `false`), note that fact once,
  briefly, next to the flag it contradicts (`DP=1（未启用 dp-attention）`) --
  do not restate the mismatch a second time in 运行时 shape or defer it
  entirely to the section 3 conflict note when it directly determines an
  引擎配置 field's value.
- 分析思路 states the selection rationale (why this repeating unit, what its wall
  time is) in one sentence -- it is not a place for conclusions or judgement,
  and it is the only prose sentence in the entire report that is not a table
  caption. Everything else is a table, a table's caption, or a `<!-- TODO -->`.
- Do not explain the methodology, the verification steps or the closure
  invariants. That belongs in references/output-spec.md, not in a report a
  performance engineer reads to decide what to optimise.
- Do not attribute a gap or a low MFU to a cause the tables do not evidence,
  and do not draw a conclusion at all -- report the number and let the reader
  draw their own. This report has no 结论/潜在优化点 section by design.
- Say each thing once. A number that appears in one table does not need
  restating in another table's caption or a follow-up paragraph -- pick the one
  place a reader looking for that number would check first.
- If a `<!-- TODO -->` marker has no number-backed content to add, leave it as
  the marker rather than writing a placeholder sentence -- an unfilled marker
  is honest about "not analysed yet"; a padded sentence is not.

## Paste-fidelity rules (learned the hard way)

如流's editor keeps some of the pasted HTML and silently drops the rest. What
survives, and what does not, is not obvious -- these were established by trial:

- **Kept**: per-cell `style` (border, `background-color`, `text-align`,
  `vertical-align`), `<b>`, `<code>`, `<h1>`/`<h2>`/`<h3>`, `border`/`cellpadding`
  attributes. Use the `background-color` longhand rather than the `background`
  shorthand: pasting accepts either, but the open API only reads the longhand
  (see "Writing through the open API" below).
- **Dropped**: `<caption>` entirely -- put the table title in a
  `<p style="margin:0">` line immediately above the table instead (2.2.4's
  table has no title line since the caption there is the note explaining the
  ranking rule, not a bare title).
- **Dropped**: every width declaration -- `style="width"`, `min-width`,
  `padding`, `<table width>`, `<colgroup><col width>`, `<th width>`. Column
  widths are recomputed from the content, so do not try to set them, and never
  set `text-align:left` on a table's own cells for this reason either -- a
  left-aligned column with short content renders narrower than the header text
  next to it and gets clipped in 如流. Every table in this report, including
  the vertical fact tables in sections 1 and 3, is `text-align:center`. Widen a
  column by making its header wording longer (`Target 主模型` instead of
  `Target`); leading/trailing full-width spaces get trimmed and do not work.
- A markdown `#` heading carries its own default bottom margin that this
  file's inline styles cannot override, and a blank line in the markdown
  source (or any rendered `margin-bottom` on the block above a table) both come
  through as an extra blank line in 如流. This is why the whole document is
  HTML, not just the tables: mixing markdown headings with HTML tables left a
  gap before the markdown heading that no `margin:0` on the HTML side could
  close. Hence `margin:0` on every block and zero blank lines anywhere in the
  markdown source, not just around tables.

## Writing through the open API

`scripts/publish_report_to_ku.py` puts the same HTML on a 如流 page through the
open API instead of by pasting. What survives there was established by probing a
live document, and it is more than pasting keeps:

- **Kept**: `border` → the cell's `borderColor`/`borderIndex`,
  `vertical-align` → `verticalAlign`, `text-align` → `textAlign` on the cell's
  inner paragraph, `<b>` → a bold text run, `<code>` → an `inline-code` node,
  `&lt;`/`&gt;` → literal text, `colspan`/`rowspan`. Column widths are computed
  and stored, so the API path actually keeps widths that pasting drops.
- **Dropped**: the `background` shorthand and a `bgcolor` attribute. Only the
  `background-color` longhand reaches the cell's `backgroundColor`, which is why
  `build_final_report.py` emits the longhand -- pasting accepts either form, so
  the longhand is the one that works through both paths.
- An edit lands in the document's *edit* state. The API answers 200 while
  readers still see the old page until `publish-doc` runs, so publishing is part
  of writing rather than an optional follow-up, and the script always reads the
  page back to compare table/cell/tint counts against the source.

## List conventions

Anything that would otherwise become a long comma-run goes in
`<ul style="margin:0;padding-left:22px">`, one item per line, with a
`<p style="margin:0"><b>title</b></p>` above it, if the report ever needs a
list at all -- the current skeleton does not: what used to be a "小算子 Top 5"
list is now the 2.2.4 table, which is strictly more informative (module
attribution, percentage, launch count) at the same information density. Keep
this convention in mind only if a future addition needs a list; do not
reintroduce a list where a table already covers the same ground.

## Table conventions

- Metrics are rows, entities are columns, so a table stays readable when it has
  eleven functional modules -- except the vertical fact tables in sections 1
  and 3, and 2.2.4/2.2.5's per-kernel tables, where each row is one entity (one
  config field, one kernel) because there is exactly one metric per row, not
  several.
- Header row background `#b4c7e7`, first column `#d9e2f3`, every cell centred
  (`text-align:center`, see paste-fidelity rules above for why left-align is
  never used even in a fact table).
- No table carries a title line: the header row already names what the table
  shows, so a bold caption above it only repeats it. 2.2.2/2.2.4/2.2.5 keep a
  plain `<p style="margin:0">` note above the table where the table needs a
  caveat (coverage/overlap, the ranking rule, the ordering rule) -- that is a
  note, not a title.
- ms with two decimals, percentages with two decimals and a `%`; half-up
  rounding, so a hand check against the CSV agrees with the report.
- Percentages are computed from durations, never summed from the tables'
  already-rounded percentage columns.
- 2.3 is conditional: `build_final_report.py` only emits it when the pipeline
  table actually has a draft phase row with children (speculative decoding
  was on for this capture). Draft's own children are not always broken down
  by variant -- a CUDA-graph capture can only place the draft window's
  boundary, so its children are one aggregate "draft N 层 forward" row plus
  一个 其他 residual; an eager capture with a detected MTP/NextN layer
  segments draft by variant exactly like target's 2.2.1. Both shapes render
  through the same table, so do not assume 2.3.1 always looks like 2.2.1.
- 2.2.4 ranks by each kernel's own total duration across every module and unit
  position it appears in -- one row per kernel, never split into one row per
  (kernel, module) pair, so 启动次数 and 耗时(ms) always agree with the totals
  reported elsewhere in the package. 所属模块 lists every module the kernel
  contributed to, ordered by that module's own share of the kernel's time.
- Every breakdown table carries the unit's own total, so a column can be read
  against the whole instead of only against its siblings: 2.2.1/2.3.1 put the
  phase total (`Target 主模型总耗时` / `Draft 模型总耗时`) in the first column,
  2.2.2/2.2.3 put `pattern总耗时` there, and 2.2.4/2.2.5 -- whose rows are
  entities, not metrics -- put it in footer rows instead (`Top N 累积耗时` +
  `pattern总耗时`, `核心计算合计` + `pattern总耗时`). The children never sum
  exactly to that total: they fall short by the unclassified remainder, or
  exceed it under multi-stream overlap. Showing the total is what makes that
  gap visible; do not normalise the columns to make them add up.
- 2.2.5 lists core-compute kernels in **execution order**, taken from the origin
  table's `start_ns` within one `unit_position` and joined to the other tables on
  `module`. Do not take the row order of `<prefix>_core_compute_table.csv` or the
  operator overview for this -- both are grouped by functional module, which puts
  an attention output projection ahead of the attention core that feeds it. MFU
  and MBU are the mean over the kernel's occurrences in the unit (they differ by
  well under a percentage point across unit positions); a kernel with no shape
  evidence keeps an empty MFU/MBU rather than a fabricated one.
