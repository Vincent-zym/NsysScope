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

`build_final_report.py <package> --check` enforces the mechanical half of this
document: it regenerates the skeleton, requires every generated line to be
byte-identical (so a hand-added row or a rewritten generated note fails there),
and measures each filled slot and each prose line. Length is counted as 中文字数
plus one per run of Latin/digits, because an identifier like
`flashinfer_mxfp4` is not compressible -- by that measure the calibration
sentence below is 28 and the 240-character version it replaced was 90. The
limits are 75 for a prose line and 35 for a section-1 config value. Run it
after filling the markers; when it flags a generated line, fix the generator.

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
                                       首列 序号，含所属模块、占pattern耗时、
                                       启动次数，表尾 Top N 累积耗时 +
                                       pattern总耗时 两行
    <h3>2.2.5 按核心计算统计</h3>  一个 pattern 内全部核心计算算子，首列 序号，
                                     按执行顺序，含 shape/MFU/MBU/启动次数，
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
  Concretely, it does **not** carry the sampling statistics behind the wall time
  (`132 个稳态样本均值，min 40.21 / max 43.37 ms`) nor the per-variant 单层耗时 --
  the first is methodology, the second is already 2.2.1's own row, and both are
  what turned a 100-character line into a 240-character one in practice.
- The notes above 2.2.2/2.2.4/2.2.5 are generated, already carry the numbers they
  need, and are not an invitation to append. If a generated note is factually
  wrong for this package (e.g. it says 余量为未归类的零散算子 when every operator
  is classified), fix `build_final_report.py` so it derives the right wording from
  the data -- do not hand-patch the report and leave the generator wrong.
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
  widths are recomputed from the content, which is also why short cells are
  centred rather than left-aligned: a left-aligned column narrower than the
  header text next to it gets clipped in 如流. Widen a column by making its
  header wording longer (`Target 主模型` instead of `Target`);
  leading/trailing full-width spaces get trimmed and do not work.
  Section 3 (输出物料) is the one left-aligned table: a multi-line launch prompt
  and a file list read as a wall of text when centred, and those values are long
  enough that the clipping problem never arises. Everything else, including
  section 1's fact table, stays `text-align:center`.
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
- Attachments need the editor JSON (mdsl has no `attachment` node), so they are a
  second pass: read the page back, swap the `nsys 文件`/`工具产物` value cells for
  attachment nodes, `cover` the whole document, publish. Uploads are capped at
  64 MiB. The `.nsys-rep` goes up as-is -- when it is missing or over the cap the
  cell keeps the file name the report already shows, never a compressed
  substitute -- while the result-directory zip falls back to repacking without
  `trace/`. A cover issued right after the mdsl publish is accepted and then
  silently discarded, so that pass retries until the attachments read back.
- `textAlign` is only stored when it differs from the editor default, so a cell
  with no `textAlign` is left-aligned, not centred. An attachment inherits the
  alignment of the cell it replaces, which is what keeps section 1's attachment
  centred and section 3's left-aligned without configuring each separately.

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

- 2.2.4 and 2.2.5 carry a leading 序号 column so a row can be referred to by
  number in a discussion; it is the row's position in that table's own ordering
  (ranking for 2.2.4, execution order for 2.2.5), not an id from any CSV. The
  footer rows put their label in that column.
- Metrics are rows, entities are columns, so a table stays readable when it has
  eleven functional modules -- except the vertical fact tables in sections 1
  and 3, and 2.2.4/2.2.5, where each row is one entity (one config field, one
  kernel, one core-compute operator) because there is exactly one metric per
  row, not several.
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
- 算子名称 keeps only what distinguishes one kernel from another **in this
  package**, decided from the package's whole name set rather than per name.
  A template argument that is identical across every specialization of the same
  base name tells the reader nothing and collapses into `…`:
  `per_token_group_quant_8bit_kernel<__nv_bfloat16, __nv_fp8_e4m3, (bool)1,
  (bool)1, unsigned int>` is the only specialization of its base, so it shows as
  `per_token_group_quant_8bit_kernel<…>`; two specializations that differ in one
  argument keep exactly that argument. Symbols under 60 characters are left
  exactly as the table spelled them (shortening a readable name only loses
  information), and a label must never merge two different kernels -- when
  collapsing would collide, the dropped arguments come back until the labels
  differ again.
- Symbols whose parameters are baked into the identifier instead of a `<...>`
  list (`bmm_MxE4m3_..._sm100f` at 203 characters, `kernel_cutlass_kernel_...`,
  `fmhaSm103aKernel_QkvBfloat16...`, `triton_poi_fused__to_copy_...`) go through
  the same idea applied to `_`-separated tokens: keep as much as fits in 60
  characters -- a leading run of tokens, or the family name plus the tokens no
  sibling has -- with `…` marking the elision and the last token (variant index,
  arch tag) always kept. There is deliberately **no list of known families**:
  the previous version special-cased two prefixes by name, so `bmm_...` and
  `fmhaSm103aKernel_...` were not shortened at all, and the next new symbol
  would not have been either. When nothing separates two names within the
  budget, the full names are kept -- over-long beats ambiguous.
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
  gap visible; do not normalise the columns to make them add up. 2.2.2's note
  names the cause of the shortfall from the data rather than by habit: when the
  module durations already sum to the operator table's 总计, nothing is
  unclassified and the remainder is the GPU gap between kernels; only when they
  sum below it is the remainder unclassified operators.
- 2.2.5 lists **every core-compute operator** of one pattern in **execution
  order**, one row per operator. An operator here is a (`module`, 算子名称,
  `shape`) triple, not a kernel name: a package whose core table shortens every
  deep-gemm specialization to `sm100_fp8_fp4_gemm_1d1d_impl<…>` has seven
  different GEMMs sharing one name, and keying on the name merged q_b_proj,
  o_proj, gate_up_proj and the indexer projections into one row spanning four
  modules — which is the clustering this table must not do. Rows are aggregated
  over the pattern's repeated unit positions (启动次数 says how many, MFU/MBU are
  the mean over them, differing by well under a percentage point) because the
  same operator at four layer positions is the same line of the timeline, not
  four. The order comes from the origin table's `start_ns` -- the earliest
  occurrence of each operator -- joined on (`unit_position`, `module`, kernel base
  name); the core table shortens a symbol to `name<…>` while origin keeps the full
  `void ns::name<args>`, so only the base names match, and an operator with no
  start evidence goes last rather than being guessed into the middle. Do not take
  the row order of `<prefix>_core_compute_table.csv` or the operator overview for
  this -- both are grouped by functional module, which puts an attention output
  projection ahead of the attention core that feeds it.
