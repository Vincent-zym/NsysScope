# final_report.md format

`final_report.md` is the package's human-facing deliverable. It is pasted into
如流知识库 (the team wiki), which is why the tables are inline-styled HTML instead
of markdown pipe tables. `scripts/build_final_report.py` emits the whole
skeleton; this file explains what it emits and why, so the agent can fill in the
prose without breaking the paste.

`final_report.example.md` in this directory is a complete filled-in report (a
glm5_next prefill analysis, reviewed and accepted by the requester). Read it as
the reference for depth and tone: how long a conclusion is, what a 潜在优化点 item
looks like, and how much explanation belongs next to a table. When in doubt, match
it rather than inventing a new shape.

## Section layout

```text
# <model> <stage> 性能分析报告        结果包 / 硬件+rank / 阶段+trace / 模型结构
# 1. 结论                            3~5 conclusions, then #### 潜在优化点
2. 链路与算子耗时分析                 本节结论 -> 空行 -> 以下是具体分析过程 -> 分析思路
  2.1 forward 链路耗时               Token 链路耗时 + Target 内部构成
  2.2 算子耗时分析                   按功能模块划分（按执行顺序）+ 按算子类型划分
# 3. 算子分析工具数据                 popo 发布链接
# 4. 物料                            nsys 文件路径
```

Section 2 and its two subsections are emitted as `<h1>`/`<h2>` HTML, not markdown
headings, for the margin reason below. Sections 1, 3 and 4 stay markdown.

## What the reader wants

- Conclusions are facts plus numbers plus consequence, ordered by impact. A
  conclusion nobody can act on does not belong in section 1.
- 潜在优化点 items need a measured basis in this package. Do not suggest a runtime
  flag whose effect this capture cannot show.
- Do not explain the methodology, the verification steps or the closure
  invariants. That belongs in references/output-spec.md, not in a report a
  performance engineer reads to decide what to optimise.
- Do not attribute a gap or a low MFU to a cause the tables do not evidence.

## Paste-fidelity rules (learned the hard way)

如流's editor keeps some of the pasted HTML and silently drops the rest. What
survives, and what does not, is not obvious -- these were established by trial:

- **Kept**: per-cell `style` (border, `background`, `text-align`,
  `vertical-align`), `<b>`, `<code>`, `<h1>`/`<h2>`, `border`/`cellpadding`
  attributes.
- **Dropped**: `<caption>` entirely -- put the table title in a
  `<p style="margin:0">` line immediately above the table instead.
- **Dropped**: every width declaration -- `style="width"`, `min-width`,
  `padding`, `<table width>`, `<colgroup><col width>`, `<th width>`. Column
  widths are recomputed from the content, so do not try to set them. Widen a
  column by making its header wording longer (`Forward 耗时` instead of
  `Forward`); leading/trailing full-width spaces get trimmed and do not work.
- A blank line in the markdown source, and any rendered `margin-bottom` on the
  block above a table, both come through as an extra blank line. Hence
  `margin:0` on every block, and no blank line between a block and the table
  that follows it. Where a blank line *is* wanted, use an explicit
  `<p style="margin:0">&nbsp;</p>` spacer -- a `<br>` stacks with the
  surrounding margins and yields three blank lines.

## Table conventions

- Metrics are rows, entities are columns, so a table stays readable when it has
  eleven functional modules.
- Header row background `#b4c7e7`, first column `#d9e2f3`, every cell centred.
- Table title above the table, bold except the forward-config note.
- ms with two decimals, percentages with two decimals and a `%`; half-up
  rounding, so a hand check against the CSV agrees with the report.
- Percentages are computed from durations, never summed from the tables'
  already-rounded percentage columns.
