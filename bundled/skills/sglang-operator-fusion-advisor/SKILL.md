---
name: sglang-operator-fusion-advisor
description: "Analyze an existing sglang-nsys-static-analysis six-table/analysis.json package plus the model's source tree, and propose operator-fusion/optimization opportunities for auxiliary kernels within a single repeating unit. Use only as an optional follow-up to a completed static-analysis job. Do not use for producing the six-table package itself, and do not use for cross-layer or cross-model comparisons."
---

# SGLang operator fusion advisor

## Outcome

Given a completed `sglang-nsys-static-analysis` package (six CSVs +
`analysis.json` + `<prefix>_analysis_manifest.json`) and the model's source
tree, identify auxiliary (non-core) kernels within **one repeating unit**
(one `unit_position` × `unit_id` × `unit_variant`) that are plausible fusion
or optimization candidates, and propose concrete, evidence-backed options.

Write exactly one file: `optimization.json` (schema below), placed at the job
directory root next to `analysis.json`.

## Scope boundaries

- **Single repeating unit only.** Compare and group candidates strictly
  within one `(unit_position, unit_id, unit_variant)` triple from the input
  package. Do not propose fusions that span two different layers/positions
  or two different variants — that comparison is out of scope for this skill.
- **Auxiliary-kernel focus.** Primary targets are `辅助算子` (auxiliary)
  category rows — elementwise, cast, copy, norm epilogues, gather/scatter,
  quantize/dequantize, pack/unpack, and similar small kernels — especially
  when several such kernels execute back-to-back around the same core GEMM
  or attention kernel. Core GEMM/attention kernels themselves are only ever
  discussed as fusion *targets* (e.g. "fuse into the preceding GEMM
  epilogue"), never re-scored on their own compute efficiency — that is
  `mfu`/`mbu`'s job, not this skill's.
- **No invented kernels.** Every suggestion must name kernels that actually
  appear in the input package's origin table. Do not suggest fusing a kernel
  that isn't in the analyzed window.
- **Not a substitute for the static-analysis skill.** This skill never
  regenerates or edits the six-table package, `analysis.json`, or any of
  their sidecars. Treat them as read-only input evidence.
- **Read-only on user materials.** Never edit the model source tree,
  config, or any supplied evidence file.

## Required evidence

Collect, in order of how they're used:

1. `analysis.json` — the primary index of operators, stages, categories,
   `unitPosition`/`unitId`/`unitVariant`, `mfu`/`mbu`, `pythonFunction`,
   `dispatchCodeSnippet` (when present).
2. `<prefix>_operator_origin_table.csv` and `<prefix>_auxiliary_operator_table.csv`
   — authoritative kernel names, durations and call chains.
3. `<prefix>_analysis_manifest.json` — hardware, dtypes, MFU/MBU evidence.
4. The model source tree (`source_path`) — to check whether a candidate
   fusion is already implemented elsewhere in the same codebase (e.g. a
   fused kernel variant exists behind a flag, or a similar module already
   uses a fused epilogue).
5. When running under a provider with web search (see
   [references/research-and-estimation.md](references/research-and-estimation.md)):
   check whether the upstream project (e.g. `sgl-project/sglang`,
   `flashinfer-ai/flashinfer`, vendor kernel libraries referenced in
   `dispatchCodeSnippet`/`pythonFunction`) already ships a fused
   implementation, and cite the exact file/PR/commit found.

## Read references

- Read [references/candidate-selection.md](references/candidate-selection.md)
  before selecting which auxiliary kernels to flag.
- Read [references/research-and-estimation.md](references/research-and-estimation.md)
  before writing `rationale`, `referenceLinks` or `estimatedGainPct`.
- Read [references/output-schema.md](references/output-schema.md) before
  writing `optimization.json`.

## Workflow

1. Load `analysis.json` and filter to `operators` matching the single
   requested `(unitPosition, unitId, unitVariant)` (or, if the job's
   repeating unit is homogeneous with only one variant, that variant).
2. Walk the filtered operators in timeline order (`startNs`). Identify runs
   of adjacent auxiliary kernels, and auxiliary kernels immediately
   surrounding a core kernel, as candidate groups per
   [references/candidate-selection.md](references/candidate-selection.md).
3. For each candidate group, check the model source tree for an existing
   fused implementation or a documented pattern that already does this
   fusion elsewhere in the codebase.
4. When available, use web search to check upstream projects for an
   existing fused kernel/PR that matches the candidate group's operators
   and dtypes.
5. Draft 1–3 ranked options per candidate group following
   [references/research-and-estimation.md](references/research-and-estimation.md)'s
   convergence rule — never more than 3, always sorted by
   `estimatedGainPct` descending, and never propose an option without a
   stated `confidence` and `estimatedGainBasis`.
6. Write `optimization.json` per
   [references/output-schema.md](references/output-schema.md). If no
   candidate group in the requested unit clears the minimum bar in
   candidate-selection.md, write an empty `suggestions: []` array — do not
   pad output with speculative entries.

## What this skill must never do

- Never suggest fusing across two different repeating-unit positions or
  variants.
- Never claim a specific numeric gain without stating the basis
  (`estimatedGainBasis`) for that number.
- Never cite a GitHub link, PR, or file path that was not actually found in
  this run's material or web search — no invented references.
- Never emit more than 3 `options` per suggestion.
- Never write to any file other than `optimization.json`.
