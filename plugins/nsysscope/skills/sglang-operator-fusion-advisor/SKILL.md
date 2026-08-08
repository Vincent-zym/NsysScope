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

Candidate selection is done by a deterministic prescan against an imported
fusion-pattern registry, not by free judgement. Every suggestion therefore
carries a `verdict` saying whether the fusion already exists upstream and is
simply not in effect here, or whether it is a genuinely new idea.

Write two files at the job directory root, next to `analysis.json`:

- `candidates.json` — the prescan output (evidence)
- `optimization.json` — the final report (schema below)

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
2. `candidates.json` — produced by
   [scripts/scan_fusion_candidates.py](scripts/scan_fusion_candidates.py). This
   is the authoritative candidate list; see step 1 of the workflow.
3. `<prefix>_operator_origin_table.csv` and `<prefix>_auxiliary_operator_table.csv`
   — authoritative kernel names, durations and call chains.
4. `<prefix>_analysis_manifest.json` — hardware, dtypes, MFU/MBU evidence.
5. The model source tree (`source_path`) — to confirm the registry's candidate
   implementation really exists here, and to check whether a split form is
   deliberate (behind a flag, disabled under CUDA graph, etc.).
6. When running under a provider with web search (see
   [references/research-and-estimation.md](references/research-and-estimation.md)):
   check whether the upstream project already ships a fused implementation for
   an unexplained cluster, and cite the exact file/PR/commit found.

## Read references

- Read [references/candidate-selection.md](references/candidate-selection.md)
  before interpreting `candidates.json`.
- Read [references/research-and-estimation.md](references/research-and-estimation.md)
  before writing `rationale`, `referenceLinks` or `estimatedGainPct`.
- Read [references/output-schema.md](references/output-schema.md) before
  writing `optimization.json`.
- [references/fusion-registry.json](references/fusion-registry.json) is the
  imported pattern registry consumed by the prescan. Do not edit it by hand;
  regenerate it with
  [scripts/refresh_fusion_registry.py](scripts/refresh_fusion_registry.py).

## Workflow

1. **Run the prescan first.** It is deterministic and it owns candidate
   selection:

   ```bash
   python3 scripts/scan_fusion_candidates.py \
     --analysis <job_dir>/analysis.json \
     --source <source_path> \
     --out <job_dir>/candidates.json
   ```

   Add `--unit-variant <variant>` (or `--unit-position` / `--unit-id`) when the
   user pinned a repeating unit; otherwise the heaviest unit is chosen.
2. Read `candidates.json`. Split its rows three ways:
   - `registryMatches` with `isActionable: false` → `activePatterns`
   - `registryMatches` with `isActionable: true` → suggestions, `source: "registry"`
   - `adjacencyClusters` with `explainedByRegistry: false` → suggestions,
     `source: "cluster"`, subject to the bar in candidate-selection.md
3. For every candidate, open the resolved `candidatePaths` in the source tree and
   confirm the fused implementation is really there and reachable for this model
   and dtype. If it is deliberately disabled, say so instead of reporting a defect.
4. When web search is available, use it only for `source: "cluster"` candidates —
   registry rows already carry their evidence.
5. Draft 1–3 ranked options per suggestion following
   [references/research-and-estimation.md](references/research-and-estimation.md)'s
   convergence rule — never more than 3, always sorted by `estimatedGainPct`
   descending, always with a stated `confidence` and `estimatedGainBasis`.
6. Write `optimization.json` per
   [references/output-schema.md](references/output-schema.md), carrying over
   `scope`, `limits` and `prescan.registryProvenance` from `candidates.json`.
   If nothing cleared the bar, write `suggestions: []` — do not pad output.
7. Validate before finishing:

   ```bash
   python3 scripts/validate_optimization_package.py <job_dir>/optimization.json \
     --analysis-json <job_dir>/analysis.json \
     --candidates-json <job_dir>/candidates.json
   ```

## What this skill must never do

- Never claim a registry pattern is missing when the prescan did not report it.
- Never overrule or rewrite a prescan `verdict`.
- Never put an informational verdict (`* 已生效`) into `suggestions`.
- Never suggest fusing across two different repeating-unit positions or variants.
- Never claim a specific numeric gain without stating the basis
  (`estimatedGainBasis`) for that number.
- Never cite a GitHub link, PR, or file path that was not actually found in
  this run's material or web search — no invented references.
- Never emit more than 3 `options` per suggestion.
- Never propose splitting an already-fused kernel; this skill only fuses.
- Never write to any file other than `optimization.json` and `candidates.json`.
