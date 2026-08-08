# Candidate selection for operator fusion

## Contents

1. The prescan owns candidate selection
2. Reading `candidates.json`
3. Minimum bar to report
4. What to exclude
5. Limits you must not talk past

## The prescan owns candidate selection

Candidate selection is **not** a judgement call any more. Run
`scripts/scan_fusion_candidates.py` first and work only from its
`candidates.json`. The script is deterministic; you are not.

```bash
python3 scripts/scan_fusion_candidates.py \
  --analysis <job_dir>/analysis.json \
  --source <source_path> \
  --out <job_dir>/candidates.json
```

Useful flags:

- `--unit-variant KDA-MoE` pins the repeating unit. Without any `--unit-*` flag
  the script picks the unit with the largest summed duration.
- `--include-origins mainline,inflight,upstream` adds the cross-framework rows.
  They are off by default because vLLM / TensorRT-LLM / TokenSpeed keywords are
  noisy against an SGLang trace; turn them on only when the user asks about
  migrating an optimization from another framework.
- `--pattern-window-ops` (default 6) controls how far apart a split pattern's
  witness kernels may sit, measured in operator positions.

The script's authority is binding in both directions:

- **You may not invent a registry match.** If `registryMatches` does not name a
  pattern, you may not claim that pattern is missing. The validator rejects it.
- **You may not overrule a verdict.** If the prescan says `已有实现未启用`, do not
  rewrite it as a new idea; if it says `需人工确认`, do not upgrade it.
- You may *decline* to turn a prescan row into a suggestion (for example when the
  source tree shows the split form is deliberate) — say so in the option's
  `rationale`, and keep the prescan's verdict.

## Reading `candidates.json`

Three arrays matter:

- `registryMatches` — each row already carries `verdict`, `confidence`,
  `targetOperators`, `familySharePctOfUnit` and resolved `candidatePaths`.
  Rows with `isActionable: false` are already-active patterns: put them in
  `optimization.json` → `activePatterns`, never in `suggestions`.
- `adjacencyClusters` — timeline-adjacent auxiliary kernels on one stream.
  `explainedByRegistry: false` plus a non-zero `nativeElementwiseCount` is the
  strongest "細碎算子未走任何融合路径" signal available, because a kernel named
  `vectorized_elementwise_kernel` / `index_elementwise_kernel` /
  `unrolled_elementwise_kernel` is a raw framework kernel by construction.
- `operators` — the full per-operator table for the scoped unit, including
  `kernelFamily`, `nativeElementwise`, `alreadyFused`, `mbu`. Use it to write
  the memory-traffic arithmetic in `estimatedGainBasis`.

## Minimum bar to report

A `registryMatches` row with `isActionable: true` has already cleared its bar
(the registry's own `minSharePct`) — report it.

For an `adjacencyClusters` row, require at least one of:

- `nativeElementwiseCount >= 1` — raw framework elementwise kernels in the run.
- `sharePctOfUnit` above the run's `minClusterSharePct` **and** a mechanism you
  can name: a structurally similar fusion elsewhere in the supplied source tree
  (cite file:line), or an upstream implementation found via web search this run.

"These are small and adjacent so maybe they could be fused" is not a mechanism.
If a cluster has no mechanism, leave it out. An empty `suggestions` array is a
valid and expected result.

## What to exclude

- Operators with `alreadyFused: true` — never propose re-fusing them, and never
  propose splitting them. This skill only fuses.
- `category: communication` kernels are never fusion candidates here; overlap and
  scheduling are a different optimization space.
- Anything outside the single scoped `(unitPosition, unitId, unitVariant)`. Do
  not propose a fusion spanning two positions or two variants.

## Limits you must not talk past

Copy `candidates.json` → `limits` into `optimization.json` → `limits` verbatim,
and respect them in your prose:

- The trace shows what actually ran, not what is legally fusable. Adjacency and
  same-stream are necessary clues, not proof of a data dependency.
- A missing fusion may be deliberate (for example a dual-stream split that is
  intentionally disabled under CUDA graph capture). Check the source tree before
  calling it a defect.
- Keyword matching can collide on same-named different implementations.
- `candidatePaths` resolution is by file name; symbol-level existence is not
  verified.
