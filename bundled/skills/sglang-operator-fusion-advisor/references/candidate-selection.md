# Candidate selection for operator fusion

## Contents

1. What counts as a candidate group
2. Minimum bar to report
3. Grouping rules
4. What to exclude

## What counts as a candidate group

A candidate group is a contiguous (in timeline order) run of kernels within
the single requested repeating unit where **at least one** of these holds:

- Two or more adjacent `辅助算子` (auxiliary) kernels with no intervening
  core/communication kernel — e.g. a cast, then an elementwise add, then
  another cast.
- One or more auxiliary kernels that sit immediately before or after a core
  GEMM/attention kernel and plausibly belong to that kernel's
  prologue/epilogue (e.g. a quantize kernel immediately before a GEMM that
  consumes its output, or a bias-add/activation kernel immediately after).
- A small repeated auxiliary sequence that recurs at every occurrence of a
  sub-pattern inside the unit (e.g. the same norm+cast pair before every
  expert GEMM in a MoE block) — report once per distinct sequence, not once
  per repetition, but note the repetition count in `rationale`.

## Minimum bar to report

Only report a candidate group when you can point to a concrete mechanism,
not just "these are small and adjacent so maybe they could be fused". At
least one of:

- The same or a structurally similar fusion is already implemented
  elsewhere in the supplied source tree (cite file:line).
- The same fusion is documented or implemented upstream (cite the found
  file/PR/commit from web search, when available).
- The group's combined duration is a non-trivial fraction of the unit's
  total wall span (a reasonable default is >1% of
  `summary.normalizedLayerDurationUs`, but do not treat this threshold as a
  hard requirement if a smaller group has strong mechanism evidence anyway)
  AND the kernels are provably memory-bound (elementwise/cast/copy patterns,
  or a low `mbu`-implied intensity), which is the profile that benefits most
  from kernel fusion.

If a candidate group fails this bar, do not include it — an empty
`suggestions` array is a valid and expected output for units with nothing
actionable.

## Grouping rules

- Group by shared `unitPosition`/`unitId`/`unitVariant` (already filtered
  upstream) and contiguous `startNs` ordering.
- Do not merge two groups that are separated by a core/communication kernel
  unless the intervening kernel is itself one of the group's fusion targets
  (e.g. "fuse the cast into the GEMM epilogue" legitimately spans the cast +
  the GEMM).
- Keep `targetOperators` as the exact `index` values from `analysis.json`
  for every kernel involved, including the core kernel if it's a fusion
  target (not just the auxiliary kernels).

## What to exclude

- Communication kernels (`category: communication`, e.g. NCCL AllReduce/
  SendRecv) are never fusion candidates in this skill — that is a distinct
  optimization space (overlap/scheduling), not operator fusion.
- Kernels whose `dispatchCodeSnippet`/`pythonFunction` shows they are
  already part of a fused call (e.g. the name itself says `_fused_`) should
  not be re-suggested for the same fusion.
- Do not propose splitting an already-fused kernel into more kernels — this
  skill only proposes fusing, never un-fusing.
