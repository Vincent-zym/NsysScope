# Research, rationale and gain estimation

## Contents

1. Web search availability
2. Rationale requirements
3. Estimating gain
4. Convergence rule (why not more options)

## Web search availability

Web search is not guaranteed. This skill runs under two providers:

- **Codex**: has built-in web search when the invocation enables it. If
  available, use it to check upstream repositories (primarily
  `sgl-project/sglang`, `flashinfer-ai/flashinfer`, and any vendor kernel
  library named in the candidate group's `dispatchCodeSnippet`) for an
  existing fused kernel, open PR, or design doc that matches the candidate.
- **Comate**: may not have an equivalent tool in this run. When no web
  search tool is available, say so explicitly in that suggestion's
  `estimatedGainBasis` (e.g. "本次运行未接入网络检索，收益基于访存字节数估算") instead
  of silently omitting the caveat, and do not fabricate a `referenceLinks`
  entry.

Never claim "official implementation confirmed at <URL>" without having
actually retrieved that URL's content in this run.

## Rationale requirements

Every option's `rationale` must state the **mechanism**, not just the
outcome. Good: "两次连续的 `per_token_group_quant` + `elementwise_add` 可以合并进
后续 GEMM 的 epilogue，参考同代码库 `moe/fused_moe_kernel.py` 中 `use_fp8_w8a8` 分支
已实现的类似融合". Bad: "这两个算子可以融合，能提速".

State one of:
- an existing implementation found in the supplied source tree (cite
  file:line)
- an existing implementation found via web search (cite URL/commit/PR)
- a first-principles mechanism (e.g. "these are both memory-bound
  elementwise ops on the same tensor shape; fusing avoids one full
  read+write round trip to HBM") — only when neither of the above applies,
  and label `confidence: low` in that case

## Estimating gain

`estimatedGainPct` is always an estimate of **wall-clock duration reduction
for the candidate group**, not for the whole unit. Compute it from one of:

1. **Measured comparable case**: if source or web search turns up a
   before/after benchmark for the same or a structurally similar fusion,
   use that ratio and cite it in `estimatedGainBasis`.
2. **Memory-traffic reduction estimate**: for elementwise/cast/copy chains,
   estimate bytes read+written per kernel from the operator's known
   `shape`/dtype where available (or the tensor's known element count from
   `pythonFunction`/config), sum across the group, and compare against the
   estimated bytes for a single fused pass (typically the largest single
   read + the final write, since intermediate round-trips are eliminated).
   State this arithmetic in `estimatedGainBasis`.
3. **Launch-overhead reduction estimate**: when kernels are small and
   duration is dominated by fixed launch overhead rather than data volume
   (e.g. each kernel takes only a few microseconds), estimate the gain as
   the number of eliminated launches × a stated fixed overhead assumption
   (be explicit about the assumed per-launch overhead figure used).

Never report `estimatedGainPct` above what the group's own combined
duration allows (a fused kernel cannot take negative time) — sanity check
against the group's summed `durationUs` from `analysis.json`.

## Convergence rule (why not more options)

Cap every suggestion at 3 `options`, sorted by `estimatedGainPct` descending.
When more than 3 plausible approaches exist:

- Prefer the option with the strongest mechanism evidence (existing
  implementation > web-search-confirmed > first-principles) as option 1.
- Drop options that differ from another option only in minor
  implementation detail without a materially different estimated gain or
  risk profile — consolidate them into one option and mention the
  variants in that option's `rationale` instead of listing them separately.
- If two options have the same `estimatedGainBasis` mechanism but target
  slightly different kernel subsets, keep only the one covering more of the
  candidate group's kernels, unless the smaller one is meaningfully lower
  risk (state that tradeoff in `rationale` if so).
