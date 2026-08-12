# Seven-table output specification

## Contents

1. Required artifacts
2. Columns
3. Total rows
4. Taxonomy and semantic mapping
5. MFU and operator categories
6. Validation invariants
7. Forward pipeline table

## Required artifacts

Always create:

- `<prefix>_operator_origin_table.csv`
- `<prefix>_opreator_table.csv`
- `<prefix>_core_compute_table.csv`
- `<prefix>_auxiliary_operator_table.csv`
- `<prefix>_op_classification_table.csv`
- `<prefix>_stage_table.csv`
- `<prefix>_forward_pipeline_table.csv`
- `<prefix>_analysis_manifest.json`
- `<prefix>_architecture_taxonomy.json`

`<prefix>_forward_pipeline_table.csv` is the seventh required table (see "Forward
pipeline table"), not an extra: it is the only place the package says how the measured
repeating unit relates to a whole forward step. It lives in the same directory as the
other tables, so the organised result package carries it under `csv/` with an `xlsx/`
counterpart. A capture that cannot support it — a single forward step, no usable step
marker — is not a valid input for this analysis; fix the capture instead of shipping
six tables.

Use `duration_avg_us` when stable samples exist; otherwise use `duration_us` and
record the fallback. All percentages use the repeating-unit wall-span. Preserve
the structural position and variant in every composite-unit row.

## Columns

Origin:

```text
序号,module,operator_name,duration_us,start_ns,end_ns,device,stream,layer_id,
unit_position,unit_id,unit_variant,duration_min_us,duration_max_us,
duration_diff_us,duration_avg_us,duration_avg_pct_of_total,python_function,
function_introduction,mapping_reason,dispatch_code_snippet
```

`device` and `stream` must be bare integers (`3`, `156`). Do not annotate them
(`cuda:3 (pp_rank 3)`, `GPU 3`) — record rank/PP context in the manifest or in
`mapping_reason` instead. The NsysScope converter tolerates an annotated value by
taking its first integer, but the column contract is the plain id.

`dispatch_code_snippet` is the actual source text of the call-chain's deepest
frame captured at analysis time (the statement that dispatches the CUDA/Triton
work, plus any immediately enclosing branch condition needed to understand why
it fired). It is a text snapshot, not a line-number reference: source line
numbers drift as code changes after the trace was captured, but the quoted
text remains checkable evidence. See
[references/mapping-and-stats.md](references/mapping-and-stats.md) for the
capture rule. Leave blank only when the dispatch site cannot be identified,
and record that in `uncertain_mappings`.

Operator overview:

```text
序号,单元位置,单元ID,单元类型,功能模块,module,算子名称,算子耗时(us),
算子耗时占比(%),shape,mfu,模块耗时(us),模块耗时占比(%),python_function,功能介绍
```

The overview `module` column is mandatory and immediately precedes `算子名称`.
It is copied from the origin table without semantic relabeling.

Core compute:

```text
序号,单元位置,单元ID,单元类型,功能模块,module,算子名称,算子耗时(us),
算子耗时占比(%),shape,mfu,mbu,python_function,功能介绍
```

Auxiliary:

```text
序号,单元位置,单元ID,单元类型,功能模块,算子名称,算子耗时(us),
算子耗时占比(%),python_function,功能介绍
```

Classification:

```text
序号,算子类型,算子数量,总耗时(us),耗时占比(%)
```

Always emit `核心计算`, `通信`, and `辅助算子`, including zero-count categories.

Stage:

```text
序号,单元位置,单元ID,单元类型,功能模块,模块耗时(us),模块耗时占比(%),
代表区间并集(us),代表墙钟跨度(us),耗时口径,功能介绍
```

`模块耗时(us)` is the sum of position-aware average kernel durations.
`代表区间并集(us)` removes overlap among the module's representative intervals.
`代表墙钟跨度(us)` is its first-start to last-end span and includes internal
gaps. None is automatically a critical-path attribution.

Origin keeps full demangled names. Other tables use compact CUDA leaf symbols.
Put semantic descriptions in `功能介绍`, never in `算子名称`.

## Total rows

Every CSV ends with exactly one total row:

- origin: the final `module=__layer_total__` row reports repeating-unit wall
  span and `duration_avg_pct_of_total=100`;
- overview: `序号=总计` reports the sum of all operator average durations in
  both operator/module total fields;
- core and auxiliary: `序号=总计` reports the corresponding category sum;
- classification: the fourth row is `总计`, with total operator count and the
  sum of all three category durations;
- stage: `序号=总计` reports the sum of all position-aware functional modules,
  plus full representative-sample interval union and wall span.

Operator, category and stage totals are accumulated GPU work. They may exceed
the repeating-unit wall span or 100% when streams overlap. Do not replace them
with wall time merely to force 100%.

## Taxonomy and semantic mapping

Create and validate `<prefix>_architecture_taxonomy.json` before semantic
mapping. Follow [architecture-taxonomy.md](architecture-taxonomy.md).

Build ordered model-aware rules with optional:

```text
module_regex,operator_regex,functional_module,operator_name,category,core_kind,
shape,dtypes,unit_position,unit_id,unit_variant,introduction
```

Every `category: core` rule declares `core_kind: gemm` or
`core_kind: attention`.

Keep these concepts distinct:

- `unit_variant`: architecture-defined subtype inside a heterogeneous cycle
- `unit_id`: concrete occurrence such as a network layer
- `module`: fine-grained model-description path
- `功能模块`: broader current-model functional stage
- category: core, communication or auxiliary operator class

Keep `功能模块` coarse enough to support comparison: normally 5–8 ordered
stages per architecture variant. It must not be a renamed one-to-one copy of
`module`. Preserve projections, norms, gates, cache operations, router packing
and other implementation detail in `module`; merge them into the enclosing
architecture phase in `功能模块`. More than eight stages requires the taxonomy's
current-model `granularity_exception`.

Aggregate functional modules by:

```text
(unit_position, unit_id, unit_variant, functional_module)
```

Never aggregate only by functional-module name in a composite unit. Use
module-count heuristics per variant, not as a global cap for a heterogeneous
cycle. Do not merge state-space/linear attention with full/latent attention just
because both can be described as Attention.

## MFU and operator categories

Only core GEMMs may receive shape/MFU. Require verified M/N/K, active branch,
operand formats, accumulator behavior, Tensor Core compute mode, duration and
dense per-GPU theoretical peak:

```text
MFU = 2*M*N*K / (duration_seconds * dense_peak_flops)
```

Use `references/hardware-peaks.json`. Reject MFU above 100%. When shape, compute
mode and verified hardware profile exist, missing MFU is an error.

Core GEMMs also receive an approximate `mbu` (Memory Bandwidth Utilization),
reported alongside `mfu` in the same form — a peak-relative percentage such as
`49.72%` — and left empty when the byte estimate or the bandwidth peak is
unavailable:

```text
accessed_bytes = (M*K + K*N + M*N) * dtype_bytes
achieved_gb_per_s = accessed_bytes / duration_seconds / 1e9
mbu = achieved_gb_per_s / hbm_bandwidth_gb_per_s * 100   # percent of peak
```

`dtype_bytes` is derived from the GEMM's operand `dtypes` (falling back to the
resolved `compute_dtype`); this is a coarse estimate that ignores cache reuse
and intermediate quantization/dequantization traffic, so treat `mbu` as an
order-of-magnitude read on whether a kernel is bandwidth-bound, not an exact
utilization. `hbm_bandwidth_gb_per_s` comes from the matched profile in
`references/hardware-peaks.json`; when a hardware profile has no bandwidth peak
registered, leave `mbu` empty rather than reporting raw bytes/second.

Classify each operator independently. Core compute is a strict allow-list:

- GEMM/BMM/matmul, including verified fused/grouped expert GEMMs
- actual attention/state-update score, normalization and value aggregation

Communication is separate. Quantization, normalization, RoPE, cache management,
TopK/router/dispatch, gather/scatter, permutations, copies/casts, activations
and metadata transforms are auxiliary even inside a core-owning stage.

## Validation invariants

- exact filenames and headers
- every table has exactly one correctly calculated final total row
- overview has origin `module` immediately before `算子名称`
- valid CSV quoting for demangled templates
- origin coverage equals selected kernels plus one total row
- every non-total row occurs once in overview and one category
- core category is never inherited from its broader functional module
- total duration is repeating-unit wall-span, not summed kernels
- full Python call chains and evidence-backed mapping reasons
- no final unknown/misc/other modules
- runtime/source/config conflicts are recorded
- every distinct variant and position is present in origin, overview and stage
- every variant emits its ordered and distinctive functional modules
- heterogeneous cycle duration is not called generic single-layer duration
- fused kernels use indivisible attribution unless a trace proves separate work
- frontend unit/variant/category/sample/device data matches package evidence

Legacy packages without taxonomy remain importable, but new composite
analysis must satisfy this contract.

### Hard gates against an evidence-free package

These are checked mechanically, because a package can satisfy every structural
invariant above and still be useless. Each one was added after a real package
passed validation while being unusable:

- **call sites**: at least 60% of origin rows carry a `file:line` in
  `python_function`. Real packages score 80-100%; the rejected one scored 0.
- **descriptions**: at least 60% of origin rows have a Chinese
  `function_introduction`. A single English noun phrase (`KDA beta gate
  projection`) is not a functional description.
- **snippets**: when `dispatch_code_snippet` exists, at least 60% of rows must
  contain a real call expression. Prose in that column is not evidence.
- **shapes**: the core-compute table must have at least one filled `shape`. All
  three of shape/mfu/mbu blank across every GEMM is a refusal to do the work, not
  a lack of evidence.
- **attribution**: no operator may last longer than the repeating unit it is
  assigned to, and a handoff kernel (`SendRecv`) taking more than 20% of a
  position is a rank-level wait, not layer work. Small intra-layer collectives
  (TP all-reduce, DCP all-to-all at a few percent) stay allowed.
- **window alignment**: positions of the *same* variant must hold within 15% the
  same operator count. `32 / 41 / 42` for three KDA positions means the unit window
  is phase-shifted against the layer boundary.
- **sampling**: `single_sample_fallback` is rejected, and fewer than 3 accepted
  occurrences is rejected.

None of these are waived by an unverified source commit; see
[runtime-evidence-and-mfu.md](runtime-evidence-and-mfu.md).

## Forward pipeline table

`<prefix>_forward_pipeline_table.csv` describes one **forward step** — the full
decode iteration, not the repeating layer unit. Where the other tables answer
"where does the time inside one layer go", this one answers "where does the time
inside one output token go".

```text
环节,环节类型,层数,子步数,单次耗时(us),总耗时(us),占forward步(%),占父环节(%),
样本数,min_us,max_us,备注
```

### Phase decomposition

A step is measured as four **contiguous, non-overlapping** intervals in execution
order, anchored at the target model's forward:

- `target` — the target/verify model forward
- `prep draft` — everything between the end of the target forward and the start of
  the draft forward: verify acceptance, KV bookkeeping, next-draft input assembly
- `draft` — the draft model forward plus its own output head and speculative
  sampling loop
- `prep verify` — everything between the end of the draft forward and the start of
  the next target forward: speculative-token concat, page-table build

They are **reported** as two top-level phases. `prep draft` and `prep verify` are the
target/verify path's own host-side bookkeeping and do not scale with layer count,
which is exactly what `other` collects, so they are folded into the `target` phase's
`其他` row instead of being separate environments. A step therefore splits into
`target` + `draft`, where `target` = target forward + prep draft + prep verify. Say so
in the target row's and the `其他` row's `备注`, and keep the split available as
`forward_pipeline.prep_draft_us` / `prep_verify_us` in the manifest.

Without speculative decoding there is no draft model: emit only the `target` phase
with its `prep draft` stage covering the inter-step bookkeeping, and record
`speculative_tokens: 0` in the manifest.

Nested rows use `环节类型`:

- `total` — the whole step; exactly one row
- `phase` — a top-level phase (`target`, `draft`)
- `variant` — a layer variant inside `target` (KDA, MLA, ...), with `层数` set
- `stage` — a named sub-interval inside a phase (for example the draft forward)
- `other` — the remainder of a phase after its `variant`/`stage` rows. Non-layer
  work such as embedding, lm_head, output all-gather and the sampling loop belongs
  here, because it does not scale with layer count and keeping it separate makes
  layer cost comparable across models.
- `gap` — the inter-token gap; see below. Does not participate in any sum.

### Step boundary detection

Boundaries must come from a kernel that fires **exactly once per forward**. Record
the chosen marker, its launch count and the derived step count in the manifest
under `forward_pipeline.step_marker`. Never infer boundaries from a fixed period.

A reliable pattern on SGLang decode captures: the vocab-parallel embedding kernel
fires once per model forward, and its `gridX` separates draft from target because
it equals `batch * draft_block` for the draft and `batch * (draft_block + 1)` for
the target verify. That also yields the batch size — record how it was derived, or
state that it came from the launch config instead.

Cross-validate with a second once-per-step kernel and report both counts. If the
two disagree by more than one step, treat the segmentation as failed rather than
publishing it.

### Phase boundaries without CUDA graphs

On a CUDA-graph decode path each phase carries its own `graphId`, which is the most
reliable discriminator: the target forward, the draft forward and the non-captured
host bookkeeping fall into distinct graphs. Prefill and eager decode have no graph,
and prefill **may still run speculative decoding**, so a graph-free fallback is
required rather than optional:

- the target forward ends at the end of its **last layer block**, found with the
  same layer segmentation the table pipeline uses (so `--variant-marker` or a
  `--taxonomy` is mandatory in this mode)
- the draft forward **starts** at the draft population of the step marker, and
  **ends** one median layer stride after the last occurrence of its layer core

Consequence to record in `备注` and in `forward_pipeline.draft_boundary_source`: a
forward's tail after its last layer — lm_head, output aggregation, the speculative
sampling loop — is not separable from the bookkeeping that follows it, so it is
counted in the next `prep` phase instead of in the forward. Phase closure still
holds; only the boundary semantics differ from graph mode. Record the mode in
`forward_pipeline.phase_discriminator` (`CUPTI graphId` or
`marker + layer segmentation`) so a reader never compares the two blindly.

`--ignore-cuda-graphs` forces the fallback on a graph-captured trace, which is how
this path is cross-checked: layer counts and the draft layer-forward time must
reproduce the graph-derived ones.

### Step marker and rank selection on a serving capture

A capture taken from a live server is not a benchmark loop, and three of its
properties broke earlier versions of this table:

- **the period jitters.** Chunk fill and queued requests make the per-forward period
  vary (a real prefill capture ranged 349–593ms), so the marker's inter-arrival
  coefficient of variation sits near `0.15`, not `0.01`. The marker search walks
  `cv <= 0.05 / 0.20 / 0.40` and, past the strict tier, only accepts a launch count
  that **at least 3 independent kernels agree on**. Record the tier in
  `forward_pipeline.marker_auto_selected.cv_threshold` / `cv_relaxed`: a relaxed
  marker is legitimate but must stay visible to the reader.
- **gridX does not imply speculation.** The marker's `gridX` follows the per-step
  token count, so prefill shows several populations that have nothing to do with
  draft/verify. Two populations are read as draft vs verify **only** when they repeat
  as a fixed pattern (K draft launches then exactly one verify, identical in every
  period); otherwise all populations are merged into one plain per-forward marker.
- **each pipeline rank holds a different layer mix.** With a `KDA/KDA/KDA/MLA`
  pattern one rank can own 9 KDA + 3 MLA while its neighbour owns 8 KDA + 4 MLA, so
  the busiest device is not necessarily the one the taxonomy describes. Devices are
  ranked by how closely their variant mix matches the declared unit and tried in that
  order; `forward_pipeline.device_candidates` and `device_rejected` record the walk.

### Inter-token gap

`token 间间隙` counts GPU idle **only inside the `prep draft` and `prep verify`
phases**, and only holes longer than the gap threshold (default `50us`, exposed as
`--gap-threshold-us`).

Idle inside the `target` and `draft` phases is deliberately excluded: on a
CUDA-graph replay path it is launch-gap scatter spread over hundreds of kernel
boundaries, not a recoverable stall, and summing it produces a large number that
looks like an optimization opportunity but is not.

Record the threshold and every qualifying hole (start, end, length, phase) in the
manifest so the number is reproducible and can be recomputed under a different
threshold without re-running the analysis. A `0.0` result is a valid, meaningful
answer — write the threshold and the hole count in `备注` so a reader can tell
"measured, none found" from "not measured".

### Closure invariants

All three must hold, with a tolerance no looser than `0.5%` of the step:

- `target + draft = forward step 总计`
- inside `target`: `sum(variant rows) + other = target`, where `other` carries the
  non-layer forward work plus prep draft / prep verify
- inside `draft`: `sum(stage rows) + other = draft`

Plus one sanity bound: `token 间间隙 <= prep draft + prep verify`.

These are the correctness test for the segmentation. A mis-placed boundary marker
shows up as a closure failure, so never "fix" a closure error by adjusting the
`other` row — re-derive the boundaries.

### Statistics

Every row reports `样本数` / `min_us` / `max_us` over the steps actually measured,
and `总耗时(us)` is the **mean** across those steps. The mean, not the median,
because the closure invariants above are checked on these numbers: each step closes
exactly by construction (`其他` is that step's residual), but a median is not
additive — on a jittery capture every row's median comes from a different step and
the children overshoot the phase by over 1% with nothing mis-attributed. Robustness
against one mis-cut step comes from `min_us` / `max_us` and the gap row, not from
the estimator. State the step range used in `备注` when it is not the whole capture.
