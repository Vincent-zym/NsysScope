---
name: sglang-nsys-static-analysis
description: "Analyze an SGLang Nsight Systems `.nsys-rep` or `.sqlite` report, map kernels to model-described modules and Python dispatch sites, and generate a normalized six-table static-analysis package covering the detailed timeline, operator overview, core compute, auxiliary operators, operator-class totals, and functional-stage totals. Use when Codex receives an nsys report plus deployment parameters, model config, architecture notes or source and needs layer/repeating-unit timings, attribution, shapes, MFU, or standardized CSV analysis. Read model evidence first, preserve the captured timeline, leave uncertain shape/MFU blank, and do not use for Nsight Compute `.ncu-rep` diagnosis."
---

# SGLang Nsight Systems Static Analysis

## Outcome

Analyze one complete repeating unit from an existing Nsight Systems capture and
write these six CSV files in a new task directory:

1. `<prefix>_operator_origin_table.csv`
2. `<prefix>_opreator_table.csv` (retain this spelling for compatibility)
3. `<prefix>_core_compute_table.csv`
4. `<prefix>_auxiliary_operator_table.csv`
5. `<prefix>_op_classification_table.csv`
6. `<prefix>_stage_table.csv`

Also write:

- `<prefix>_analysis_manifest.json`
- a repeating-unit/operator-statistics sidecar
- the exported `.sqlite` path when the input is `.nsys-rep`

The trace is authoritative. Emit the smallest complete sequence that actually
repeats; do not split one captured timeline into hypothetical modes.

Do not treat a new model or kernel family as permission to guess. Use the
bundled deterministic audits and fail validation when timing scope,
classification, shape, peak or frontend parity is internally inconsistent.

## Embedded six-table contract

This section is the self-contained runtime source of truth. External development
notes are not part of the skill contract.

### 1. Detailed origin table

Filename: `<prefix>_operator_origin_table.csv`

```text
序号,module,operator_name,duration_us,start_ns,end_ns,device,stream,layer_id,duration_min_us,duration_max_us,duration_diff_us,duration_avg_us,duration_avg_pct_of_total,python_function,function_introduction,mapping_reason
```

- Keep the full demangled `operator_name`.
- Preserve one row per selected kernel plus one `__layer_total__` row.
- `function_introduction` briefly explains the operator's role.
- Keep detailed evidence in `python_function` and `mapping_reason` without
  duplicating the same prose across columns.

### 2. Operator overview table

Filename: `<prefix>_opreator_table.csv`

```text
序号,功能模块,算子名称,算子耗时(us),算子耗时占比(%),shape,mfu,模块耗时(us),模块耗时占比(%),python_function,功能介绍
```

- Generate this immediately after the origin table.
- `功能模块` is a broader architecture stage, not a micro-operation. Merge
  quantization, projection, RoPE, logits, TopK, rearrangement and communication
  belonging to one Indexer path into `NSA Indexer`; retain their distinctions
  only in `module`, category and operator rows. Apply the same principle to
  other cohesive stages instead of translating each fine-grained `module`.
- Review and consolidate the complete layer, not only user-mentioned examples.
  A normal Transformer/MoE repeating unit should usually yield roughly 5–12
  architecture-level functional modules, not one stage per kernel family.
  Derive boundaries from the current model's design and active forward path.
  Transformer examples may include QKV projection, Indexer, Attention
  preparation/core/output, dense FFN or MoE routing/experts, but none of these
  names is mandatory for a model that does not implement that component.
  Merge normalization and quantization into their owning stage unless they
  cross a real functional boundary.
- `算子名称` must be the compact CUDA kernel name, not a semantic function
  label. Remove the return type, outer namespace and runtime argument list while
  retaining the leaf kernel symbol. Keep a short template payload when it
  distinguishes variants (for example
  `sm100_fp8_fp4_gemm_1d1d_impl<224,128,128>`); abbreviate only oversized
  template payloads as `<…>`. Put meanings such as `Q-B 投影`, `输入量化` or
  `路由权重归一化` in `功能介绍`, never in `算子名称`. Keep the untouched
  demangled symbol only in the origin table.
- Repeat aggregate module duration/percentage on every operator row belonging to
  that functional module.
- Fill `shape` and `mfu` only for defensible core-compute GEMMs; leave them blank
  for communication and auxiliary operators.

### 3. Core-compute table

Filename: `<prefix>_core_compute_table.csv`

```text
序号,功能模块,module,算子名称,算子耗时(us),算子耗时占比(%),shape,mfu,python_function,功能介绍
```

- Include only core-compute operators.
- Keep both broader `功能模块` and fine-grained `module`.
- Sort by operator duration descending.

### 4. Auxiliary-operator table

Filename: `<prefix>_auxiliary_operator_table.csv`

```text
序号,功能模块,算子名称,算子耗时(us),算子耗时占比(%),python_function,功能介绍
```

- Include only auxiliary operators.
- Sort by operator duration descending.

### 5. Operator-classification table

Filename: `<prefix>_op_classification_table.csv`

```text
序号,算子类型,算子数量,总耗时(us),耗时占比(%)
```

Always emit exactly these three categories in this order, including zero-count
rows:

1. `核心计算`
2. `通信`
3. `辅助算子`

Every non-total origin row must belong to exactly one category.

Classify each operator independently; never copy a category from its owning
functional module. Treat `核心计算` as a strict allow-list:

- include GEMM/BMM/matmul kernels, including verified fused or grouped expert
  GEMMs
- include the actual Attention score/normalization/value-aggregation kernels
- classify communication kernels as `通信`
- classify everything else as `辅助算子`

Quantization/dequantization, LayerNorm/RMSNorm, RoPE, cache operations, TopK,
router/dispatch, gather/scatter, permutation, copies/casts, elementwise
activations and metadata/index transforms are always auxiliary, even when they
sit inside an Attention, projection, Indexer or MoE functional module. Thus
`per_token_group_quant_8bit_kernel` and `generalLayerNorm` must never appear in
the core-compute table.

For every semantic-map rule marked `category: core`, set `core_kind` to `gemm`
or `attention`. Keep the rule narrow enough to identify one operator family or
fine-grained module; do not mark a broad functional stage as core. An explicit
auxiliary kernel signature overrides a broad or erroneous core rule.

### 6. Functional-stage table

Filename: `<prefix>_stage_table.csv`

```text
序号,功能模块,模块耗时(us),模块耗时占比(%),功能介绍
```

Aggregate operators by the broader Chinese functional module and sort modules by
duration descending.

### Shared numerical rules

- Prefer position-aware `duration_avg_us`; fall back to representative
  `duration_us` only when the manifest marks a single-sample fallback.
- Distinguish a representative sample, a fixed network-layer position and a
  structural repeating unit. Never display a fixed KDA/MLA/NSA layer as the
  model's generic `单层耗时` when the architecture contains multiple layer
  variants. For a composite unit, record its layer count and both total
  repeating-unit duration and normalized average-per-layer duration.
- Use the SQLite/NVTX repeating-unit wall-span (`__layer_total__`) as every
  percentage denominator. Never use the sum of kernel durations.
- Overlapping streams may make module/category percentages sum above 100%;
  report this and never normalize them away.
- Derive prefill shapes from the effective filled chunk/batch behavior, decode
  shapes from the captured batch, and `N/K` from config plus the active source
  branch.
- Compute GEMM MFU as
  `2*M*N*K / (duration_seconds * verified_dense_peak_flops)`.
- Record operand formats in `dtypes`, accumulator behavior separately, and the
  actual Tensor Core `compute_dtype` used to select the peak. Do not limit a
  BF16 Tensor Core operation by the scalar FP32 peak merely because it
  accumulates in FP32.
- Use `references/hardware-peaks.json` for its verified B200/B300 per-GPU dense
  peaks. A semantic map may override a peak only with an equally explicit
  source. Never divide a system peak incorrectly or use a sparse peak.
- When hardware, shape and compute mode are all verified, MFU is required for
  each eligible GEMM. “New model” is not a reason to leave it blank.
- Treat MFU above 100% as a validation failure requiring correction of shape,
  duration, dtype or peak—not as a presentable result.
- Leave shape/MFU blank if M/N/K, datatype, active branch, duration or verified
  dense hardware peak is uncertain. Never substitute a sparse peak.

## Required evidence

Collect existing paths/context before asking. Required:

- `.nsys-rep` or pre-exported `.sqlite`
- model description/design notes naming architectural submodules
- relevant layer/forward source
- deployment command or YAML
- model config

Useful when available:

- target layer/repeating-unit hint
- prefill/decode stage, chunk size or actual batch size
- device/rank selection
- hardware model and verified dense datatype peak
- explicit time/kernel-index window

If the user requests an uninterrupted best-effort run, continue with defensible
fallbacks and record uncertainty. Never invent layer IDs, source call sites,
shape operands, datatype, or MFU.

## Read references as needed

- Read [references/nsys-workflow.md](references/nsys-workflow.md) before exporting,
  inspecting SQLite, or selecting a repeating unit.
- Read [references/mapping-and-stats.md](references/mapping-and-stats.md) before
  mapping kernels, tracing Python calls, handling dual streams, or computing
  position-aware statistics.
- Read [references/output-spec.md](references/output-spec.md) before generating or
  validating edge cases and manifest invariants for the six-table package. The
  embedded contract above remains authoritative for filenames and columns.
- Read [references/runtime-evidence-and-mfu.md](references/runtime-evidence-and-mfu.md)
  before resolving launch/runtime/source conflicts or calculating MFU.
- Use [references/semantic_map.example.json](references/semantic_map.example.json)
  only as a schema example; replace its placeholder hardware peaks and rules with
  evidence from the current task.

## Workflow

### 1. Establish the model taxonomy

Before using config or source defaults, run:

```bash
python scripts/audit_runtime_evidence.py \
  --sqlite /path/to/report.sqlite \
  --launch /path/to/launch.sh \
  --source /path/to/source \
  --output /path/to/runtime_evidence.json
```

Read this audit and copy its resolved runtime fields/conflicts into the analysis
manifest. A launch flag is intent, not proof that the captured server retained
that branch. A supplied source tree is unverified when its commit cannot be
matched to captured `SGLANG_BUILD_COMMIT`.

Read architecture/design notes first. Extract:

- canonical submodule names, aliases and nesting
- forward execution order and layer variants
- repeating/composite layer pattern
- conditional branches, attention backend and MoE stream behavior

Then validate that taxonomy against source, config and deployment parameters.
Resolve conflicts in this order:

1. captured runtime evidence
2. config/design notes
3. launch intent
4. source defaults

Record conflicts instead of silently choosing.

Create the functional-module vocabulary anew for every task. Do not reuse the
previous run's semantic map as naming authority. A prior map may be consulted
only as a schema/example, and every retained label must be re-justified by the
current model evidence. Record `functional_module_taxonomy_source` in the
manifest with the design/config/source paths and any explicit user naming
choices.

### 2. Export and inspect the trace

For `.nsys-rep`, export once with:

```bash
nsys export --type sqlite --output /path/to/report.sqlite /path/to/report.nsys-rep
```

Do not overwrite an existing SQLite file unless refresh was requested. Verify
`CUPTI_ACTIVITY_KIND_KERNEL` and `StringIds`, inspect version-dependent columns,
and check NVTX/runtime/process tables. Use full demangled names when available.

### 3. Select one complete repeating unit

Within one representative device/process:

- exclude warmup unless explicitly targeted
- locate layer heads/tails using NVTX, source order and repeated kernel motifs
- find the smallest full sequence that repeats
- include every distinct sub-layer in a composite unit
- start at the first sub-layer head and end at the last sub-layer tail

For Transformer-like layers, verify the complete forward checklist through the
final FFN/MoE merge. Do not stop at the attention tail. Exact network layer IDs
require NVTX/runtime/config evidence or a verified full model-depth signature;
otherwise leave them blank and document the limitation.

If config/timeline shows an interleaved pattern such as
`KDA,KDA,KDA,MLA`, the default repeating unit contains the entire pattern.
Selecting one subtype is allowed only when the user explicitly requests that
subtype; label it as that subtype rather than a generic model layer.

### 4. Map every kernel

Assign every selected kernel to a canonical model-description module. Source
names and kernel signatures are evidence, not the primary taxonomy.

For each non-total row:

- provide a full Python call chain from layer `forward` to the deepest
  kernel-dispatching statement
- cite the deepest repository-relative `@ path:start-end`
- keep the leaf range narrow, normally under 15 lines
- explain both module attribution and call-site attribution in `mapping_reason`

One module occurrence must use exactly one CUDA stream. Split occurrence/group
identities when semantically related work runs on different streams. Keep groups
contiguous in the detailed CSV even when streams overlap.

Do not emit `unknown`, `misc` or `other` module labels in final artifacts. Use
the nearest defensible parent and record uncertainty in the manifest.

### 5. Generate the detailed origin data

Use `scripts/extract_layer_operator_csv.py` after the selected device, index
window and complete mapping files are known. Generate one combined origin
timeline for the entire repeating unit.

The total row must use SQLite/NVTX wall-span timestamps:

```text
(layer_end_ns - layer_start_ns) / 1000
```

Never sum kernel durations for the total because streams overlap and contain
gaps.

Compute stable min/max/avg by exact template position, not by kernel name. Match
the full raw repeating-unit template on each included rank, then project matches
into module-group CSV order. Use the selected sample as a clearly marked
single-sample fallback if no additional full match exists.

## Non-negotiable trace and mapping rules

### Repeating-unit boundary proof

A repeating unit may contain one layer type or several interleaved variants.
Whatever distinct layers appear inside the smallest repeating sequence must be
captured together in one raw window and one origin CSV.

Select boundaries in raw timeline order:

```text
head(sub-layer 1) -> complete forward of every contained sub-layer ->
tail(sub-layer N)
```

For every selected sub-layer, verify the active source-order checklist rather
than relying on a recognizable attention kernel. For Transformer-like layers,
check:

1. pre-attention/input normalization
2. Q/K/V or QKV preparation/projection
3. RoPE/cache/indexer and attention core
4. attention output projection and post-attention merge
5. pre-FFN normalization
6. dense MLP or MoE router/dispatch/experts/combine path
7. final post-FFN merge

Reject a window that ends at an attention/output marker if following kernels
continue the same layer's FFN/MoE path. For a composite unit, the last row must
be the true tail of the final contained variant. Record these manifest fields:

- `complete_sublayer_check`: `passed`, `failed`, or `partial_debug_window`
- `expected_tail_module`
- `actual_tail_module`
- `next_kernel_after_window`
- `boundary_evidence`

If completeness cannot be proved but output is still required, name and mark it
as a partial/debug window; do not present it as a complete layer.

### Exact layer-ID evidence

Assign `layer_id` only from:

- explicit NVTX/runtime layer metadata
- a user-provided target layer
- model config plus a complete model-depth or pipeline-stage signature sequence
  matched against the captured timeline

Do not derive an exact network layer ID from local repeating period, modulo
position, or assumed warmup length. When matching a complete sequence, record:

- expected signature source and configured layer count
- candidate pass start/end indices
- observed layer signatures
- selected candidate offsets
- `layer_id_mapping` for every selected sub-layer and total range

Leave IDs blank when evidence is insufficient.

### Recursive Python call-chain proof

For each module occurrence, start at the layer-level `forward` and recursively
open every called module/helper/backend until reaching the narrowest statement
that launches or dispatches CUDA/C++/Triton work.

The CSV value must contain every traversed frame:

```text
DecoderLayer.forward -> self.self_attn(...) -> Attention.forward ->
attn_backend.forward(...) -> Backend.forward ->
leaf_extension_call(...) @ repo/relative/deepest_file.py:120-128
```

Rules:

1. Never stop at the first model file when it only calls `self.<module>(...)`.
2. The `@ path:start-end` belongs to the deepest frame, not the layer file.
3. Prefer the exact call/selected branch, normally 1–15 lines.
4. A non-total range over 25 lines requires an `uncertain_mappings` manifest
   entry explaining why it cannot be narrowed.
5. Different semantic sites under the same coarse `module` may and should have
   different `python_function` values.
6. `mapping_reason` must explain both module placement and leaf call-site
   evidence using source order, timestamps, runtime branch, neighboring kernels,
   correlation metadata or kernel signature.

Reject blank call chains/reasons for non-total final rows.

### Complete module coverage and dual streams

`module_map` must cover every raw kernel index in the selected inclusive window,
using global indices or relative offsets. Missing entries are errors. Reject
final rows named `unknown/<index>`, `misc`, `other`, `self_attn/other` or
`mlp/other`.

When the same label appears more than once, use an explicit occurrence/group
identity. Each occurrence group must:

- contain kernels from exactly one CUDA stream
- remain contiguous in final origin CSV order
- sort internally by `start_ns` and raw index

Sort complete occurrence groups by their earliest `start_ns`. This grouping may
differ from global raw time order when streams overlap. Preserve raw order
separately for repeating-template matching and record raw index coverage in the
manifest.

For dual-stream MoE, do not collapse routed experts, shared experts, router/topk,
dispatch/combine and communication into a vague single block when design/source
evidence separates their streams or submodules.

### Position-aware stable statistics

Never aggregate stable row statistics by `operator_name` alone. Same-named GEMMs
or helper kernels often occur at different semantic sites.

Use the selected raw window as the exact template:

1. Let `unit_len = end_index - start_index + 1`.
2. Keep both each raw template offset/index and its final module-group CSV
   position.
3. Define row identity using CSV position, raw kernel index, module occurrence
   and full operator name.
4. Set `stable_start_ns` to the representative unit start unless the user
   provides another stable range.
5. By default include every model-rank device present in SQLite.
6. For each device and accepted CUDA-graph instance, match the full template at
   equivalent graph positions. For a structural composite cycle, accept every
   complete cycle occurrence whose layer-variant signature matches; for a
   fixed network layer, accept only that fixed graph position.
7. Accept a sample only when every operator name at every raw offset matches.
8. Project each accepted raw match into the same module-group CSV order.
9. Compute per-position min/max/avg/diff and each full-unit wall-span.

Write `position_operator_stats.json` with:

- identity scheme, `stable_start_ns` and `unit_len`
- included devices/ranks
- accepted full-template sample count
- per-device sample counts
- per-offset sample count/min/max/avg
- scope kind (`fixed_layer_position` or `structural_cycle`) and accepted
  occurrence count per graph instance

If no match beyond the representative window exists, copy representative values
into stable-stat columns and explicitly mark the result as a single-sample
heuristic. Exclude unmatched ranks rather than forcing partial matches.

The total row's representative/min/max/avg values come from unit wall-spans, not
summed kernels. Calculate:

```text
duration_diff_us = duration_max_us - duration_min_us
duration_avg_pct_of_total =
    duration_avg_us / __layer_total__.duration_avg_us * 100
```

Set the total percentage to `100.000%`.

### Mandatory origin manifest evidence

Alongside the detailed origin CSV and position-statistics sidecar, record:

- input report and exported SQLite paths
- description/design/source/config/deployment evidence paths
- model-description taxonomy and description-to-source alias table
- device/process and raw start/end kernel indices/timestamps
- repeating-unit sub-layer composition and discriminator signatures
- exact layer-ID evidence or explicit ambiguity
- module occurrence ordering and stream assignments
- stable aggregation devices/ranks and accepted sample counts
- whether names are demangled or short-name fallbacks
- `uncertain_mappings`
- completeness and boundary fields listed above

### 6. Build the six-table package

Create a model-specific ordered semantic map, then run:

```bash
python scripts/build_static_analysis_tables.py \
  --origin-csv /path/to/origin.csv \
  --output-dir /path/to/new-task-directory \
  --prefix glm52 \
  --semantic-map /path/to/model_semantic_map.json \
  --stage prefill \
  --chunk-size 64 \
  --hardware "Nvidia B200"
```

Keep these concepts distinct:

- `module`: fine-grained model-description path
- `功能模块`: broader Chinese analytical stage inferred from the real forward path

Do not mechanically copy or translate `module` into `功能模块`.
Consolidate the entire repeating unit systematically, not only one named
example. Use architecture-level stages and keep implementation-level
distinctions in `module`. In particular, all NSA Indexer internal steps use
`NSA Indexer` for this architecture; QKV normalization/projections normally use
one QKV stage; shared-expert projections normally use one shared-expert stage.
Name router/gating stages from the terminology used by the current model
description or user. For the GLM5.2 NSA/MoE mapping used in the bundled eval,
the requested label is `MoE Gate/TopK`; it is not a universal name.
Attention is the exception: keep preparation, core computation and output
projection as three distinct functional stages.

Shape and MFU apply only to defensible core GEMMs:

- derive prefill `M` from effective filled chunk/batch behavior
- derive decode `M` from the captured batch
- derive `N/K` from config and the selected source branch
- use `MFU = 2*M*N*K / (duration_seconds * verified_dense_peak_flops)`
- for mixed precision, declare operand `dtypes`, accumulator behavior and the
  Tensor Core `compute_dtype`; use the dense peak for that actual compute mode
- for grouped MoE GEMMs, state whether `M` is logical routed rows
  (`tokens × topk`) or physical padded rows; label logical-work MFU explicitly

Leave shape/MFU blank if any operand, branch, datatype, duration or dense peak is
uncertain. Never substitute sparse peak numbers.

### 7. Validate and report

Before handoff, verify:

- exact filenames and column order
- every selected kernel has one module and one analysis category
- core classification is applied per operator, not inherited from `功能模块`;
  every core row is a GEMM/BMM/matmul (including verified expert GEMM) or an
  actual Attention score/normalization/value-aggregation kernel
- no quant/dequant, LayerNorm/RMSNorm, RoPE, cache, TopK/router/dispatch,
  gather/scatter, permutation, copy/cast, activation or metadata/index kernel
  appears in the core-compute table
- row count equals selected kernel count plus one total row
- full call chains, deepest paths and non-empty mapping reasons
- module occurrence contiguity and one-stream rule
- position-aware stats and sidecar sample counts
- `duration_diff_us = max - min`
- total-row percentage is `100.000%`
- total duration is wall-span, not summed kernel time
- core/auxiliary and stage tables are duration-descending
- every human-facing `算子名称` is a compact CUDA kernel symbol rather than a
  semantic function label: remove return type, outer namespace and runtime
  arguments; retain a short distinguishing template payload and abbreviate only
  oversized payloads as `<…>`
- cohesive stages are not fragmented into micro-operation labels; specifically,
  validate every functional module in the layer, with NSA Indexer, QKV,
  shared-expert and Gate/TopK internals aggregated by default
- do not carry functional-module names from a previous model into a new model;
  derive a fresh taxonomy from its description, config, active source branch
  and user terminology
- when several accurate names are possible, prefer the model documentation's
  term, then an explicit user-provided term, then a concise functional name
- Name stages by model function, not by backend mechanism or source class:
  use `Attention 核心计算` rather than `NSA 稀疏注意力`, and `MoE 核心计算`
  rather than `MoE Routed Experts`. Keep `sparse`, `FlashMLA`, `routed` and
  similar implementation details in `module`, operator name or introduction.
- Keep `Attention 核心计算` narrow: only kernels that perform attention
  scores/normalization/value aggregation belong there. Absorbed Q projection,
  RoPE, KV-cache packing/update, collectives and backend input preparation
  belong to `Attention 计算准备`; value reconstruction, output quantization and
  O projection belong to `Attention 输出投影`.
- Validate that `Attention 核心计算` contains no cache, communication, RoPE,
  quantization, reconstruction or linear-output kernels.
- every non-empty MFU is at most 100% and its manifest records shape, all
  participating dtypes, compute mode, selected dense per-GPU peak and source
- every eligible core GEMM with verified shape/hardware/compute mode has MFU;
  a blank cell in that situation is a validation failure
- all three classification rows exist, including zero-count classes
- origin non-total row count is exactly `end_index - start_index + 1`
- every non-total row has non-empty stable-stat columns unless the manifest marks
  a single-sample fallback
- repeated operator names at different template positions retain independent
  statistics
- the stats sidecar includes per-device and per-offset sample counts
- manifest completeness fields prove the raw window begins at the first
  sub-layer head and ends at the final sub-layer tail
- the manifest records the current model's functional-module taxonomy source;
  no label is inherited solely from a previous model or eval
- the runtime audit exists and every launch/runtime/source conflict is resolved
  in favor of captured evidence or explicitly blocks source-derived claims
- the selected unit contains every layer variant in the smallest repeating
  pattern unless the user explicitly requested one subtype
- the frontend `analysis.json`, when produced, preserves six-table category
  membership, durations, stable sample count, devices and repeating-unit
  evidence exactly
- never call zero-kernel GPU idle time a `CPU gap`. Attribute CPU delay only
  when CUDA Runtime launch timestamps prove the next work was submitted late;
  otherwise label it GPU idle/queue/dependency gap

Run the deterministic package validator before handoff:

```bash
python scripts/validate_analysis_package.py /path/to/csv-package \
  --prefix model_prefix \
  --analysis-json /path/to/analysis.json \
  --output /path/to/validation_report.json
```

Use `duration_avg_us` for downstream tables when position-aware samples exist;
otherwise use `duration_us` and flag the fallback. Percentages use the repeating
unit wall-span. Overlapping module/category kernel sums may exceed 100%; state
that fact and do not normalize it away.

Report input/export/output paths, repeating-unit composition and boundaries,
device/ranks, layer-ID evidence, stable sample counts, model taxonomy/aliases,
shape/MFU evidence, and uncertain mappings. Keep the response concise; the CSV
files and manifests are the primary artifacts.
