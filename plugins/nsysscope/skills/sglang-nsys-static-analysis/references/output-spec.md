# Six-table output specification

## Contents

1. Required artifacts
2. Columns
3. Total rows
4. Taxonomy and semantic mapping
5. MFU and operator categories
6. Validation invariants

## Required artifacts

Always create:

- `<prefix>_operator_origin_table.csv`
- `<prefix>_opreator_table.csv`
- `<prefix>_core_compute_table.csv`
- `<prefix>_auxiliary_operator_table.csv`
- `<prefix>_op_classification_table.csv`
- `<prefix>_stage_table.csv`
- `<prefix>_analysis_manifest.json`
- `<prefix>_architecture_taxonomy.json`

Use `duration_avg_us` when stable samples exist; otherwise use `duration_us` and
record the fallback. All percentages use the repeating-unit wall-span. Preserve
the structural position and variant in every composite-unit row.

## Columns

Origin:

```text
序号,module,operator_name,duration_us,start_ns,end_ns,device,stream,layer_id,
unit_position,unit_id,unit_variant,duration_min_us,duration_max_us,
duration_diff_us,duration_avg_us,duration_avg_pct_of_total,python_function,
function_introduction,mapping_reason
```

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
算子耗时占比(%),shape,mfu,python_function,功能介绍
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

Legacy six-table packages without taxonomy remain importable, but new composite
analysis must satisfy this contract.
