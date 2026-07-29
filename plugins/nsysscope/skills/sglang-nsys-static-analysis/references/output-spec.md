# Six-table output specification

## Contents

1. Files and source of truth
2. Columns
3. Semantic map and MFU
4. Validation invariants

## Files and source of truth

Always create:

- `<prefix>_operator_origin_table.csv`
- `<prefix>_opreator_table.csv`
- `<prefix>_core_compute_table.csv`
- `<prefix>_auxiliary_operator_table.csv`
- `<prefix>_op_classification_table.csv`
- `<prefix>_stage_table.csv`
- `<prefix>_analysis_manifest.json`

Use `duration_avg_us` when stable samples exist; otherwise use `duration_us` and
record the fallback. All percentages use the repeating-unit wall-span.

## Columns

Origin:

```text
序号,module,operator_name,duration_us,start_ns,end_ns,device,stream,layer_id,
duration_min_us,duration_max_us,duration_diff_us,duration_avg_us,
duration_avg_pct_of_total,python_function,function_introduction,mapping_reason
```

Operator overview:

```text
序号,功能模块,算子名称,算子耗时(us),算子耗时占比(%),shape,mfu,
模块耗时(us),模块耗时占比(%),python_function,功能介绍
```

Core compute:

```text
序号,功能模块,module,算子名称,算子耗时(us),算子耗时占比(%),
shape,mfu,python_function,功能介绍
```

Auxiliary:

```text
序号,功能模块,算子名称,算子耗时(us),算子耗时占比(%),
python_function,功能介绍
```

Classification:

```text
序号,算子类型,算子数量,总耗时(us),耗时占比(%)
```

Always emit `核心计算`, `通信`, and `辅助算子`, including zero-count categories.

Stage:

```text
序号,功能模块,模块耗时(us),模块耗时占比(%),功能介绍
```

Origin keeps full demangled names. Other tables use compact CUDA kernel names:
remove return type, outer namespace and runtime arguments, retain short
distinguishing template parameters, and abbreviate only oversized template
payloads as `<…>`. A semantic-map `operator_name` override must itself be a
compact kernel symbol, never a functional description. Put descriptions such as
`Q-B 投影` or `输入量化` in `功能介绍`.

## Semantic map and MFU

Build ordered model-aware rules with optional `module_regex`,
`operator_regex`, `functional_module`, `operator_name`, `category`,
`core_kind`, `shape`, `dtypes`, and `introduction`. Every rule with
`category: core` must declare `core_kind: gemm` or `core_kind: attention`.

`module` is the fine-grained architecture label. `功能模块` is a broader
architecture stage inferred from the real forward path; never copy it
mechanically. Merge all NSA Indexer internals into `NSA Indexer` unless finer
granularity was explicitly requested. Apply that review to the whole layer:
merge QKV preparation and cohesive shared-expert internals. Derive router,
TopK and dispatch naming from the current model taxonomy; `MoE Gate/TopK` is
appropriate for the GLM5.2 eval and user terminology, but is not a global
constant. A typical Transformer/MoE repeating unit normally has about 5–12
functional modules; justify model-specific exceptions in the manifest.

Use stable functional names rather than implementation terminology. The
attention execution stage is `Attention 核心计算` even when its backend is NSA,
sparse attention or FlashMLA. Expert GEMM execution is `MoE 核心计算`, not
`MoE Routed Experts`; record routed/shared/backend details in finer fields.
Treat these as naming guidance for comparable Transformer/MoE paths, not as a
fixed vocabulary for unrelated architectures.

Do not use `Attention 核心计算` as an umbrella for the whole attention path:

- `Attention 计算准备`: absorbed Q projection, RoPE, KV-cache packing/update,
  collectives and backend input preparation.
- `Attention 核心计算`: only attention scores, normalization and value
  aggregation.
- `Attention 输出投影`: value reconstruction, output quantization and O
  projection.

Only core GEMMs may receive shape/MFU. Require verified M/N/K, active branch,
all participating datatypes, duration and dense theoretical peak:

```text
MFU = 2*M*N*K / (duration_seconds * dense_peak_flops)
```

For mixed precision use the lowest verified dense peak among activation,
weight and accumulation paths; weight dtype alone is insufficient. Reject MFU
above 100% and fix the evidence. Leave unavailable values blank. Hardware peaks in
`semantic_map.example.json` are placeholders, not authoritative specifications.

Classify operators independently of their owning functional module. Core
compute is a strict allow-list containing GEMM/BMM/matmul (including verified
fused/grouped expert GEMMs) and actual Attention
score/normalization/value-aggregation kernels. Communication is separate.
Everything else is auxiliary. In particular, quant/dequant, LayerNorm/RMSNorm,
RoPE, cache management, TopK/router/dispatch, gather/scatter, permutations,
copies/casts, activations and metadata/index transforms remain auxiliary even
inside Attention, projection, Indexer or MoE stages. Reject a core table
containing `per_token_group_quant_8bit_kernel`, `generalLayerNorm`, or another
operator from those excluded families.

## Validation invariants

- exact filenames and headers
- valid CSV quoting for demangled templates
- origin row coverage equals selected kernels plus one total row
- every non-total row occurs once in overview and exactly one category
- every core row is justified as `gemm` or `attention`; core category is never
  inherited from the broader `功能模块`
- excluded auxiliary families never appear in the core-compute table
- core/auxiliary rows sorted by duration descending
- stage rows sorted by aggregate duration descending
- `duration_diff_us = duration_max_us - duration_min_us`
- total average percentage is `100.000%`
- module/category percentages use wall-span and are not normalized
- full Python call chains and evidence-backed mapping reasons
- no final unknown/misc/other modules
- manifest records input/output paths, stage, hardware, semantic map,
  denominator, fallback/uncertainty, repeating-unit boundaries and sample scope
