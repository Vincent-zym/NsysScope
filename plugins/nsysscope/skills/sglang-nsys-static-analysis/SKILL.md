---
name: sglang-nsys-static-analysis
description: "Analyze an SGLang Nsight Systems .nsys-rep or .sqlite report, derive a model-specific architecture taxonomy from design/config/runtime/source evidence, preserve heterogeneous layer or block variants, map CUDA kernels to Python dispatch sites, and generate validated six-table timing/MFU packages. Use for repeating-unit timing, module attribution, KDA/MLA/NSA/MoE or other mixed architectures, operator classification, shapes, MFU, and NsysScope-compatible output. Do not use for Nsight Compute .ncu-rep bottleneck diagnosis."
---

# SGLang Nsight Systems static analysis

## Outcome

Analyze the smallest complete repeating sequence actually present in the trace.
Generate:

1. `<prefix>_operator_origin_table.csv`
2. `<prefix>_opreator_table.csv`
3. `<prefix>_core_compute_table.csv`
4. `<prefix>_auxiliary_operator_table.csv`
5. `<prefix>_op_classification_table.csv`
6. `<prefix>_stage_table.csv`
7. `<prefix>_architecture_taxonomy.json`
8. `<prefix>_analysis_manifest.json`
9. a position-aware statistics sidecar
10. `validation_report.json`

Treat the trace as timing authority. Treat current-model design/config/runtime
and verified source as semantic authority. Never use a previous model's labels
as naming authority.

## Required evidence

Collect:

- `.nsys-rep` or exported `.sqlite`
- model design notes when available
- model config
- deployment command/YAML
- relevant model and backend source

Useful evidence includes the requested subtype, hardware, batch/chunk size,
rank/device and explicit trace window.

Resolve conflicts in this order:

1. captured runtime and trace
2. model design/config
3. verified source branch
4. launch intent

Record conflicts. Do not infer runtime branches from source defaults.

## Read references

- Read [references/nsys-workflow.md](references/nsys-workflow.md) before
  export, SQLite inspection or repeating-unit selection.
- Read [references/architecture-taxonomy.md](references/architecture-taxonomy.md)
  before naming modules or mapping a heterogeneous/composite unit.
- Read [references/mapping-and-stats.md](references/mapping-and-stats.md) before
  kernel mapping, call-chain tracing, dual-stream handling or stable statistics.
- Read [references/runtime-evidence-and-mfu.md](references/runtime-evidence-and-mfu.md)
  before resolving runtime/source conflicts or calculating MFU.
- Read [references/output-spec.md](references/output-spec.md) before generating
  or validating the package.

## Workflow

### 1. Audit runtime evidence

Run:

```bash
python scripts/audit_runtime_evidence.py \
  --sqlite /path/report.sqlite \
  --launch /path/launch.sh \
  --source /path/source \
  --output /path/runtime_evidence.json
```

Copy resolved fields and conflicts into the manifest. A supplied source tree is
unverified when its commit cannot be matched to captured build evidence.

### 2. Establish the architecture taxonomy

Read model design/config before examining kernel names. Extract:

- canonical component names and nesting
- ordered forward path
- conditional branches
- structural unit variants
- repeating/composite pattern
- shared versus variant-specific modules
- stream/overlap design
- fused operations with multiple logical owners

Create `<prefix>_architecture_taxonomy.json` using the current task's evidence.
Validate it:

```bash
python scripts/validate_architecture_taxonomy.py \
  /path/<prefix>_architecture_taxonomy.json
```

For every repeating-unit position, record:

```text
unit_position, unit_id, unit_variant, layer_id/module discriminator
```

For every variant, record source evidence, trace/config discriminators and its
ordered functional modules.

Keep the two semantic levels distinct: `module` is the fine-grained
source/execution attribution, while `功能模块` is the architecture rollup.
Default to 5–8 ordered functional modules per variant. Merge adjacent
projection/norm/cache/gate steps into attention preparation, backend internals
into the variant core, projection/collective/residual work into attention
output, router/top-k/pack/quantize/dispatch into MoE input and routing, expert
math into MoE experts, and expert finalize/reconstruction/communication into
MoE output. More than eight stages requires a current-model
`granularity_exception` in the taxonomy and must represent independently
actionable architecture boundaries, not implementation substeps.

Keep a variant-specific core stage semantically narrow. If the trace exposes a
single fused attention/state-space core kernel, map only that kernel (or the
verified core kernels) to the core stage. Do not absorb input projections,
cache preparation, output postprocessing, LSE/value reconstruction, gates or
collectives into `MLA 核心计算` merely because they are adjacent in the
forward call. Put those operations in explicit preparation/output-rebuild or
communication stages while preserving their fine-grained `module` labels.

For a pattern such as `KDA,KDA,KDA,MLA`, preserve all four positions and both
variants. Do not present `cycle_duration / 4` as the duration of either subtype.
Report the cycle, every position and variant summaries separately.

If one CUDA kernel fuses several logical modules, assign it to one named fusion
group, associate that group with one declared coarse `functional_module`, list
its logical owners and use `attribution_policy: indivisible`. Never invent a
fractional timing split.

### 3. Export and inspect the trace

For `.nsys-rep`, export once:

```bash
nsys export --type sqlite --output /path/report.sqlite /path/report.nsys-rep
```

Do not overwrite an existing export unless refresh was requested. Inspect
kernel, string, NVTX, runtime, process and graph tables using their actual
version-dependent columns. Prefer full demangled symbols.

### 4. Select the complete repeating sequence

Within one representative device/process:

- exclude warmup unless requested;
- locate heads/tails from NVTX, model order and repeating motifs;
- select the smallest full sequence that repeats;
- include every distinct variant in a composite pattern;
- include attention/state-space, output, FFN/MoE and final merge tails.

Verify each selected structural unit through:

1. input/pre-attention aggregation or norm
2. projections and preparation
3. attention/state-space core
4. output projection and merge
5. pre-FFN/MoE preparation
6. dense FFN or router/dispatch/experts/shared path
7. final post-FFN merge

Assign exact network layer IDs only from explicit metadata or a verified full
model-depth signature. Otherwise leave them blank.

### 5. Map every kernel

Assign every selected kernel:

- one structural position and variant
- one fine-grained model module
- one broader current-model functional module
- one category: core, communication or auxiliary
- one module occurrence and CUDA stream
- one recursive Python call chain
- one evidence-backed mapping reason

Follow wrappers to the narrowest Python statement that launches CUDA/C++/Triton
work. Cite the deepest repository-relative path and normally fewer than 15
lines. Do not emit `unknown`, `misc` or `other`.

Keep module occurrences contiguous and one-stream-only. Split occurrences when
related work uses separate streams.

Do not map fine-grained `module` labels one-to-one onto `功能模块`. A useful
functional module normally contains several related fine modules; keep the
fine detail in the `module` column and mapping evidence.

### 6. Compute position-aware statistics

Use the complete selected sequence as an exact template. Match every operator
position and full symbol on every accepted device and graph instance. Never
aggregate by operator name alone.

Record:

- raw template offset and final CSV position
- structural position/variant
- included devices
- accepted full-template count
- per-device counts
- per-position min/max/avg
- whole-cycle wall spans
- per-structural-position wall spans

Use wall-span for total percentages. Do not use summed kernel durations as the
cycle or layer total.

### 7. Build the package

Create an ordered task-local semantic map and run:

```bash
python scripts/build_static_analysis_tables.py \
  --origin-csv /path/origin.csv \
  --output-dir /path/result \
  --prefix model \
  --semantic-map /path/model_semantic_map.json \
  --taxonomy /path/model_architecture_taxonomy.json \
  --stage decode \
  --batch-size 20 \
  --hardware "Nvidia B300"
```

Aggregate functional modules by:

```text
(unit_position, unit_id, unit_variant, functional_module)
```

Never aggregate only by functional-module label in a composite unit.

For the secondary stage/core/auxiliary views, emit a separate pattern-level
rollup when the selected repeating unit has multiple positions. Group
identical `功能模块` names across layers for the functional-module summary and
charts, while retaining the position-aware rows and layer/unit fields as audit
detail. Functional-module selection keys in the pattern view must not include
layer IDs; operator drill-down can still filter by the original layer.

Keep model function separate from operator category. A quantization or norm
kernel inside an attention stage remains auxiliary. Core compute is restricted
to GEMM/BMM/matmul, verified grouped expert GEMMs and actual
attention/state-update score/normalization/value-aggregation kernels.

### 8. Compute shapes and MFU

Compute GEMM MFU only when M/N/K, active branch, operand formats, Tensor Core
compute dtype, duration and dense per-GPU hardware peak are verified:

```text
MFU = 2*M*N*K / (duration_seconds * verified_dense_peak_flops)
```

Use `references/hardware-peaks.json`. Record accumulator behavior separately
from compute dtype. For grouped MoE, identify logical versus padded routed rows.
Use the physical padded row count only when the trace exposes it; otherwise
label the result as logical-row MFU and do not silently substitute a padding
estimate. Low MFU for small decode projections is a valid measured result and
must not be inflated by changing shapes or peaks. Attention core kernels that
are not GEMMs should leave shape/MFU blank. Reject MFU above 100%. Leave
shape/MFU blank when evidence is insufficient.

### 9. Validate

Run:

```bash
python scripts/validate_analysis_package.py \
  /path/result \
  --prefix model \
  --taxonomy /path/model_architecture_taxonomy.json \
  --analysis-json /path/analysis.json \
  --output /path/validation_report.json
```

Fail when:

- any selected kernel is missing or duplicated;
- any CSV lacks its correctly calculated final total row;
- the operator overview omits origin `module` immediately before `算子名称`;
- category rules are inconsistent;
- a composite row lacks position/id/variant;
- a declared variant or position disappears from final tables;
- a variant's required functional modules are missing;
- a heterogeneous cycle is labeled generic single-layer duration;
- a fused kernel is split without trace evidence;
- runtime, shape, MFU or frontend parity is inconsistent.

## Numerical and presentation rules

- Preserve the captured timeline as-is.
- Use position-aware averages for operator work.
- Use wall-span for cycle/unit denominators.
- Keep kernel-duration sum, representative interval union and representative
  wall span as separate module metrics.
- Report overlap explicitly; never normalize overlapping stages to 100%.
- Keep full demangled symbols only in origin data.
- Use compact CUDA leaf symbols in human-facing tables.
- Put semantic meanings in introductions, not operator-name cells.
- End every CSV with one total row. Use accumulated operator work for
  operator/category/stage totals and repeating-unit wall span for the origin
  total; allow accumulated percentages above 100% under overlap.
- Preserve legacy package import, but require the taxonomy contract for every
  new composite/heterogeneous analysis.
