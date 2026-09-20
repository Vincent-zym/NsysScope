---
name: sglang-nsys-static-analysis
description: "Analyze an SGLang Nsight Systems .nsys-rep or .sqlite report, derive a model-specific architecture taxonomy from design/config/runtime/source evidence, preserve heterogeneous layer or block variants, map CUDA kernels to Python dispatch sites, and generate validated seven-table timing/MFU packages. Use for repeating-unit timing, module attribution, KDA/MLA/NSA/MoE or other mixed architectures, operator classification, shapes, MFU, and NsysScope-compatible output. Do not use for Nsight Compute .ncu-rep bottleneck diagnosis."
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
7. `<prefix>_forward_pipeline_table.csv` (required; see below)
8. `<prefix>_architecture_taxonomy.json`
9. `<prefix>_analysis_manifest.json`
10. a position-aware statistics sidecar
11. `validation_report.json`
12. `final_report.md` (the human-readable report; see step 11)
13. `analysis.json` (the stable frontend contract; see step 8)
14. `xlsx/` (one workbook per table; written by step 12)

The result directory is a package with a fixed shape, not a pile of files. Author
the analysis flat if that is easier, then run step 12, which is what puts every
artifact in its place and writes the manifest:

```text
result/
├── analysis.json                 # frontend contract
├── final_report.md               # the report a human reads
├── nsysscope-package.json        # manifest: prefix, table list, directories
├── csv/                          # the six or seven tables, nothing else
├── xlsx/                         # one workbook per table
├── metadata/                     # taxonomy, manifest, semantic map,
│                                 # statistics sidecar, validation report,
│                                 # and any scratch CSV the analysis produced
├── trace/                        # the exported .sqlite
└── logs/                         # only when a job produced a log
```

Never hand-place files to imitate this: run the packager, so a Skill run by hand
and a run driven by the NsysScope service end in the same directory.

All seven tables are required. The seventh (the forward-pipeline table) relates
the measured repeating unit to a whole forward step, so the package can answer
what fraction of a step the unit is. Generate and validate it as part of every
package; `scripts/build_forward_pipeline_table.py` produces it from the trace
after the six tables exist (whole-step, region-agnostic — for a multi-region
taxonomy it counts variant layers across all regions).

When generating the seventh table, always pass the job's own declarations --
`--model-config` (config.json), `--launch`, `--runtime-evidence` and `--stage`. They
decide the step's layer counts and whether a draft forward runs; the trace only has to
reproduce them, and a contradiction is a hard failure. Inferring both from the
timeline's shape is the fallback, not the plan (see references/output-spec.md,
"Declaration first, trace second").

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
- Use [references/functional-modules.json](references/functional-modules.json) as
  the `功能模块` vocabulary; it exists so two analyses of one model stay comparable.
  Its `reuse` section says which stages a new model must reuse verbatim and how to
  add a block for a new attention mechanism. The validator checks five things
  against it -- unknown names, deprecated names, per-variant stage count against
  the 5-8 window, stages under ~2% of their block, and whether the emitted stage
  set matches the declared family's `typical_layer` -- so record the family id in
  `functional_module_policy` and match its stage list, or state why you deviated.
- Read [references/mapping-and-stats.md](references/mapping-and-stats.md) before
  kernel mapping, call-chain tracing, dual-stream handling or stable statistics.
- Read [references/runtime-evidence-and-mfu.md](references/runtime-evidence-and-mfu.md)
  before resolving runtime/source conflicts or calculating MFU.
- Read [references/output-spec.md](references/output-spec.md) before generating
  or validating the package.

## Filesystem guardrails

- NEVER run a whole-filesystem search. No `find /`, `grep -r /`, `ls -R /`, or any
  scan rooted at `/`, `/root`, `/home`, `$HOME` or another huge tree. On a shared
  server `/` spans Docker overlay layers and network mounts, so such a scan blocks
  in uninterruptible I/O for many minutes and stalls the whole job. This is a real
  incident: an agent's `find / -name "*semantic_map*.json"` sat in D-state for
  20+ minutes and the job never recovered.
- Scope every file search to a known directory: the job/package dir, the skill dir
  (`scripts/`, `references/`), or a path named in the task. Prefer `ls DIR`,
  `find DIR -maxdepth N`, or a grep with an explicit directory over an unbounded
  walk, and always bound the depth.
- You do not need to hunt the filesystem for example packages or origin CSVs.
  Everything required is the trace, the model config/source named in the task, and
  this skill's own `references/`. Build the origin CSV from the trace, never from a
  found example.

## Workflow

### 1. Recover the launch command

The command that produced the capture is inside the trace -- nsys stores the argv
it wrapped in `META_DATA_CAPTURE`, one row per argument -- so it is read from
there instead of being asked for as a deployment script:

```bash
python scripts/extract_launch_command.py /path/report.sqlite \
  --output /path/metadata/launch_command.txt
```

Every later step takes this file as `--launch`. It beats a supplied script on
both counts: shell variables and `$(( ))` arithmetic are already resolved, and it
cannot disagree with what actually ran. Verified on 86/86 SQLite traces across
nsys export versions 2025.1 -> 2026.4. A trace with no `PROCESS_*:COMMAND` is a
hard failure -- the declared parallelism, chunk size and speculative-decoding
flags all come from here, and inferring them from the kernels would produce a
confident wrong answer instead of an error. Note this is *declared* intent: the
`server_args=ServerArgs(...)` line that step 2 reads out of the trace is what the
runtime actually resolved each flag to, and it still wins on conflict.

### 2. Audit runtime evidence

Run:

```bash
python scripts/audit_runtime_evidence.py \
  --sqlite /path/report.sqlite \
  --launch /path/metadata/launch_command.txt \
  --source /path/source \
  --output /path/runtime_evidence.json
```

Copy resolved fields and conflicts into the manifest. A supplied source tree is
unverified when its commit cannot be matched to captured build evidence.

An unverified or mismatched commit downgrades exactly one thing: source-derived
**default values and branch assertions** may no longer be quoted as runtime truth.
It is not a licence to drop evidence. Call chains with `file:line`, dispatch code
snippets, Chinese functional descriptions and GEMM shape/MFU/MBU remain required,
annotated `(source commit unverified)` where a claim depends on the source. A
package whose evidence columns hold boilerplate is rejected by
`scripts/validate_analysis_package.py`, not accepted with a disclaimer.

### 3. Establish the architecture taxonomy

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

Also fill the machine-readable marker fields whenever the evidence exists:
`repeating_unit.boundary_evidence.layer_start_kernel` and each variant's
`trace_marker_kernels` (bare kernel shortName substrings from this capture,
positive markers only). The forward-pipeline table cuts a step into layers with
them; without them it parses prose and may refuse to publish. See
references/architecture-taxonomy.md ("Machine-readable trace markers").

If the model is not one periodic region but several periodic regions stacked
back to back with **different periods** — e.g. DeepSeek-V4.1-Flash's encoder
region (NSA index recomputed every 6 layers: 1 source + 5 reuse) followed by its
decoder region (recomputed every 4 layers: 1 reindex + 3 reuse) — do not force
them into one `repeating_unit`, and do not treat one as the other's head/tail.
Use the schema-1.1 `regions` form: bump `schema_version` to `1.1` and declare a
`regions` array, each region carrying its own `layer_range` and `repeating_unit`
(positions contiguous from 1 within that region), with `variants` staying one
global list. Then in step 5 select and map **one representative period per
region** so the origin CSV covers every region; downstream the tables, the
report and the package gate produce one §2.x block per region, each with its own
denominator. See references/architecture-taxonomy.md ("Multiple periodic
regions"). A model with a single periodic region keeps the single
`repeating_unit` form unchanged.

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

### 4. Export and inspect the trace

For `.nsys-rep`, export once:

```bash
nsys export --type sqlite --output /path/report.sqlite /path/report.nsys-rep
```

Do not overwrite an existing export unless refresh was requested. Inspect
kernel, string, NVTX, runtime, process and graph tables using their actual
version-dependent columns. Prefer full demangled symbols.

### 5. Select the complete repeating sequence

Within one representative device/process:

- exclude warmup unless requested;
- locate heads/tails from NVTX, model order and repeating motifs;
- select the smallest full sequence that repeats;
- include every distinct variant in a composite pattern;
- include attention/state-space, output, FFN/MoE and final merge tails.
- for a multi-region taxonomy (step 3, schema 1.1): repeat this selection **once
  per region** — pick each region's own smallest repeating period and map its
  layers — so the origin CSV carries one representative period for every region.
  Each row's `layer_id` decides its region via the region's `layer_range`.

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

When a dispatch-site cache was supplied, the same pre-pass also wrote a layer
segmentation next to it (`final_report.md`, `rankings/slowest_layers.csv`,
`mappings/kernel_to_layer.csv`). Read it first as a starting hypothesis for the
anchor kernel and layer count, then confirm it against the nsys SQLite -- it comes
from a different capture, so it saves search, not verification. See
references/mapping-and-stats.md ("Layer-segmentation priors from the same
pre-pass").

### 6. Map every kernel

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

Both name columns have to be reproducible, because two analyses of the same model
are compared by name. Two runs of this skill on one GLM5.2 capture produced 8
functional modules each and agreed on 2 names, and 62 versus 56 distinct `module`
values with 7 in common -- which makes per-module timing impossible to diff.

- `module`: lift every segment from the source. The leaf is an `nn.Module`
  attribute or the dispatching function, not a phrase you compose. If the leaf's
  words appear nowhere in the row's call chain, dispatch statement or kernel name,
  it was invented -- pick the source symbol instead.
- `功能模块`: choose from references/functional-modules.json for the matching
  architecture family. Only invent a name when no canonical entry fits, and then
  record it plus the reason in the taxonomy's `functional_module_policy`.

`validate_analysis_package.py` reports both as warnings, so drift shows up in the
validation report without failing an otherwise correct package.

When the task supplies a pre-resolved dispatch-site cache (produced by
`reconstruct-profiler-call-tree` from a torch profiler trace of the same
deployment), consult it per kernel symbol before searching the source tree, and
read references/mapping-and-stats.md ("Pre-resolved dispatch-site cache") for the
per-entry trust levels. It is a lookup table, not an authority: it never
overrides the trace for timing, and kernels missing from it still require the
normal search.

Once the repeating unit is fixed, get the mechanical half of this step written for
you instead of transcribing it:

```bash
python scripts/build_slot_skeleton.py \
  --sqlite <trace.sqlite> --cache <dispatch_site_cache_with_snippets.json> \
  --anchor <layer_start_kernel> --anchor-confirm <once_per_layer_kernel> \
  --layers-per-unit <N> --output work/slot_skeleton.json
```

It selects a representative unit deterministically and prefills, per slot, the
launch geometry, duration, template integers and -- from the cache -- the call
chain, dispatch statement and `line_drift` state. `module`, `functional_module`,
`category`, `shape`, `introduction` and `mapping_reason` stay empty because they
are the analysis. Check `coverage` and each symbol's `cache_match` before citing
anything: see references/mapping-and-stats.md ("Prefilled slot skeleton").

Keep module occurrences contiguous and one-stream-only. Split occurrences when
related work uses separate streams.

Do not map fine-grained `module` labels one-to-one onto `功能模块`. A useful
functional module normally contains several related fine modules; keep the
fine detail in the `module` column and mapping evidence.

### 7. Compute position-aware statistics

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

### 8. Build the package

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

Then build the frontend contract, which step 10 validates the tables against:

```bash
python scripts/build_analysis_json.py /path/result /path/result/analysis.json \
  --prefix model --model GLM5.2 --stage decode --hardware "Nvidia B300"
```

`analysis.json` is the stable frontend contract, so generate it before validating.
It is a derived view: never hand-edit it, fix the table it came from and
regenerate. The workbooks come later, in step 12, from the tables' final location.

### 9. Compute shapes and MFU

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

### 10. Validate

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

Passing `--analysis-json` also checks that the document is *shaped* the way the
dashboard needs -- schema version, operator categories, `operatorCount`, device
and sample scope, and, for a heterogeneous cycle, that every operator, stage and
unit keeps its structural identity. Those checks are also a script of their own:

```bash
python scripts/validate_frontend_contract.py /path/result/analysis.json
```

Nobody has to remember to run it: step 12 runs it, and repairs what it can, before
writing the manifest. Run it directly only to see the violations while fixing them
-- it exits non-zero and names every one. The NsysScope service runs the same
script as its own last gate, so a package that passes here will not be rejected by
the tool.

### 11. Write the analysis report

Every task ends with `final_report.md` in the result directory -- the one
deliverable a human reads instead of the tables. Generate it after the seventh
table exists (the forward-link tables come from it):

```bash
python scripts/build_final_report.py /path/result --prefix model
```

The script fills every fact and every table from the package's own data
(model/hardware/stage, engine parallelism, chunk/batch size, the forward step
split, target's and draft's children, functional modules over the repeating
unit, operator categories, a kernel-level ranking table, and every core-compute
operator of the pattern in execution order with shape/MFU/MBU), and leaves
`<!-- TODO ... -->`
markers only for what it could not read structurally: 代码版本 (leave as `—`
unless told otherwise), ctx len/MTP or TP/EP/PP when the manifest lacks them,
the model-structure line, and 分析思路's one-sentence selection rationale.
Replace every marker and keep the generated numbers -- if a number looks
wrong, fix the table it came from, not the report.

After the markers are filled, check the result mechanically:

```bash
python scripts/build_final_report.py /path/result --prefix model --check
```

`--check` regenerates the skeleton and requires every generated line to be
byte-identical, so a hand-added table row or a rewritten generated note fails
here instead of being spotted by a reader; it also measures every filled slot
and every prose line (中文字数 + 每段英文/数字算 1 个词) against the report's
length limits (正文 75, 第1节配置字段 35). Fix the report until it passes. If a
generated line is wrong for this package, change `build_final_report.py` and
regenerate -- do not hand-patch the line, or the next package repeats the bug.

This is a pure timeline/operator-timing report: no 结论 section, no 潜在优化点,
no prose interpretation. Every line is a fact with a number behind it. 分析思路
is the one sentence that is not a table -- state why this repeating unit was
selected and its wall time, nothing more: no sampling min/max, no per-variant
单层耗时 that 2.2.1 already carries. See references/final-report-format.md
for the section
layout, the paste-fidelity rules the HTML tables depend on, and the restraint
this implies, and references/final_report.example.md for a complete filled-in
report (GLM5.2 prefill) to match.

Only when the job supplied a 如流知识库 page, mirror the finished report onto it:

```bash
python scripts/publish_report_to_ku.py /path/result --url <知识库文档链接>
```

The report in the result package stays the deliverable -- this reads it
read-only and never rewrites it, so a failed publish leaves the analysis
output untouched. Skip the step entirely when no page was supplied; do not
invent one. It needs the `ku-doc-manage` Skill's CLI (`--ku-bin`, or
`KU_DOC_MANAGE_DIR`/`COMATE_SKILL_DIR`) and a username
(`--username`/`BAIDU_CC_USERNAME`), refuses a report that still has
`<!-- TODO -->` markers, and reads the page back afterwards to check the table,
cell and tint counts against the source rather than trusting the API's 200.

### 12. Finalize the package layout

Last step, once the tables, `analysis.json`, `validation_report.json` and
`final_report.md` are all written:

```bash
python scripts/finalize_package.py /path/result --prefix model \
  --trace /path/exported.sqlite
```

It moves the six or seven canonical tables into `csv/`, every sidecar and every
scratch CSV into `metadata/`, the trace into `trace/`, writes one workbook per
table into `xlsx/`, and emits `nsysscope-package.json` with the prefix, the table
list and the directory names. `--trace` is optional; omit it when the analysis
was not driven from a local export.

Before it writes the manifest it runs the frontend contract check (step 10) on
`analysis.json`. A violation there is usually a stale or half-written conversion
over tables that are fine, and that is repairable without judgement, so the
packager rebuilds `analysis.json` from `csv/` and re-checks instead of failing --
the rejected document is kept as `metadata/analysis.rejected.json`. Only a
violation that survives the rebuild fails the packaging, because it means the
tables themselves are missing what the frontend needs, most often the structural
position/id/variant of a heterogeneous cycle. So a package that has a manifest has
been checked, and no finished analysis is ever thrown away over a derived file.

Run it even if the analysis was authored directly in the target layout: it is
idempotent, and it is what produces the manifest. Do not skip it and hand-place
files instead -- the layout is the package's contract, and the NsysScope service
runs this exact script, so skipping it is the only way the two can disagree.

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
