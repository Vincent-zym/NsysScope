---
name: reconstruct-profiler-call-tree
description: "Inspect LLM torch profiler traces at forward-pass, layer, submodule, Python/ATen/CUDA API, and kernel level. Use when you need layer timings, anchor-kernel boundaries, representative kernel flows, Perfetto time ranges, layer-aware call trees/call graphs, kernel-to-source mapping, or slowest layer/submodule/kernel CSV outputs."
---

# LLM Pipeline Analysis

## Overview

Use this when a whole-trace profiler summary is too coarse. The scripts read a
Chrome-trace JSON file, find layer-boundary anchor kernels, group kernels into
forward passes and layers, and print timing tables you can use for Perfetto
navigation or detailed timing analysis. The skill can also reconstruct a
best-effort layer-aware execution hierarchy:

```text
Model → Layer → Layer Submodule → Python Function → ATen / Custom Op → CUDA Runtime API → CUDA Kernel
```

For that hierarchy it can emit both markdown call-tree tables and Mermaid call
graphs with time, self-time, percentage, source `file:line`, and parent-child
relationships when the trace contains sufficient CPU/CUDA correlation and stack
metadata.

## When To Use It

- when you need to know **which layers** contribute most
- when the model has alternating or heterogeneous layer types defined by its active profile/config
- when you need to compare cold-start vs steady-state forward passes
- when you need to navigate to a specific layer in Perfetto UI
- when you need to select representative layers for deep-dive analysis
- when you need a **Layer → Submodule → Python → ATen/Custom Op → CUDA API → Kernel** call tree
- when you need Mermaid call graphs, kernel-to-layer mappings, or slowest layer/submodule/kernel CSVs
- when you need to map CUDA kernels back to owning model code/source locations

## Confirmation Required

Before running scripts, collect or verify these inputs:

| Item | Why it matters | How to obtain | Default if user skips |
|---|---|---|---|
| Model name | Determines which `config.json` to use; affects layer classification | Ask user | — (required) |
| Model profile | Determines anchor kernel, blocks-per-layer, and kernel classification rules | Ask user or auto-infer from config | Auto-inferred from config |
| `config.json` path | Provides layer count and model/profile-specific fields used for inference and labeling | Ask user or search filesystem | — (required unless explicit profile/num-layers are supplied) |
| GPU type | Optional context for reports and hardware notes | Ask user | — |
| TP / EP | Parallelism config can affect kernel naming and communication count | Ask user or infer from trace filename (e.g. `TP-0`) | Note as unknown if unavailable |
| Serving mode | Decode vs prefill changes kernel mix and FLOPs profile | Ask user | decode B=1 |

If the user cannot provide `config.json`, search common locations such as
`/root/workspace/*/config.json` and the HuggingFace cache. If it is still not
available, require an explicit `--profile`.


## Input / Output Contract

When this skill generates artifacts, require or create an `output_dir` and keep
all generated content inside it.

### Inputs

Required inputs depend on the requested analysis depth:

- `trace`: torch profiler Chrome-trace JSON path (`.json` or `.json.gz`).
- `config`: model `config.json` when available; needed for profile inference, layer count, and profile-specific layer labels.
- `profile` / `anchor-kernel`: required fallback when `config` cannot infer a supported model profile.
- `fwd-pass`: forward pass index for layer/detail/call-tree analysis. First identify a steady-state pass when the user does not specify one.
- `output_dir`: required for artifact-producing runs when the caller provides a destination. If a provided `output_dir` already exists, treat the task as failed and write `final_report.md` explaining that the directory already existed.

If the user does not specify `output_dir`, create a default unique directory
under `./outputs/`, using a short trace/input summary plus rank, for example
`./outputs/TP-0_trace_fwd5_r000`. Never mix artifacts from multiple runs in the
same output directory.

### Outputs

Prefer structured subdirectories by function instead of placing all artifacts at
the output root. For current artifact layouts and file meanings, read
`references/output_contract.md`. For other scripts that primarily print to
stdout, redirect important outputs into the current run's `output_dir` when the
user asks for persistent artifacts or a report.

## Final Report Contract

Every artifact-producing workflow must finish by creating
`output_dir/final_report.md`. The final report must be structured and include:

1. **Status**: `成功` or `失败`. If failed, include the reason.
2. **Task Metadata**: trace path, config path, selected profile, anchor kernel, forward pass, and `output_dir` when applicable.
3. **Core Important Files**: relative paths and short descriptions for the key outputs used to inspect bottlenecks.
4. **Gate Check**: whether all expected generated files exist and are non-empty.
5. **All Generated Files**: relative paths and short descriptions for every file produced by the skill.

Do not mark the task successful unless `final_report.md` exists and all expected
outputs for the requested workflow exist and are non-empty.

## Gate Checks

Before ending any artifact-producing task:

1. Enumerate the expected relative output files for the requested workflow.
2. Check that every file exists and is non-empty.
3. If a gate fails, rerun or regenerate the missing artifact when possible.
4. If the gate still fails, keep `final_report.md`, set status to `失败`, and record the failing file paths and reason.

When this Skill invokes a companion Skill, first read that Skill's input/output
and final-report requirements. Pass a unique explicit `output_dir` beneath the
current run directory, then gate the next step on its `final_report.md` status being
`成功`.

## Reference Files and Extensibility

Keep model-family-specific and table-heavy knowledge in `references/` so adding
new models usually does not require changing this workflow file. Read only the
reference needed for the current request:

- `references/model_profile_guide.md`: profile table, layer-boundary concepts, and how to add a new model profile. Read when choosing or updating a model profile.
- `references/model_profiles.json`: machine-consumed profile registry for anchors, blocks/layer, auto-infer rules, kernel category rules, and simplify rules. Update this for new model families when the existing JSON match operators are sufficient.
- `references/kernel_categories.md`: human-facing kernel category table and interpretation notes. Read when explaining category columns or updating classification meanings.
- `references/output_contract.md`: structured output layout, file meanings, call-tree columns, and gate expectations. Read before changing artifact layouts or when reporting generated files.
- `references/reporting_template.md`: expandable report structure and model-agnostic architecture fields. Read when producing a final analysis report.

The Python scripts load `references/model_profiles.json` at runtime and merge it
with built-in fallbacks. For new models, prefer updating that JSON plus the
matching markdown reference. Edit Python only when a new kind of match operator
or analysis algorithm is required.

## Model Profiles

Scripts use **ModelProfile** to determine layer boundary detection, sub-block
labels, kernel classification, and kernel-name simplification. Profiles are
auto-inferred from `config.json` when possible or selected via `--profile`.

For current profiles and the data-only extension procedure, read
`references/model_profile_guide.md`. For the actual machine-consumed rules, read
or update `references/model_profiles.json`. Use
`--profile generic --anchor-kernel YOUR_KERNEL` for models not yet covered by a
profile.

## Prerequisites

- A `torch.profiler` trace in Chrome-trace JSON format (`.json` or `.json.gz`)
- The model's `config.json` when available (for profile inference and layer metadata)
- The trace must contain a recognizable layer-boundary anchor kernel
  (auto-detected from the profile, or specified via `--anchor-kernel`)

## Layer Boundary Detection

The scripts use an anchor kernel as a layer-boundary marker. The active
`ModelProfile` defines the anchor substring, `blocks_per_layer`, and sub-block
labels. This mechanism is model-family-specific; do not hard-code model-specific
examples in workflow logic.

Read `references/model_profile_guide.md` when you need exact profile details,
need to explain how a model's layers are segmented, or need to add/update a
model profile.

## Scripts

### 1. `layer_timeline_analyzer.py` — Per-layer timeline and cluster stats

```bash
# Show all forward passes summary (cold-start vs steady-state)
python3 scripts/layer_timeline_analyzer.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --show-all-passes

# Detailed per-layer breakdown for a specific forward pass
python3 scripts/layer_timeline_analyzer.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5

# Auto-select first steady-state pass
python3 scripts/layer_timeline_analyzer.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json
```

The script prints:
- Per-layer wall-clock time, sum-duration, and category breakdown using the active profile categories
- Layer cluster statistics grouped by profile/config-derived layer type
- All-passes summary showing cold-start → steady-state growth

### 2. `layer_kernel_breakdown.py` — Per-layer kernel detail and compute flow

```bash
# Single layer kernel dump
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 3

# Compute flow format (with model architecture summary and category column)
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 3 --format compute-flow

# JSON export
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 3 --format json

# Compare two layers side-by-side
python3 scripts/layer_kernel_breakdown.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layer 2 --compare-layer 3
```

Output formats:
- `--format text` (default): grouped summary + top hot kernels ranked by duration, with simplified names and percentages
- `--format compute-flow`: model architecture summary + per-kernel hotness table with `Category`, `%`, and `ts_rel(ms)` columns
- `--format json`: machine-readable per-kernel detail ranked by duration
- Kernel diff when comparing two layers (unique kernels in each)

### 3. `perfetto_time_mapper.py` — Perfetto UI time navigation

```bash
# Show all forward pass time ranges in Perfetto
python3 scripts/perfetto_time_mapper.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json

# Layer-level time ranges for a specific forward pass
python3 scripts/perfetto_time_mapper.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 --layers 2,3,38,42
```

The script prints:
- Forward pass time ranges in Perfetto-relative seconds
- Per-layer start/end times with compress_ratio labels


### 4. `layer_call_tree.py` — Layer-aware call tree and call graph outputs

Use this when the user asks for a call tree, call graph, layer-aware DAG,
source mapping, kernel-to-layer table, slowest layers/submodules, or a complete
execution hierarchy across Python/ATen/CUDA/runtime/kernel levels.

```bash
python3 scripts/layer_call_tree.py \
  --trace /path/to/TP-0.trace.json.gz \
  --config /path/to/config.json \
  --fwd-pass 5 \
  --output-dir /tmp/inference_call_tree_outputs_new

# Generic model / custom anchor
python3 scripts/layer_call_tree.py \
  --trace /path/to/trace.json.gz \
  --profile generic --anchor-kernel YOUR_LAYER_BOUNDARY_KERNEL \
  --num-layers 48 --fwd-pass 0 \
  --output-dir outputs/
```

The script writes a structured artifact directory and `final_report.md`. For the
current directory layout, file meanings, and call-tree column definitions, read
`references/output_contract.md`.

Implementation notes and caveats:

- Layer ownership still uses the active `ModelProfile` and anchor-kernel boundary logic from the existing scripts.
- CPU/Python/ATen/CUDA API ownership is best-effort: the script first uses torch-profiler correlation metadata such as `correlation` and `External id`, then falls back to timestamp containment.
- Source `file:line` is available only when the trace was collected with stack metadata (for example `with_stack=True`) or when the profiler exported callsite fields. If missing, leave the field blank and still report timing hierarchy.
- Submodule labels (`self_attn`, `mlp`, `norm`, `comm`, etc.) are inferred from kernel/op/category names. Treat them as mapping aids and verify against model code for final reports.

## Workflow

### Step 1: Identify steady-state forward pass

```bash
python3 scripts/layer_timeline_analyzer.py \
  --trace $TRACE --config $CONFIG --show-all-passes
```

Read the "all-passes" table. The first pass is cold-start (few tokens).
Find the first pass where layer-0 wall-clock stabilizes (typically pass 3-5).

### Step 2: Per-layer breakdown on steady-state pass

```bash
python3 scripts/layer_timeline_analyzer.py \
  --trace $TRACE --config $CONFIG --fwd-pass 5
```

Identify:
- Which profile/config-derived layer type dominates
- The category proportion per layer type using the active profile rules
- Which layer type or category is the best next target

### Step 3: Compute flow for representative layer(s)

Select 1-2 representative layers, ideally one per bottleneck layer type or kernel category, then:

```bash
# Human-readable compute flow table
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 3 --format compute-flow

# JSON export
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 3 --format json > /tmp/layer3_detail.json
```

The `--format compute-flow` output includes:
- Model architecture summary at the top
- Per-kernel hotness table with `# | Half | Category | Simplified Name | dur(us) | % | ts_rel(ms) | Input Dims`
- Rows are ranked by `dur(us)` descending by default; use `ts_rel(ms)` to jump back to the kernel's trace location.

### Step 4: Compare layer types (optional)

```bash
python3 scripts/layer_kernel_breakdown.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layer 2 --compare-layer 3
```

This shows the exact kernel difference between the two selected layers or layer types.

### Step 5: Navigate in Perfetto UI (optional)

```bash
python3 scripts/perfetto_time_mapper.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 --layers 2,3,38,42
```

Use the printed time ranges to navigate directly in Perfetto.


### Step 6: Generate layer-aware call tree / call graph (optional)

After choosing a representative forward pass, generate the complete hierarchy
outputs when the user needs call-tree/call-graph/source-mapping artifacts:

```bash
python3 scripts/layer_call_tree.py \
  --trace $TRACE --config $CONFIG \
  --fwd-pass 5 \
  --output-dir outputs/call_tree_run_fwd5_r000
```

The `--output-dir` must not already exist. If omitted, the script creates a
unique `./outputs/<trace>_fwd5_rXXX` directory. Read
`call_tree/layer_call_tree.md` for tabular attribution,
`call_graph/layer_call_graph_with_time.mmd` for visual DAG review, the CSVs for
ranked bottleneck tables, and `final_report.md` for final status and artifact
gate results. In reports, explicitly state whether source `file:line` was
available from the trace or inferred/missing.

## Layer Type Classification

Layer-type classification is profile/config dependent and should remain
extensible. Use the active profile and model config to label layers, and avoid
hard-coding one model family's layer labels in reports. For current profile
behavior and extension instructions, read `references/model_profile_guide.md`.

## Kernel Categories

Kernel categories are defined by the active `ModelProfile`. The machine-consumed
rules live in `references/model_profiles.json`; human-facing category meanings
live in `references/kernel_categories.md`. When adding a new model, update those
references instead of changing this workflow unless a new algorithm or match
operator is needed.

## Reporting Checklist

Use `references/reporting_template.md` for the expandable report structure and
model-agnostic architecture fields. At minimum include:

1. Trace metadata and selected `output_dir`.
2. Selected profile, anchor kernel, layer count, and why that profile was chosen.
3. Forward-pass selection rationale.
4. Per-layer / cluster timing summary and representative compute-flow evidence.
5. Call-tree/call-graph artifact paths and `final_report.md` status when those artifacts were requested.
6. Bottleneck summary: slowest layer/submodule/kernel and likely next target.
7. Caveats: whether `file:line` came from real stack metadata, was inferred, or was unavailable.

For generated artifacts, report relative paths from `output_dir`, not absolute
paths only.
