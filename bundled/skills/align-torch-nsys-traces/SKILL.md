---
name: align-torch-nsys-traces
description: Align a PyTorch torch.profiler Chrome trace (`.json`/`.json.gz`) with an Nsight Systems report (`.nsys-rep`/`.sqlite`) at forward, layer, kernel-family, and per-kernel-launch level. Use after profiling when Codex must compare torch profiler output with nsys output, split representative forwards/layers, and produce alignment CSVs.
---

# Torch/Nsys Trace Align

## Purpose

Use this skill after collection to align:

- torch profiler Chrome traces: `*.trace.json.gz`
- optional torch kernel-python mapping CSV
- nsys reports: `*.nsys-rep` or exported `*.sqlite`

The output is a standalone alignment directory with torch split files, nsys split files, and alignment CSVs.

## Inputs

Required:

- `torch_trace`: one torch profiler Chrome trace file
- `nsys_report`: `.nsys-rep`, `.sqlite`, or a directory containing one of those files
- `output_dir`

Recommended:

- `config`: model `config.json`; used to get `num_hidden_layers`
- `torch_mapping`: kernel-python mapping CSV from `torch-profiler-kernel-python-line-mapping`
- `anchor`: layer boundary kernel, e.g. `mhc_post_tilelang`

Optional:

- `num_layers`
- `blocks_per_layer`
- `torch_fwd_pass`
- `nsys_device`
- `nsys_forward`
- `strict_nsys`: fail instead of writing torch-only outputs when no nsys report is found

## Workflow

1. Validate inputs and load the torch trace.
   - Kernel events are selected by case-insensitive kernel category matching.
   - If the nsys path is a directory, search recursively for the newest `.sqlite` first, then newest `.nsys-rep`.
2. Detect layer-boundary anchor kernels.
3. Split torch forward/layers using:
   - `blocks_per_forward = num_layers * blocks_per_layer`
   - one layer = `blocks_per_layer` consecutive anchor blocks
4. Select a torch forward:
   - use `--torch-fwd-pass` when provided
   - otherwise choose a later steady forward when available
5. Load or export nsys sqlite:
   - `.sqlite` input is used directly
   - `.nsys-rep` input reuses a sibling same-stem `.sqlite` when present
   - `.nsys-rep` input is exported with `nsys export --type sqlite`
   - when no nsys report is found, write torch-only outputs and metadata unless `--strict-nsys` is set
6. Select nsys device and forward:
   - prefer devices whose anchor count forms complete forward blocks
   - use `--nsys-device` / `--nsys-forward` when provided
7. Split the selected nsys forward into layers with the same anchor rule.
8. Compare:
   - per-layer kernel count
   - per-layer canonical kernel family multiset
   - per-kernel-launch order and canonical family
9. Write CSVs and metadata.

## Script

Use:

```bash
python scripts/align_torch_nsys_traces.py \
  --torch-trace /path/to/TP-0.trace.json.gz \
  --torch-mapping /path/to/kernel_python_line_mapping_tp0.csv \
  --nsys-report /path/to/profile.nsys-rep \
  --config /path/to/model/config.json \
  --output-dir /path/to/alignment_run \
  --anchor mhc_post_tilelang \
  --blocks-per-layer 2
```

When nsys input is `.nsys-rep`, the script first reuses a sibling same-stem `.sqlite` if present; otherwise it writes an exported sqlite under `output_dir/nsys/`.
When nsys input is a torch-only profile directory with no `.nsys-rep` or `.sqlite`, the script writes the torch forward/layer split plus metadata with `mode: torch_only`; it does not claim kernel alignment.

## Output Contract

Expected layout:

```text
output_dir/
  README.md
  torch/
    selected_forward_layers_summary.csv
    selected_forward_kernels_with_python.csv
  nsys/
    selected_forward_layers_summary.csv
    selected_forward_kernels.csv
  alignment/
    layer_alignment.csv
    layer_family_overlap.csv
    kernel_launch_alignment.csv
    kernel_launch_alignment_summary_by_layer.csv
    kernel_launch_alignment_metadata.json
  scripts/
    copied command metadata is optional
```

Before reporting success, verify:

- torch and nsys selected forwards both have `num_layers` rows
- layer kernel count diffs are reported
- kernel-launch metadata reports matched/unmatched counts
- time columns include units:
  - torch: `torch_start_us`, `torch_end_us`
  - nsys: `nsys_start_ns`, `nsys_end_ns`

Torch-only fallback layout:

```text
output_dir/
  README.md
  torch/
    selected_forward_layers_summary.csv
    selected_forward_kernels_with_python.csv
  alignment/
    kernel_launch_alignment_metadata.json
```

In fallback mode, metadata contains `mode: torch_only`, `reason`, and `nsys: null`.

## Canonical Kernel Family

Use canonical families for cross-tool matching. Raw demangled names differ across tools, for example `(anonymous namespace)` vs `<unnamed>` or template integer spelling. The bundled script uses explicit substring rules first, then a function-name fallback.

Examples:

- `per_token_group_quant_8bit_kernel<...>` -> `per_token_group_quant`
- `sm100_fp8_fp4_gemm_1d1d_impl<...>` -> `deep_gemm_1d1d`
- `mhc_post_tilelang_kernel` -> `mhc_post`
- `topk_512_transform` and `topk_fused_transform` -> `moe_hash_topk_fused`

Treat `same_order_exact_name` as strongest evidence and `same_order_family` as normal successful alignment when raw names differ only by demangling/template style.
