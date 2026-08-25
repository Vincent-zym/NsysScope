# Reporting Reference

Use this reference when writing the final human-facing analysis report.

## Recommended report sections

1. Trace metadata: trace path, config path, GPU type, TP/EP, serving mode.
2. Model/profile summary: selected profile, anchor kernel, blocks per layer, layer count, and inference rationale.
3. Forward-pass selection: all-pass summary and chosen pass rationale.
4. Layer timing: per-layer table and cluster/group statistics.
5. Representative compute flow: hot kernels for selected layer(s), with category, duration, percent, and trace offset.
6. Optional call tree / graph artifacts: paths and status from `final_report.md`.
7. Bottleneck summary: slowest layer/submodule/kernel and likely next target.
8. Caveats: whether `file:line` came from real stack metadata, was inferred, or was unavailable.

## Model architecture fields to summarize when present

- `model_type`, model name, or architecture identifier
- `num_hidden_layers`, `hidden_size`
- `num_attention_heads`, `num_key_value_heads`, `head_dim`
- Attention-specific fields such as LoRA ranks or attention type
- MoE fields such as `num_experts`, top-k, shared experts, routed/shared intermediate size
- Cache/indexer/sparse-attention fields when present
- Layer-type distribution from model-specific config arrays, if available

Do not hard-code a model family in the report. If a field is absent, omit it or mark it unavailable.
