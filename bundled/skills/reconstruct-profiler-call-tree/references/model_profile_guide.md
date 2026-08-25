# Model Profile Reference

Use this reference when model-family-specific details are needed or when adding a new model. Keep the main `SKILL.md` workflow unchanged; update this reference and `model_profiles.json` instead.

## Profile table

| Profile | Anchor kernel | Blocks/layer | Layer structure | Auto-infer condition |
|---|---|---:|---|---|
| `dsv4_csa_hca` | `mhc_post_tilelang` | 2 | attention half + FFN/MoE half | `compress_ratios` is non-empty |
| `dsv3_mla` | `flash_fwd_mla_combine` | 1 | full layer | `kv_lora_rank > 0` |
| `generic` | user-specified or auto candidate | 1 | full layer | fallback |

## Layer boundary model

The scripts use an anchor kernel as a layer-boundary marker. The active `ModelProfile` defines:

- `anchor_kernel`: substring used to find layer-boundary kernels
- `blocks_per_layer`: number of anchor blocks per transformer layer
- `half_labels`: labels for each block within a layer, such as `attn` / `ffn`

For any profile, one forward pass contains:

```text
num_hidden_layers * blocks_per_layer
```

anchor blocks. Forward pass `P` starts at:

```text
P * (num_hidden_layers * blocks_per_layer)
```

For a two-block profile such as `dsv4_csa_hca`, a layer has two consecutive anchor-delimited regions. For a one-block profile such as `dsv3_mla` or `generic`, a layer has one anchor-delimited region.

## Adding a new model profile

Prefer data-only updates:

1. Add a new object to `references/model_profiles.json.profiles`.
2. Set `name`, `anchor_kernel`, `blocks_per_layer`, `half_labels`, and `default_num_layers`.
3. Add `auto_infer` conditions only if config-based inference is reliable.
4. Add model-specific `category_rules` and `simplify_rules`.
5. Reuse `include_universal: true` unless the model needs a fully custom category set.
6. Run a representative script with `--profile <new_name>` and verify output gates.

Only edit Python code when a new rule match operator is needed beyond the supported JSON match types: `contains`, `contains_any`, `lower_contains`, and `lower_contains_any`.
