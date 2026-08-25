# Kernel Category Reference

Use this reference when interpreting category columns or updating kernel classification for new models. Keep category tables here so the main workflow remains stable.

## Universal categories

| Category | Match pattern | Scope |
|---|---|---|
| `allreduce` | `AllReduce` | communication |
| `rmsnorm` | `RMSNorm`, `rms_normalize` | normalization |
| `quant` | case-insensitive `quant` | quantization |
| `topk` | case-insensitive `topk` | routing/top-k |
| `gemm_fp8` | `deep_gemm` | fp8 GEMM |
| `gemm_bf16` | `nvjet` | bf16 GEMM |
| `gemm_f32` | `sm80_xmma` | f32 GEMM |
| `activation` | `silu_mul_clamp`, `act_and_mul` | activation |
| `radixsort` | `radix_sort`, `RadixSort` | sorting |

## Current model-specific categories

| Category | Match pattern | Profiles |
|---|---|---|
| `mla` | `flash_fwd_splitkv_mla` | `dsv4_csa_hca`, `dsv3_mla` |
| `moe` | `fused_moe_kernel` | `dsv4_csa_hca`, `dsv3_mla` |
| `mla_metadata` | `get_mla_metadata` | `dsv4_csa_hca`, `dsv3_mla` |
| `mla_combine` | `flash_fwd_mla_combine` | `dsv3_mla` |
| `rope` | `deepseek_rope`, `fused_norm_rope` | `dsv4_csa_hca`, `dsv3_mla` |
| `hadamard` | case-insensitive `hadamard` | `dsv4_csa_hca` |
| `indexer` | case-insensitive `indexer` | `dsv4_csa_hca` |
| `paged_mqa` | `paged_mqa_logits` | `dsv4_csa_hca` |
| `c4_prefill` | `c4_prefill` | `dsv4_csa_hca` |
| `c128_prefill` | `c128_prefill` | `dsv4_csa_hca` |
| `mhc_pre_gemm` | `mhc_pre_gemm_sqrsum` | `dsv4_csa_hca` |
| `mhc_pre_fuse` | `mhc_pre_big_fuse` | `dsv4_csa_hca` |
| `mhc_post` | `mhc_post_tilelang` | `dsv4_csa_hca` |
| `moe_gate` | `moe_fused_gate` | `dsv4_csa_hca`, `dsv3_mla` |
| `moe_align` | `moe_align_block` | `dsv4_csa_hca`, `dsv3_mla` |
| `moe_sort` | `count_and_sort` | `dsv4_csa_hca`, `dsv3_mla` |
| `mla_cache_store` | `fused_store_flashmla_cache` | `dsv4_csa_hca`, `dsv3_mla` |
| `indexer_store` | `fused_store_indexer_cache` | `dsv4_csa_hca` |

Update `references/model_profiles.json` for machine-consumed rules. Update this markdown when adding or changing human-facing category explanations.
