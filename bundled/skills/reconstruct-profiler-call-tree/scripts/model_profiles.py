#!/usr/bin/env python3
"""Model profile definitions for LLM pipeline analysis.

Each ModelProfile captures model-family-specific knowledge needed by the
analysis scripts:

  - Which GPU kernel marks the boundary between consecutive transformer layers
    (anchor_kernel)?
  - How many anchor-kernel blocks does one transformer layer produce
    (blocks_per_layer)?
  - What labels describe each sub-block within a layer (half_labels)?
  - How to classify kernels into categories (category_rules)?
  - How to simplify verbose kernel names (simplify_rules)?

Profiles can be auto-inferred from a model's config.json via
``infer_profile(config)``, or explicitly selected via ``--profile``.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Core data structure
# ---------------------------------------------------------------------------


@dataclass
class ModelProfile:
    """Model-family profile for trace analysis.

    Attributes:
        name: Human-readable profile identifier (e.g. "dsv4_csa_hca").
        anchor_kernel: Substring to identify layer-boundary kernels in the
            trace.  ``None`` means the user must supply ``--anchor-kernel``.
        blocks_per_layer: Number of anchor-kernel invocations per transformer
            layer.  For example, DeepSeek-V4 produces 2 mhc_post blocks per
            layer (one for the attention half, one for the MoE half).
        half_labels: Labels for each sub-block within a layer.  Length must
            equal ``blocks_per_layer``.  E.g. ["attn", "ffn"] or ["full"].
        category_rules: Ordered list of ``(display_label, machine_key, rule)``
            tuples for kernel classification.  Rules are evaluated in order;
            the first match wins.  Unmatched kernels fall into "other".
        simplify_rules: Ordered list of ``(pattern, replacement)`` string
            pairs applied (in order) to simplify verbose kernel names.
        default_num_layers: Fallback layer count when auto-detection fails.
    """

    name: str
    anchor_kernel: Optional[str]
    blocks_per_layer: int
    half_labels: List[str]
    category_rules: list  # List[Tuple[str, str, Callable[[str], bool]]]
    simplify_rules: List[Tuple[str, str]]
    default_num_layers: int = 1


# ---------------------------------------------------------------------------
# Helper: build classification functions
# ---------------------------------------------------------------------------


def _any_sub(*substrings: str) -> Callable[[str], bool]:
    """Return a rule that matches if *any* substring is found in the name."""
    return lambda n: any(s in n for s in substrings)


def _sub(s: str) -> Callable[[str], bool]:
    """Return a rule that matches a single substring."""
    return lambda n: s in n


# ---------------------------------------------------------------------------
# Universal (framework-level) rules
# ---------------------------------------------------------------------------

_UNIVERSAL_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    ("● NCCL AllReduce", "allreduce", _sub("AllReduce")),
    ("  RMSNorm", "rmsnorm", _any_sub("RMSNorm", "rms_normalize")),
    ("  FP8 Quant", "quant", lambda n: "quant" in n.lower() or "Quant" in n),
    ("  TopK", "topk", lambda n: "topk" in n.lower()),
    ("  GEMM fp8", "gemm_fp8", _sub("deep_gemm")),
    ("  GEMM bf16", "gemm_bf16", _sub("nvjet")),
    ("  GEMM f32", "gemm_f32", _sub("sm80_xmma")),
    ("  Activation", "activation", _any_sub("silu_mul_clamp", "act_and_mul")),
    (
        "  RadixSort",
        "radixsort",
        lambda n: "radix_sort" in n.lower() or "RadixSort" in n,
    ),
]

_UNIVERSAL_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    ("void (anonymous namespace)::", ""),
    ("void at::native::", ""),
    ("void flashinfer::", ""),
    ("void deep_gemm::sm90_fp8_gemm_1d2d_impl", "deep_gemm::sm90_fp8_gemm_1d2d"),
    ("void fast_hadamard_transform_kernel", "fast_hadamard_transform_kernel"),
    ("void per_token_group_quant_8bit_kernel", "per_token_group_quant_8bit_kernel"),
    (
        "ncclDevKernel_AllReduce_Sum_bf16_RING_LL(ncclDevKernelArgsStorage<4096ul>)",
        "ncclAllReduce_bf16_RING_LL",
    ),
    ("norm::RMSNormKernel", "RMSNormKernel"),
]


# ---------------------------------------------------------------------------
# DeepSeek-V4 CSA/HCA profile
# ---------------------------------------------------------------------------

_DSV4_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    # Model-specific rules (evaluated before universal rules)
    ("★ MLA Attention", "mla", _sub("flash_fwd_splitkv_mla")),
    ("★ MoE Fused", "moe", _sub("fused_moe_kernel")),
    ("  Hadamard Xform", "hadamard", lambda n: "hadamard" in n.lower()),
    ("  Indexer Cache", "indexer", lambda n: "indexer" in n.lower()),
    ("  Paged MQA", "paged_mqa", _sub("paged_mqa_logits")),
    ("  MLA Metadata", "mla_metadata", _sub("get_mla_metadata")),
    ("  C4 Prefill", "c4_prefill", _sub("c4_prefill")),
    ("  C128 Prefill", "c128_prefill", _sub("c128_prefill")),
    ("  RoPE", "rope", _any_sub("deepseek_rope", "fused_norm_rope")),
    ("  MHC Pre GEMM", "mhc_pre_gemm", _sub("mhc_pre_gemm_sqrsum")),
    ("  MHC Pre Fuse", "mhc_pre_fuse", _sub("mhc_pre_big_fuse")),
    ("  MHC Post", "mhc_post", _sub("mhc_post_tilelang")),
    ("  MoE Gate", "moe_gate", _sub("moe_fused_gate")),
    ("  MoE Align", "moe_align", _sub("moe_align_block")),
    ("  MoE Sort", "moe_sort", _sub("count_and_sort")),
    ("  MLA Cache Store", "mla_cache_store", _sub("fused_store_flashmla_cache")),
    ("  Indexer Store", "indexer_store", _sub("fused_store_indexer_cache")),
]

_DSV4_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    (
        "void deep_gemm::sm90_fp8_paged_mqa_logits",
        "deep_gemm::sm90_fp8_paged_mqa_logits",
    ),
    (
        "void deep_gemm::sched::smxx_paged_mqa_logits_metadata",
        "deep_gemm::paged_mqa_logits_metadata",
    ),
    (
        "void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel",
        "flash_fwd_splitkv_mla_fp8_sparse",
    ),
    ("void smxx::decode::flash_fwd_mla_combine_kernel", "flash_fwd_mla_combine"),
    ("void smxx::decode::get_mla_metadata_kernel", "get_mla_metadata"),
    ("mhc_pre_gemm_sqrsum_tilelang_kernel", "mhc_pre_gemm_sqrsum"),
    ("mhc_post_tilelang_kernel", "mhc_post_tilelang"),
    ("mhc_pre_big_fuse_tilelang_kernel", "mhc_pre_big_fuse"),
    ("fused_moe_kernel", "fused_moe_kernel"),
]


# ---------------------------------------------------------------------------
# DeepSeek-V3 MLA profile
# ---------------------------------------------------------------------------

_DSV3_CATEGORY_RULES: List[Tuple[str, str, Callable[[str], bool]]] = [
    ("★ MLA Attention", "mla", _sub("flash_fwd_splitkv_mla")),
    ("★ MoE Fused", "moe", _sub("fused_moe_kernel")),
    ("  MLA Metadata", "mla_metadata", _sub("get_mla_metadata")),
    ("  MLA Combine", "mla_combine", _sub("flash_fwd_mla_combine")),
    ("  RoPE", "rope", _any_sub("deepseek_rope", "fused_norm_rope")),
    ("  MoE Gate", "moe_gate", _sub("moe_fused_gate")),
    ("  MoE Align", "moe_align", _sub("moe_align_block")),
    ("  MoE Sort", "moe_sort", _sub("count_and_sort")),
    ("  MLA Cache Store", "mla_cache_store", _sub("fused_store_flashmla_cache")),
]

_DSV3_SIMPLIFY_RULES: List[Tuple[str, str]] = [
    (
        "void sm90::decode::sparse_fp8::flash_fwd_splitkv_mla_fp8_sparse_kernel",
        "flash_fwd_splitkv_mla_fp8_sparse",
    ),
    ("void smxx::decode::flash_fwd_mla_combine_kernel", "flash_fwd_mla_combine"),
    ("void smxx::decode::get_mla_metadata_kernel", "get_mla_metadata"),
    ("fused_moe_kernel", "fused_moe_kernel"),
]


# ---------------------------------------------------------------------------
# Built-in profile instances
# ---------------------------------------------------------------------------

PROFILE_DSV4_CSA_HCA = ModelProfile(
    name="dsv4_csa_hca",
    anchor_kernel="mhc_post_tilelang",
    blocks_per_layer=2,
    half_labels=["attn", "ffn"],
    category_rules=_DSV4_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES + _DSV4_SIMPLIFY_RULES,
    default_num_layers=43,
)

PROFILE_DSV3_MLA = ModelProfile(
    name="dsv3_mla",
    anchor_kernel="flash_fwd_mla_combine",
    blocks_per_layer=1,
    half_labels=["full"],
    category_rules=_DSV3_CATEGORY_RULES + _UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES + _DSV3_SIMPLIFY_RULES,
    default_num_layers=61,
)

PROFILE_GENERIC = ModelProfile(
    name="generic",
    anchor_kernel=None,
    blocks_per_layer=1,
    half_labels=["full"],
    category_rules=_UNIVERSAL_CATEGORY_RULES,
    simplify_rules=_UNIVERSAL_SIMPLIFY_RULES,
    default_num_layers=1,
)


# ---------------------------------------------------------------------------
# Profile registry & inference
# ---------------------------------------------------------------------------

BUILTIN_PROFILES: Dict[str, ModelProfile] = {
    "dsv4_csa_hca": PROFILE_DSV4_CSA_HCA,
    "dsv3_mla": PROFILE_DSV3_MLA,
    "generic": PROFILE_GENERIC,
}


# ---------------------------------------------------------------------------
# Optional reference-backed profile registry
# ---------------------------------------------------------------------------

_EXTERNAL_AUTO_INFER_RULES: List[Tuple[str, list]] = []


def _rule_from_spec(spec: dict) -> Callable[[str], bool]:
    """Build a kernel-name match rule from a reference JSON rule spec.

    Supported match types:
      - contains: case-sensitive substring
      - contains_any: case-sensitive OR over substrings
      - lower_contains: case-insensitive substring
      - lower_contains_any: case-insensitive OR over substrings
    """
    match = spec.get("match", "contains")
    if match == "contains":
        needle = spec.get("pattern", "")
        return lambda n, needle=needle: needle in n
    if match == "contains_any":
        needles = list(spec.get("patterns") or [])
        return lambda n, needles=needles: any(x in n for x in needles)
    if match == "lower_contains":
        needle = str(spec.get("pattern", "")).lower()
        return lambda n, needle=needle: needle in n.lower()
    if match == "lower_contains_any":
        needles = [str(x).lower() for x in (spec.get("patterns") or [])]
        return lambda n, needles=needles: any(x in n.lower() for x in needles)
    raise ValueError(f"Unsupported profile rule match type: {match}")


def _infer_condition_matches(config: dict, cond: dict) -> bool:
    field = cond.get("field")
    op = cond.get("op", "non_empty")
    value = config.get(field) if field else None
    if op == "non_empty":
        return bool(value)
    if op == "exists":
        return field in config
    if op == "gt":
        return value is not None and value > cond.get("value", 0)
    if op == "eq":
        return value == cond.get("value")
    if op == "contains":
        return cond.get("value") in (value or [])
    raise ValueError(f"Unsupported infer condition op: {op}")


def _load_reference_profiles() -> Dict[str, ModelProfile]:
    """Load additional/override profiles from references/model_profiles.json.

    This makes model-family knowledge data-driven: adding a new model profile
    should normally only require editing the reference JSON, not the analysis
    workflow or scripts. Built-in Python profiles remain as a fallback.
    """
    global _EXTERNAL_AUTO_INFER_RULES
    ref = Path(__file__).resolve().parents[1] / "references" / "model_profiles.json"
    if not ref.is_file():
        return {}
    with ref.open() as f:
        data = json.load(f)

    universal_rules = [
        (r["label"], r["key"], _rule_from_spec(r))
        for r in data.get("universal_category_rules", [])
    ]
    universal_simplify = [tuple(x) for x in data.get("universal_simplify_rules", [])]

    profiles: Dict[str, ModelProfile] = {}
    _EXTERNAL_AUTO_INFER_RULES = []
    for spec in data.get("profiles", []):
        name = spec["name"]
        cat_rules = [(r["label"], r["key"], _rule_from_spec(r)) for r in spec.get("category_rules", [])]
        simplify_rules = [tuple(x) for x in spec.get("simplify_rules", [])]
        include_universal = spec.get("include_universal", True)
        profile = ModelProfile(
            name=name,
            anchor_kernel=spec.get("anchor_kernel"),
            blocks_per_layer=int(spec.get("blocks_per_layer", 1)),
            half_labels=list(spec.get("half_labels") or ["full"]),
            category_rules=cat_rules + (universal_rules if include_universal else []),
            simplify_rules=universal_simplify + simplify_rules,
            default_num_layers=int(spec.get("default_num_layers", 1)),
        )
        profiles[name] = profile
        infer = spec.get("auto_infer") or []
        if infer:
            _EXTERNAL_AUTO_INFER_RULES.append((name, infer))
    return profiles


REFERENCE_PROFILES: Dict[str, ModelProfile] = _load_reference_profiles()
PROFILES: Dict[str, ModelProfile] = {**BUILTIN_PROFILES, **REFERENCE_PROFILES}


def get_profile(name: str) -> ModelProfile:
    """Look up a profile by name from reference JSON first, then built-ins."""
    if name not in PROFILES:
        raise ValueError(
            f"Unknown profile '{name}'. Available: {', '.join(PROFILES)}"
        )
    return PROFILES[name]


def normalize_compress_ratios(
    config: dict, num_layers: Optional[int] = None
) -> List[int]:
    """Return per-hidden-layer compress ratios, validating known config shapes.

    Some DeepSeek-V4 configs publish one extra ratio for next-token-prediction
    layers. The pipeline analyzers operate on transformer hidden layers only,
    so that trailing nextn ratio is intentionally excluded instead of being
    silently sliced by callers.
    """
    ratios = list(config.get("compress_ratios") or [])
    if not ratios:
        return []

    n_layers = num_layers or config.get("num_hidden_layers")
    if not n_layers:
        return ratios

    if len(ratios) == n_layers:
        return ratios

    nextn_layers = config.get("num_nextn_predict_layers", 0) or 0
    if nextn_layers and len(ratios) == n_layers + nextn_layers:
        return ratios[:n_layers]

    raise ValueError(
        "compress_ratios length mismatch: "
        f"got {len(ratios)}, expected num_hidden_layers={n_layers}"
        + (f" or + num_nextn_predict_layers={nextn_layers}" if nextn_layers else "")
    )


def infer_profile(config: dict) -> ModelProfile:
    """Auto-detect the model profile from a config.json dict.

    Reference-backed profiles are checked first in the order listed in
    references/model_profiles.json. Built-in heuristics remain as fallback.
    """
    for name, conditions in _EXTERNAL_AUTO_INFER_RULES:
        if all(_infer_condition_matches(config, cond) for cond in conditions):
            return PROFILES[name]

    cr = config.get("compress_ratios", [])
    if cr:
        return get_profile("dsv4_csa_hca")

    kv_lora_rank = config.get("kv_lora_rank", 0)
    if kv_lora_rank and kv_lora_rank > 0:
        return get_profile("dsv3_mla")

    return get_profile("generic")
