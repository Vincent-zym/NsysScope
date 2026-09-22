"""Contract tests for the deterministic projection-GEMM shape/MFU filler.

Projection GEMMs run in persistent cutlass/nvjet kernels with no logical (M,N,K)
in the name, so the filler derives a module-level FLOP-equivalent shape from the
config (N,K fixed) and the token count M, and an MFU from the module's measured
GPU time. These tests pin the DeepSeek-V4.1-Flash catalog values and the
"leave unknown models / sparse modules untouched" contract.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_filler():
    path = ROOT / "scripts" / "fill_core_gemm_shapes.py"
    spec = importlib.util.spec_from_file_location("fill_core_gemm_shapes", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


DSV41_CFG = {
    "architectures": ["DeepseekV4ForCausalLM"],
    "text_config": {
        "model_type": "deepseek_v4",
        "hidden_size": 5120,
        "q_lora_rank": 1280,
        "head_dim": 512,
        "num_attention_heads": 64,
        "moe_intermediate_size": 2304,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "hc_mult": 4,
        "dspark_block_size": 5,
    },
}


def test_catalog_dims_match_config():
    mod = load_filler()
    cat = mod.dsv41_catalog(DSV41_CFG)
    # qkv: wqkv_a (H -> q_lora+head_dim) + q_b (q_lora -> heads*head_dim)
    assert cat["Attention 输入与投影"]["fp8"] == [(5120, 1792), (1280, 32768)]
    # o_proj: grouped down (heads*head_dim -> o_lora) + up (groups*o_lora -> H)
    assert cat["Attention 输出与投影"]["fp8"] == [(32768, 1024), (8192, 5120)]
    # shared expert: gate+up (H -> 2*moe_i) + down (moe_i -> H)
    assert cat["MoE 共享专家"]["fp8"] == [(5120, 4608), (2304, 5120)]
    # hc mixing GEMM from the deep_gemm template <24, hc_mult*H>
    assert cat["Attention 输入与投影"]["tf32"] == [(20480, 24)]


def test_tokens_from_dspark_block():
    mod = load_filler()
    # decode verify processes block_size+1 tokens per sequence
    assert mod.resolve_tokens(DSV41_CFG, "decode", 256, None, None) == 256 * 6
    # explicit override wins
    assert mod.resolve_tokens(DSV41_CFG, "decode", 256, None, 1536) == 1536
    # prefill uses chunk size
    assert mod.resolve_tokens(DSV41_CFG, "prefill", None, 4096, None) == 4096


def _write_core_csv(path: Path, rows: list[dict]) -> None:
    fields = ["序号", "功能模块", "module", "算子名称", "算子耗时(us)", "shape", "mfu", "mbu"]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i, r in enumerate(rows, 1):
            w.writerow({"序号": i, "shape": "", "mfu": "", "mbu": "", **r})


def test_fill_writes_module_shape_and_leaves_sparse_blank(tmp_path, monkeypatch):
    mod = load_filler()
    cfg_path = tmp_path / "config.json"
    import json
    cfg_path.write_text(json.dumps(DSV41_CFG))
    csv_path = tmp_path / "core.csv"
    _write_core_csv(csv_path, [
        # two qkv kernels (module total time 100us) -> one module MFU
        {"功能模块": "Attention 输入与投影", "module": "mla/qkv_proj_gemm",
         "算子名称": "blockscaled_gemm", "算子耗时(us)": "60"},
        {"功能模块": "Attention 输入与投影", "module": "mla/qkv_proj_gemm",
         "算子名称": "blockscaled_gemm", "算子耗时(us)": "40"},
        # sparse attention must stay blank
        {"功能模块": "MLA 稀疏核心计算", "module": "mla/core_sparse",
         "算子名称": "flash_fwd_splitkv_mla_fp8_sparse_kernel", "算子耗时(us)": "256"},
        # one experts row so units auto-resolves to 1
        {"功能模块": "MoE 专家计算与输出合并", "module": "moe/experts_mega_moe",
         "算子名称": "mega_moe", "算子耗时(us)": "548"},
    ])
    import sys
    argv = ["fill", "--core-table", str(csv_path), "--model-config", str(cfg_path),
            "--stage", "decode", "--batch-size", "256"]
    monkeypatch.setattr(sys, "argv", argv)
    mod.main()

    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    qkv = [r for r in rows if r["功能模块"] == "Attention 输入与投影"]
    assert all(r["shape"] == "(M=1536,N=10080,K=5120)" for r in qkv)
    # module MFU: 2*1536*(5120*1792+1280*32768+20480*24) FLOPs / (100us) / eff_peak
    assert qkv[0]["mfu"] == qkv[1]["mfu"] and qkv[0]["mfu"].endswith("%")
    mfu = float(qkv[0]["mfu"].rstrip("%"))
    assert 0 < mfu < 100
    # MBU filled from the fp8 projection byte model (act fp8 x weight fp8 -> bf16 out)
    assert qkv[0]["mbu"] == qkv[1]["mbu"] and qkv[0]["mbu"].endswith("%")
    mbu = float(qkv[0]["mbu"].rstrip("%"))
    assert 0 < mbu < 100
    sparse = [r for r in rows if r["功能模块"] == "MLA 稀疏核心计算"]
    assert sparse[0]["shape"] == "" and sparse[0]["mfu"] == "" and sparse[0]["mbu"] == ""


def test_unknown_model_is_left_untouched(tmp_path, monkeypatch):
    mod = load_filler()
    import json, sys
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({"architectures": ["LlamaForCausalLM"],
                                     "text_config": {"model_type": "llama",
                                                     "hidden_size": 4096}}))
    csv_path = tmp_path / "core.csv"
    _write_core_csv(csv_path, [
        {"功能模块": "Attention 输入与投影", "module": "x", "算子名称": "y",
         "算子耗时(us)": "10"},
    ])
    monkeypatch.setattr(sys, "argv",
                        ["fill", "--core-table", str(csv_path), "--model-config",
                         str(cfg_path), "--tokens", "1536"])
    mod.main()
    rows = list(csv.DictReader(csv_path.open(newline="", encoding="utf-8")))
    assert rows[0]["shape"] == "" and rows[0]["mfu"] == ""
