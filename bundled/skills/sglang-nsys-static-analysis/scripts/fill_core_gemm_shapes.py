#!/usr/bin/env python3
"""Fill projection-level GEMM shape/MFU into the core-compute table.

Attention/MoE projection GEMMs run in persistent cutlass/nvjet kernels whose
names carry no logical (M,N,K), and one functional module is split across a
variant-dependent number of kernels -- so a per-kernel shape is neither in the
trace nor well defined. But N and K are fixed by the model config and M is the
token count, so a module's total GEMM FLOPs is exactly known; dividing by the
module's measured GPU time gives a correct projection-level MFU. This is a
post-processor: run it on the core-compute CSV after build_static_analysis_tables.
It only fills modules in a per-architecture catalog; sparse/non-GEMM modules and
unknown models are left untouched (blank), never guessed.
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def _dims(cfg: dict) -> dict:
    return cfg.get("text_config", cfg)


def dsv41_catalog(cfg: dict) -> dict:
    """functional_module -> {dtype: [(K, N), ...]} of dense sub-GEMMs, per token.

    Verified against sglang/srt/models/deepseek_v4.py + deepseek_v2.py and the
    known moe/experts shape (M=9216,N=6912,K=5120). hc mixing dims come from the
    deep_gemm template sm100_tf32_hc_prenorm_gemm_impl<24,20480,...>.
    """
    t = _dims(cfg)
    H = int(t["hidden_size"])
    q_lora = int(t["q_lora_rank"])
    head_dim = int(t["head_dim"])
    n_heads = int(t["num_attention_heads"])
    moe_i = int(t["moe_intermediate_size"])
    o_lora = int(t["o_lora_rank"])
    o_groups = int(t.get("o_groups", 8))
    hc_k = int(t.get("hc_mult", 4)) * H
    qkv_out = n_heads * head_dim
    return {
        "Attention 输入与投影": {
            "fp8": [(H, q_lora + head_dim), (q_lora, qkv_out)],
            "tf32": [(hc_k, 24)],
        },
        "Attention 输出与投影": {
            "fp8": [(qkv_out, o_lora), (o_lora * o_groups, H)],
            "tf32": [(hc_k, 24)],
        },
        "MoE 共享专家": {
            "fp8": [(H, 2 * moe_i), (moe_i, H)],
        },
    }


def resolve_tokens(cfg: dict, stage: str, batch: int | None, chunk: int | None,
                   explicit: int | None) -> int:
    if explicit:
        return explicit
    t = _dims(cfg)
    if stage == "prefill" and chunk:
        return chunk
    if stage == "decode" and batch:
        # verify processes block_size+1 tokens per sequence under DSPARK/MTP
        return batch * (int(t.get("dspark_block_size", 0)) + 1 or 1)
    raise SystemExit("cannot resolve M: pass --tokens (or --stage + --batch-size/--chunk-size)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--core-table", type=Path, required=True)
    p.add_argument("--model-config", type=Path, required=True)
    p.add_argument("--tokens", type=int, help="M; else derived from stage+batch/chunk")
    p.add_argument("--stage", choices=["prefill", "decode"], default="decode")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--chunk-size", type=int)
    p.add_argument("--peak-fp8-tflops", type=float, default=4500.0)
    p.add_argument("--peak-tf32-tflops", type=float, default=1125.0)
    p.add_argument("--units", type=int, default=0, help="forwards represented; 0=auto")
    args = p.parse_args()

    cfg = json.loads(args.model_config.read_text())
    model = _dims(cfg).get("model_type", "") + str(cfg.get("architectures", ""))
    if "deepseek_v4" not in model.lower() and "deepseekv4" not in model.lower():
        print(f"[core-shapes] no catalog for this model; leaving shapes unchanged")
        return
    catalog = dsv41_catalog(cfg)
    H = int(_dims(cfg)["hidden_size"])
    M = resolve_tokens(cfg, args.stage, args.batch_size, args.chunk_size, args.tokens)

    with args.core_table.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = reader.fieldnames or []
        rows = list(reader)
    data = [r for r in rows if (r.get("序号") or "").strip().isdigit()]
    units = args.units or sum(1 for r in data if "experts" in (r.get("module") or "")) or 1

    filled = 0
    for module, dtypes in catalog.items():
        mrows = [r for r in data if (r.get("功能模块") or "") == module]
        dur_us = sum(float(r.get("算子耗时(us)") or 0) for r in mrows)
        if not mrows or dur_us <= 0:
            continue
        flops = weighted_peak = sum_nk = 0.0
        for dtype, gemms in dtypes.items():
            peak = args.peak_fp8_tflops if dtype == "fp8" else args.peak_tf32_tflops
            for K, N in gemms:
                sum_nk += K * N
                flops += 2.0 * M * K * N * units
                weighted_peak += 2.0 * M * K * N * units * peak * 1e12
        eff_peak = weighted_peak / flops if flops else args.peak_fp8_tflops * 1e12
        mfu = flops / (dur_us * 1e-6 * eff_peak) * 100.0
        shape_txt = f"(M={M},N={round(sum_nk / H)},K={H})"
        for r in mrows:
            r["shape"] = shape_txt
            r["mfu"] = f"{mfu:.2f}%"
            filled += 1
        print(f"[core-shapes] {module}: {shape_txt} MFU={mfu:.2f}% ({dur_us / units:.1f}us/步)")

    with args.core_table.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[core-shapes] filled {filled} rows across {len(catalog)} projection modules (M={M})")


if __name__ == "__main__":
    main()
