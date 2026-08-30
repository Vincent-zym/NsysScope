#!/usr/bin/env python3
"""Regenerate ``references/fusion-registry.json`` from the upstream community skill.

The registry is imported, not hand-maintained: the source of truth is
``FUSION_PATTERN_REGISTRY`` inside sgl-project/sglang's
``.claude/skills/llm-torch-profiler-analysis/scripts/triage_kernel_helpers.py``.
Re-run this after pulling a newer SGLang checkout so pattern keywords, share
thresholds and subsumption rules stay aligned with upstream.

The extraction is done with ``ast.literal_eval`` on the module's AST, so nothing
from the upstream file is executed and no upstream imports are required.

Usage:
    python3 scripts/refresh_fusion_registry.py \\
        --sglang-root /path/to/sglang \\
        [--out references/fusion-registry.json]
"""

from __future__ import annotations

import argparse
import ast
import datetime as dt
import json
import os
import subprocess
import sys
from typing import List, Optional, Sequence

SKILL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(SKILL_ROOT, "references", "fusion-registry.json")
UPSTREAM_RELPATH = os.path.join(
    ".claude", "skills", "llm-torch-profiler-analysis", "scripts",
    "triage_kernel_helpers.py",
)

# Chinese labels for the frontend and for Chinese-language advisor output. Any
# pattern missing here falls back to its English name, so a newly added upstream
# pattern still works -- it just shows untranslated until a label is added.
PATTERN_ZH = {
    "Fused residual add + RMSNorm": "残差加 + RMSNorm 融合",
    "FlashInfer unified allreduce_fusion": "FlashInfer 统一 allreduce 融合（AllReduce + 残差 + RMSNorm）",
    "AITER allreduce fusion": "AITER（ROCm）AllReduce + RMSNorm 融合",
    "Fused activation-and-mul (SwiGLU / GeGLU)": "激活与逐元素乘融合（SwiGLU / GeGLU）",
    "In-place QK RMSNorm": "Q/K RMSNorm 原地融合",
    "Fused QK RMSNorm + RoPE": "Q/K RMSNorm + RoPE 融合",
    "Fused QK RoPE reshape + KV cache write": "Q/K RoPE + reshape + KV cache 写入融合",
    "Fused RoPE + KV cache store": "RoPE + KV cache 存储融合",
    "Fused decode metadata setup": "decode 元数据准备融合",
    "NSA fused metadata copy for graph replay": "NSA 图重放元数据拷贝融合",
    "DeepSeek MLA fused projection + norm + RoPE": "DeepSeek MLA 投影 + norm + RoPE 融合",
    "Fused QK RoPE concat + MLA cache write": "Q/K RoPE 拼接 + MLA cache 写入融合",
    "Qwen3 decode fused QK norm + 3D mRoPE + KV cache write":
        "Qwen3 decode Q/K norm + 3D mRoPE + KV cache 写入融合",
    "Fused MoE router / top-k / softcapping": "MoE router / top-k / softcap 融合",
    "Fused MoE grouped-topk / gate kernels": "MoE grouped-topk / gate 融合",
    "Qwen-style shared-expert append into routed top-k output":
        "Qwen 风格共享专家并入 routed top-k 输出",
    "Fused MoE sum + all-reduce": "MoE 求和 + AllReduce 融合",
    "Fused MoE activation + quant / re-quant": "MoE 激活 + 量化 / 重量化融合",
    "DeepSeek comm-prep fused RMSNorm + quant / flatten-quant":
        "DeepSeek 通信前 RMSNorm + 量化融合",
    "NSA fused top-k transform / page-table build": "NSA top-k 变换 / page-table 构建融合",
    "NSA fused quantize + indexed K-cache store": "NSA 量化 + 索引 K-cache 写入融合",
    "Fused sampling temperature + softmax": "采样温度 + softmax 融合",
    "Fused logit softcap": "logit softcap 融合",
    "PR #20667 Qwen3.5 fused QK norm + RoPE + KV cache write":
        "PR #20667 Qwen3.5 Q/K norm + RoPE + KV cache 写入融合",
    "PR #22392 CUTLASS FP8 scaled MM replacing nvjet":
        "PR #22392 CUTLASS FP8 scaled MM 替换 nvjet",
    "SGLang LTX2 fused Ada values": "SGLang LTX2 Ada 值融合",
    "SGLang LTX2 residual-gate add CUDA fast path": "SGLang LTX2 residual-gate 加法 CUDA 快路径",
    "TokenSpeed CuTe DSL MLA prefill / decode": "TokenSpeed CuTe DSL MLA prefill / decode 融合",
    "TokenSpeed MLA KV pack + FP8 quantize": "TokenSpeed MLA KV 打包 + FP8 量化融合",
    "TokenSpeed fused top-k + top-p sampling": "TokenSpeed top-k + top-p 采样融合",
    "TokenSpeed persistent lm_head GEMM": "TokenSpeed 常驻 lm_head GEMM",
    "TokenSpeed NVFP4 GEMM + SwiGLU + quant": "TokenSpeed NVFP4 GEMM + SwiGLU + 量化融合",
    "vLLM-origin Attention + Quantization": "vLLM 源流：attention + 量化融合",
    "vLLM-origin DSV3.2 fused indexer projections": "vLLM 源流：DSV3.2 indexer 投影融合",
    "vLLM-origin RMSNorm + Quantization": "vLLM 源流：RMSNorm + 量化融合",
    "vLLM-origin SiLU+Mul + Quantization": "vLLM 源流：SiLU+Mul + 量化融合",
    "vLLM-origin DSV3 router GEMM": "vLLM 源流：DSV3 router GEMM",
    "vLLM-origin GPT-OSS router GEMM": "vLLM 源流：GPT-OSS router GEMM",
    "vLLM-origin DeepSeek min-latency fused QKV-A projection":
        "vLLM 源流：DeepSeek 低延迟 QKV-A 投影融合",
    "PR #38621 fused QK norm + RoPE + cache + quant":
        "PR #38621 Q/K norm + RoPE + cache + 量化融合",
    "vLLM-origin MiniMax allreduce_rms kernels": "vLLM 源流：MiniMax allreduce_rms 融合",
    "vLLM fused residual add + RMSNorm": "vLLM 残差加 + RMSNorm 融合",
    "vLLM fused activation-and-mul": "vLLM 激活与逐元素乘融合",
    "TensorRT-LLM FlashInfer residual add + RMSNorm":
        "TensorRT-LLM FlashInfer 残差加 + RMSNorm 融合",
    "TensorRT-LLM Triton fused residual add + RMSNorm + FP8 quant":
        "TensorRT-LLM Triton 残差加 + RMSNorm + FP8 量化融合",
    "TensorRT-LLM FlashInfer RMSNorm family": "TensorRT-LLM FlashInfer RMSNorm 家族",
    "TensorRT-LLM FlashInfer activation / gate epilogues":
        "TensorRT-LLM FlashInfer 激活 / gate epilogue",
}


def extract_specs(upstream_path: str) -> List[dict]:
    tree = ast.parse(open(upstream_path, encoding="utf-8").read())
    node = None
    for stmt in tree.body:
        targets = getattr(stmt, "targets", None) or (
            [stmt.target] if hasattr(stmt, "target") else []
        )
        for target in targets:
            if isinstance(target, ast.Name) and target.id == "FUSION_PATTERN_REGISTRY":
                node = stmt.value
    if node is None:
        raise SystemExit(f"FUSION_PATTERN_REGISTRY not found in {upstream_path}")
    specs = []
    for call in node.elts:
        if not isinstance(call, ast.Call):
            raise SystemExit("unexpected registry element; upstream format changed")
        specs.append({kw.arg: ast.literal_eval(kw.value) for kw in call.keywords})
    return specs


def make_id(pattern: str) -> str:
    slug = "".join(c.lower() if c.isalnum() else "_" for c in pattern)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def path_present(sglang_root: str, path: str) -> bool:
    return os.path.exists(os.path.join(sglang_root, path.split("::", 1)[0]))


def git_head(root: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", required=True,
                        help="local sgl-project/sglang checkout containing .claude/skills")
    parser.add_argument("--out", default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    upstream = os.path.join(args.sglang_root, UPSTREAM_RELPATH)
    if not os.path.exists(upstream):
        raise SystemExit(f"upstream registry not found: {upstream}")

    entries = []
    for spec in extract_specs(upstream):
        pattern = spec["pattern"]
        paths = [p.strip() for p in spec.get("candidate_path", "").split("<br>") if p.strip()]
        entries.append(
            {
                "id": make_id(pattern),
                "pattern": pattern,
                "patternZh": PATTERN_ZH.get(pattern, pattern),
                "origin": spec.get("origin", "mainline"),
                "activeKeywords": list(spec.get("active_keywords", ())),
                "splitGroups": [list(g) for g in spec.get("split_groups", ())],
                "candidatePaths": paths,
                "candidatePathsPresentLocally": [path_present(args.sglang_root, p) for p in paths],
                "minSharePct": spec.get("min_share", 0.25),
                "likelySharePct": spec.get("likely_share", 3.0),
                "priority": spec.get("priority", 0),
                "subsumes": list(spec.get("subsumes", ())),
                "modelInclude": list(spec.get("model_include", ())),
                "modelExclude": list(spec.get("model_exclude", ())),
                "requireTp": spec.get("require_tp", False),
                "minTpSize": spec.get("min_tp_size", 1),
                "rationaleHint": spec.get("rationale_hint", ""),
            }
        )

    doc = {
        "schemaVersion": "1.0",
        "provenance": {
            "derivedFrom": f"sgl-project/sglang {UPSTREAM_RELPATH}::FUSION_PATTERN_REGISTRY",
            "importedAt": dt.date.today().isoformat(),
            # Only the checkout's directory name, never its absolute path: this
            # registry is committed and shared, and whose home it was imported
            # from is not provenance anyone can act on. `sglangHead` is the part
            # that identifies the upstream state.
            "sglangCheckout": os.path.basename(os.path.abspath(args.sglang_root)),
            "sglangHead": git_head(args.sglang_root),
            "note": (
                "candidatePathsPresentLocally is a snapshot of the checkout at import "
                "time; scan_fusion_candidates.py re-resolves paths by basename against "
                "the analysed source tree, so upstream file moves do not break matching."
            ),
        },
        "patterns": entries,
    }

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=2)
        fh.write("\n")

    present = sum(1 for e in entries for ok in e["candidatePathsPresentLocally"] if ok)
    total = sum(len(e["candidatePathsPresentLocally"]) for e in entries)
    untranslated = [e["pattern"] for e in entries if e["pattern"] == e["patternZh"]]
    print(f"wrote {len(entries)} patterns -> {args.out}")
    print(f"candidate paths present in checkout: {present}/{total}")
    if untranslated:
        print(f"missing Chinese labels ({len(untranslated)}): {untranslated}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
