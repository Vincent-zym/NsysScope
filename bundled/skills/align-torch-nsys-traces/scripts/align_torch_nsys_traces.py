#!/usr/bin/env python3
"""Align torch profiler trace and nsys report at forward/layer/kernel level."""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path


def load_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def load_config(path: Path | None) -> dict:
    if not path:
        return {}
    with path.open() as f:
        return json.load(f)


def read_csv(path: Path | None) -> list[dict]:
    if not path:
        return []
    with path.open(newline="", errors="replace") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        if not rows:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def kernel_family(name: str) -> str:
    rules = [
        ("per_token_group_quant_8bit_kernel", "per_token_group_quant"),
        ("sm100_fp8_fp4_gemm_1d1d_impl", "deep_gemm_1d1d"),
        ("sm100_fp8_fp4_mega_moe_impl", "deep_gemm_mega_moe"),
        ("sm100_tf32_hc_prenorm_gemm_impl", "hc_prenorm_gemm"),
        ("transpose_and_pack_fp32_into_ue8m0", "deep_gemm_transpose_pack"),
        ("mhc_pre_big_fuse_with_norm_tilelang", "mhc_pre"),
        ("mhc_post_tilelang", "mhc_post"),
        ("flash_fwd_splitkv_mla", "flash_mla_splitkv"),
        ("flash_fwd_mla_combine", "flash_mla_combine"),
        ("fused_q_norm_rope", "fused_q_norm_rope"),
        ("fused_k_norm_rope", "fused_k_norm_rope"),
        ("fused_norm_rope_flashmla", "fused_norm_rope_flashmla"),
        ("fused_norm_rope_indexer", "fused_norm_rope_indexer"),
        ("fused_q_indexer_rope_hadamard_quant", "fused_q_indexer_rope_hadamard_quant"),
        ("deepseek_rope_kernel", "deepseek_rope"),
        ("all_reduce_one_shot_push", "all_reduce_push"),
        ("ncclDevKernel_AllGather", "nccl_allgather"),
        ("splitKreduce_kernel", "cublas_splitk_reduce"),
        ("RMSNormKernel", "rmsnorm"),
        ("rmsnorm", "rmsnorm"),
        ("moe_hash_topk_fused", "moe_hash_topk_fused"),
        ("topk_512_transform", "moe_hash_topk_fused"),
        ("topk_fused_transform", "moe_hash_topk_fused"),
        ("moe_fused_gate_kernel", "moe_fused_gate"),
        ("mega_moe_pre_dispatch_kernel", "mega_moe_pre_dispatch"),
        ("silu_mul_clamp_kernel", "silu_mul_clamp"),
        ("CatArrayBatchedCopy", "cat_array_batched_copy"),
        ("index_elementwise_kernel", "index_elementwise"),
        ("unrolled_elementwise_kernel", "unrolled_elementwise"),
        ("vectorized_elementwise_kernel", "vectorized_elementwise"),
        ("elementwise_kernel_with_index", "elementwise_with_index"),
        ("elementwise_kernel", "elementwise"),
        ("distribution_elementwise_grid_stride_kernel", "distribution_elementwise"),
        ("indexSelectSmallIndex", "index_select_small"),
        ("clamp_position_kernel", "clamp_position"),
    ]
    for substring, family in rules:
        if substring in name:
            return family
    if "nvjet_sm100_tss" in name or "nvjet_sm100_tst" in name:
        return "nvjet_tss_tst"
    if name.startswith("triton_"):
        return name
    match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)(?:<|\()", name)
    return match.group(1) if match else name[:80]


def normalize_name(name: str) -> str:
    name = name.replace("(anonymous namespace)", "<unnamed>").replace("anonymous namespace", "<unnamed>")
    name = re.sub(r"\((?:int|unsigned int|unsigned long|bool)\)", "", name)
    return re.sub(r"\s+", " ", name).strip()


def layer_type(layer_id: int, compress_ratios: list, num_layers: int, num_hash_layers: int) -> tuple[str, str]:
    cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else ""
    hash_start = num_layers - num_hash_layers if num_hash_layers else num_layers
    if layer_id == 0:
        return "FIRST", str(cr)
    if layer_id >= hash_start:
        return "HASH", str(cr)
    if cr == 0:
        return "FULL_ATTN", str(cr)
    if cr == 4:
        return "C4_LIGHT", str(cr)
    if cr == 128:
        return "C128_HEAVY", str(cr)
    return f"CR{cr}", str(cr)


def split_torch(args, config: dict) -> tuple[list[dict], list[dict], dict]:
    events = load_trace(args.torch_trace)
    gpu = sorted([e for e in events if "kernel" in str(e.get("cat", "")).lower()], key=lambda e: e.get("ts", 0))
    anchors = [i for i, e in enumerate(gpu) if args.anchor in e.get("name", "")]
    num_layers = args.num_layers or config.get("num_hidden_layers")
    if not num_layers:
        raise SystemExit("num_layers is required when config has no num_hidden_layers")
    blocks_per_forward = num_layers * args.blocks_per_layer
    n_forwards = (len(anchors) - 1) // blocks_per_forward
    if n_forwards <= 0:
        raise SystemExit(
            f"no complete torch forward found: anchors={len(anchors)}, "
            f"required_per_forward={blocks_per_forward}, anchor={args.anchor!r}"
        )
    torch_fwd = args.torch_fwd_pass if args.torch_fwd_pass is not None else min(3, max(0, n_forwards - 1))
    if torch_fwd < 0 or torch_fwd >= n_forwards:
        raise SystemExit(f"torch fwd pass {torch_fwd} out of range; detected forward passes: {n_forwards}")
    base = torch_fwd * blocks_per_forward
    mapping = read_csv(args.torch_mapping)
    compress_ratios = config.get("compress_ratios") or []
    num_hash_layers = int(config.get("num_hash_layers") or 0)
    trace_start = min((e.get("ts", 0) for e in events if e.get("ts") is not None), default=0)

    layers = []
    kernels = []
    for lid in range(num_layers):
        layer_start_anchor = base + lid * args.blocks_per_layer
        next_layer_anchor = base + (lid + 1) * args.blocks_per_layer
        si = anchors[layer_start_anchor]
        ei = anchors[next_layer_anchor]
        selected = gpu[si:ei]
        ltype, cr = layer_type(lid, compress_ratios, num_layers, num_hash_layers)
        start_us = selected[0].get("ts", 0)
        end_us = selected[-1].get("ts", 0) + selected[-1].get("dur", 0)
        layers.append(
            {
                "fwd_pass": torch_fwd,
                "layer_id": lid,
                "layer_type": ltype,
                "compress_ratio": cr,
                "wall_start_us": start_us,
                "wall_end_us": end_us,
                "wall_ms": (end_us - start_us) / 1000,
                "sum_kernel_dur_ms": sum(k.get("dur", 0) for k in selected) / 1000,
                "kernel_count": len(selected),
                "anchor_kernel": args.anchor,
                "blocks_per_layer": args.blocks_per_layer,
            }
        )
        for order, k in enumerate(selected):
            idx = si + order
            m = mapping[idx] if idx < len(mapping) else {}
            kernels.append(
                {
                    "fwd_pass": torch_fwd,
                    "layer_id": lid,
                    "layer_type": ltype,
                    "compress_ratio": cr,
                    "kernel_order_in_layer": order,
                    "global_kernel_idx": idx,
                    "trace_ts_us": k.get("ts", 0),
                    "trace_dur_us": k.get("dur", 0),
                    "trace_rel_ms": (k.get("ts", 0) - trace_start) / 1000,
                    "kernel_name": k.get("name", ""),
                    "stream": m.get("Stream", ""),
                    "op": m.get("Op", ""),
                    "python_file_line": m.get("Python file:line", ""),
                    "configured_python_frames": m.get("Configured Python frames", ""),
                    "python_call_stack_file_lines": m.get("Python call stack file:lines", ""),
                }
            )
    meta = {
        "torch_trace": str(args.torch_trace),
        "torch_fwd_pass": torch_fwd,
        "detected_forward_passes": n_forwards,
        "gpu_kernel_rows": len(gpu),
        "anchor_blocks": len(anchors),
        "num_layers": num_layers,
        "blocks_per_layer": args.blocks_per_layer,
        "selected_forward_kernel_rows": len(kernels),
    }
    return layers, kernels, meta


def export_nsys_sqlite(nsys_report: Path, output_dir: Path) -> Path:
    if nsys_report.suffix == ".sqlite":
        return nsys_report
    sibling_sqlite = nsys_report.with_suffix(".sqlite")
    if sibling_sqlite.exists():
        return sibling_sqlite
    sqlite_path = output_dir / "nsys" / (nsys_report.stem + ".sqlite")
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    if not sqlite_path.exists():
        subprocess.run(
            ["nsys", "export", "--type", "sqlite", "--output", str(sqlite_path), str(nsys_report)],
            check=True,
        )
    return sqlite_path


def discover_nsys_report(path: Path) -> Path | None:
    if path.is_file() and path.suffix in {".sqlite", ".nsys-rep"}:
        return path
    if path.is_dir():
        sqlite_files = sorted(path.rglob("*.sqlite"), key=lambda p: p.stat().st_mtime, reverse=True)
        if sqlite_files:
            return sqlite_files[0]
        nsys_files = sorted(path.rglob("*.nsys-rep"), key=lambda p: p.stat().st_mtime, reverse=True)
        if nsys_files:
            return nsys_files[0]
        return None
    return None


def write_torch_only_outputs(args, torch_layers: list[dict], torch_kernels: list[dict], torch_meta: dict, reason: str) -> None:
    write_csv(args.output_dir / "torch/selected_forward_layers_summary.csv", torch_layers)
    write_csv(args.output_dir / "torch/selected_forward_kernels_with_python.csv", torch_kernels)
    meta = {
        "mode": "torch_only",
        "reason": reason,
        "torch": torch_meta,
        "nsys": None,
        "kernel_launch_status_counts": {},
        "matched_count": 0,
        "kernel_launch_rows": 0,
        "match_rate": None,
    }
    align_dir = args.output_dir / "alignment"
    align_dir.mkdir(parents=True, exist_ok=True)
    (align_dir / "kernel_launch_alignment_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    (args.output_dir / "README.md").write_text(
        "# Torch/Nsys Trace Alignment\n\n"
        "Status: torch-only fallback\n\n"
        f"- reason: `{reason}`\n"
        f"- torch trace: `{args.torch_trace}`\n"
        f"- requested nsys report: `{args.nsys_report}`\n"
        f"- anchor: `{args.anchor}`\n"
        f"- torch forward: `{torch_meta['torch_fwd_pass']}`\n"
        "- nsys alignment: not run\n"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


def load_nsys_kernels(sqlite_path: Path, device: int) -> list[tuple[int, int, int, str]]:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    query = """
        select k.start, k.end, k.streamId, s.value
        from CUPTI_ACTIVITY_KIND_KERNEL k
        join StringIds s on k.demangledName = s.id
        where k.deviceId = ?
        order by k.start, k.end
    """
    return list(cur.execute(query, (device,)))


def choose_nsys_device(sqlite_path: Path, anchor: str, anchors_per_forward: int) -> int:
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    rows = list(
        cur.execute(
            """
            select k.deviceId, count(*)
            from CUPTI_ACTIVITY_KIND_KERNEL k
            join StringIds s on k.demangledName = s.id
            where s.value like ?
            group by k.deviceId
            order by k.deviceId
            """,
            (f"%{anchor}%",),
        )
    )
    if not rows:
        raise SystemExit(f"no nsys anchor kernels found for {anchor}")
    exact = [r for r in rows if r[1] % anchors_per_forward == 0]
    return max(exact or rows, key=lambda r: r[1])[0]


def split_nsys(args, sqlite_path: Path, torch_layers: list[dict]) -> tuple[list[dict], list[dict], dict]:
    num_layers = len(torch_layers)
    anchors_per_forward = num_layers * args.blocks_per_layer
    device = args.nsys_device
    if device is None:
        device = choose_nsys_device(sqlite_path, args.anchor, anchors_per_forward)
    rows = load_nsys_kernels(sqlite_path, device)
    anchors = [i for i, row in enumerate(rows) if args.anchor in row[3]]
    complete = (len(anchors) - 1) // anchors_per_forward

    torch_counts = [int(r["kernel_count"]) for r in torch_layers]
    if args.nsys_forward is not None:
        nsys_forward = args.nsys_forward
    else:
        candidates = []
        for fwd in range(complete):
            base = fwd * anchors_per_forward
            counts = []
            for lid in range(num_layers):
                counts.append(anchors[base + (lid + 1) * args.blocks_per_layer] - anchors[base + lid * args.blocks_per_layer])
            diff = sum(abs(a - b) for a, b in zip(counts, torch_counts))
            candidates.append((diff, abs(fwd - 16), fwd))
        nsys_forward = sorted(candidates)[0][2]

    base = nsys_forward * anchors_per_forward
    layers = []
    kernels = []
    for lid, torch_layer in enumerate(torch_layers):
        si = anchors[base + lid * args.blocks_per_layer]
        ei = anchors[base + (lid + 1) * args.blocks_per_layer]
        selected = rows[si:ei]
        start_ns = selected[0][0]
        end_ns = selected[-1][1]
        layers.append(
            {
                "nsys_device": device,
                "nsys_forward": nsys_forward,
                "layer_id": lid,
                "layer_type": torch_layer["layer_type"],
                "compress_ratio": torch_layer["compress_ratio"],
                "torch_kernel_count": torch_layer["kernel_count"],
                "nsys_kernel_count": len(selected),
                "count_diff": len(selected) - int(torch_layer["kernel_count"]),
                "wall_ms": (end_ns - start_ns) / 1e6,
                "start_ns": start_ns,
                "end_ns": end_ns,
            }
        )
        for order, row in enumerate(selected):
            kernels.append(
                {
                    "nsys_device": device,
                    "nsys_forward": nsys_forward,
                    "layer_id": lid,
                    "layer_type": torch_layer["layer_type"],
                    "compress_ratio": torch_layer["compress_ratio"],
                    "kernel_order_in_layer": order,
                    "start_ns": row[0],
                    "end_ns": row[1],
                    "duration_us": (row[1] - row[0]) / 1000,
                    "stream": row[2],
                    "kernel_name": row[3],
                }
            )
    meta = {
        "sqlite_path": str(sqlite_path),
        "nsys_device": device,
        "nsys_forward": nsys_forward,
        "device_kernel_rows": len(rows),
        "anchor_blocks": len(anchors),
        "complete_forward_floor": complete,
        "selected_forward_kernel_rows": len(kernels),
    }
    return layers, kernels, meta


def align_outputs(torch_layers, torch_kernels, nsys_layers, nsys_kernels) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    layer_alignment = []
    for t, n in zip(torch_layers, nsys_layers):
        layer_alignment.append(
            {
                "layer_id": t["layer_id"],
                "layer_type": t["layer_type"],
                "compress_ratio": t["compress_ratio"],
                "torch_kernel_count": t["kernel_count"],
                "nsys_kernel_count": n["nsys_kernel_count"],
                "count_diff": n["count_diff"],
                "torch_wall_ms": t["wall_ms"],
                "nsys_wall_ms": n["wall_ms"],
                "torch_sum_kernel_dur_ms": t["sum_kernel_dur_ms"],
            }
        )

    tk_by_layer = collections.defaultdict(list)
    nk_by_layer = collections.defaultdict(list)
    for row in torch_kernels:
        tk_by_layer[int(row["layer_id"])].append(row)
    for row in nsys_kernels:
        nk_by_layer[int(row["layer_id"])].append(row)

    family_overlap = []
    launch_rows = []
    launch_summary = []
    for lid in sorted(tk_by_layer):
        tks = sorted(tk_by_layer[lid], key=lambda r: int(r["kernel_order_in_layer"]))
        nks = sorted(nk_by_layer[lid], key=lambda r: int(r["kernel_order_in_layer"]))
        tc = collections.Counter(kernel_family(r["kernel_name"]) for r in tks)
        nc = collections.Counter(kernel_family(r["kernel_name"]) for r in nks)
        shared = sum((tc & nc).values())
        union = sum((tc | nc).values())
        family_overlap.append(
            {
                "layer_id": lid,
                "layer_type": tks[0]["layer_type"],
                "torch_family_total": sum(tc.values()),
                "nsys_family_total": sum(nc.values()),
                "shared_multiset_count": shared,
                "union_multiset_count": union,
                "multiset_jaccard": shared / union if union else 1.0,
                "torch_top_families": ";".join(f"{k}:{v}" for k, v in tc.most_common(8)),
                "nsys_top_families": ";".join(f"{k}:{v}" for k, v in nc.most_common(8)),
            }
        )
        status_counts = collections.Counter()
        for tk, nk in zip(tks, nks):
            tf = kernel_family(tk["kernel_name"])
            nf = kernel_family(nk["kernel_name"])
            exact = normalize_name(tk["kernel_name"]) == normalize_name(nk["kernel_name"])
            status = "same_order_exact_name" if exact else ("same_order_family" if tf == nf else "same_order_mismatch")
            status_counts[status] += 1
            launch_rows.append(
                {
                    "layer_id": lid,
                    "layer_type": tk["layer_type"],
                    "compress_ratio": tk["compress_ratio"],
                    "torch_order": tk["kernel_order_in_layer"],
                    "nsys_order": nk["kernel_order_in_layer"],
                    "order_delta": int(nk["kernel_order_in_layer"]) - int(tk["kernel_order_in_layer"]),
                    "status": status,
                    "torch_family": tf,
                    "nsys_family": nf,
                    "torch_kernel_name": tk["kernel_name"],
                    "nsys_kernel_name": nk["kernel_name"],
                    "torch_start_us": tk["trace_ts_us"],
                    "torch_end_us": float(tk["trace_ts_us"]) + float(tk["trace_dur_us"]),
                    "nsys_start_ns": nk["start_ns"],
                    "nsys_end_ns": nk["end_ns"],
                    "torch_dur_us": tk["trace_dur_us"],
                    "nsys_dur_us": nk["duration_us"],
                    "torch_stream": tk.get("stream", ""),
                    "nsys_stream": nk.get("stream", ""),
                    "torch_op": tk.get("op", ""),
                    "torch_python_file_line": tk.get("python_file_line", ""),
                    "torch_configured_python_frames": tk.get("configured_python_frames", ""),
                    "torch_python_call_stack_file_lines": tk.get("python_call_stack_file_lines", ""),
                }
            )
        matched = status_counts["same_order_exact_name"] + status_counts["same_order_family"]
        launch_summary.append(
            {
                "layer_id": lid,
                "layer_type": tks[0]["layer_type"],
                "torch_kernel_count": len(tks),
                "nsys_kernel_count": len(nks),
                "matched_count": matched,
                "same_order_exact_name": status_counts["same_order_exact_name"],
                "same_order_family": status_counts["same_order_family"],
                "same_order_mismatch": status_counts["same_order_mismatch"],
                "match_rate": matched / len(tks) if tks else 0,
            }
        )
    return layer_alignment, family_overlap, launch_rows, launch_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-trace", required=True, type=Path)
    parser.add_argument("--nsys-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--torch-mapping", type=Path)
    parser.add_argument("--anchor", default="mhc_post_tilelang")
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--blocks-per-layer", type=int, default=2)
    parser.add_argument("--torch-fwd-pass", type=int)
    parser.add_argument("--nsys-device", type=int)
    parser.add_argument("--nsys-forward", type=int)
    parser.add_argument(
        "--strict-nsys",
        action="store_true",
        help="Fail instead of producing torch-only outputs when no .nsys-rep/.sqlite is found.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    torch_layers, torch_kernels, torch_meta = split_torch(args, config)
    discovered_nsys = discover_nsys_report(args.nsys_report)
    if discovered_nsys is None:
        reason = f"no .nsys-rep or .sqlite found at {args.nsys_report}"
        if args.strict_nsys:
            raise SystemExit(reason)
        print(f"warning: {reason}; writing torch-only outputs", file=sys.stderr)
        write_torch_only_outputs(args, torch_layers, torch_kernels, torch_meta, reason)
        return
    sqlite_path = export_nsys_sqlite(discovered_nsys, args.output_dir)
    nsys_layers, nsys_kernels, nsys_meta = split_nsys(args, sqlite_path, torch_layers)
    layer_alignment, family_overlap, launch_rows, launch_summary = align_outputs(
        torch_layers, torch_kernels, nsys_layers, nsys_kernels
    )

    write_csv(args.output_dir / "torch/selected_forward_layers_summary.csv", torch_layers)
    write_csv(args.output_dir / "torch/selected_forward_kernels_with_python.csv", torch_kernels)
    write_csv(args.output_dir / "nsys/selected_forward_layers_summary.csv", nsys_layers)
    write_csv(args.output_dir / "nsys/selected_forward_kernels.csv", nsys_kernels)
    write_csv(args.output_dir / "alignment/layer_alignment.csv", layer_alignment)
    write_csv(args.output_dir / "alignment/layer_family_overlap.csv", family_overlap)
    write_csv(args.output_dir / "alignment/kernel_launch_alignment.csv", launch_rows)
    write_csv(args.output_dir / "alignment/kernel_launch_alignment_summary_by_layer.csv", launch_summary)

    status_counts = collections.Counter(row["status"] for row in launch_rows)
    meta = {
        "mode": "full_alignment",
        "torch": torch_meta,
        "nsys": nsys_meta,
        "kernel_launch_status_counts": dict(status_counts),
        "matched_count": status_counts["same_order_exact_name"] + status_counts["same_order_family"],
        "kernel_launch_rows": len(launch_rows),
        "match_rate": (status_counts["same_order_exact_name"] + status_counts["same_order_family"]) / len(launch_rows)
        if launch_rows
        else 0,
    }
    (args.output_dir / "alignment/kernel_launch_alignment_metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False)
    )
    (args.output_dir / "README.md").write_text(
        "# Torch/Nsys Trace Alignment\n\n"
        f"- torch trace: `{args.torch_trace}`\n"
        f"- nsys report: `{args.nsys_report}`\n"
        f"- anchor: `{args.anchor}`\n"
        f"- torch forward: `{torch_meta['torch_fwd_pass']}`\n"
        f"- nsys device/forward: `{nsys_meta['nsys_device']}` / `{nsys_meta['nsys_forward']}`\n"
        f"- kernel launch match rate: `{meta['match_rate']}`\n"
    )
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
