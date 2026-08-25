#!/usr/bin/env python3
"""Generate layer-aware call tree and Mermaid call graph from a torch profiler trace.

This is a best-effort correlator for Chrome-trace JSON exported by torch.profiler.
It preserves the existing layer-boundary model-profile logic and adds a hierarchy:

  Model -> Layer -> Submodule -> Python/CPU op -> ATen/Custom op -> CUDA API -> CUDA Kernel

Outputs are written under --output-dir using structured subdirectories:
  final_report.md
  call_tree/layer_call_tree.md
  call_graph/layer_call_graph.mmd
  call_graph/layer_call_graph_with_time.mmd
  mappings/kernel_to_layer.csv
  rankings/slowest_layers.csv
  rankings/slowest_submodules.csv
  rankings/slowest_kernels_by_layer.csv
"""

import argparse
import csv
import gzip
import html
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from model_profiles import ModelProfile, get_profile, infer_profile, normalize_compress_ratios


# ── loading and small helpers ──────────────────────────────────────────────


def load_trace(path: str) -> List[dict]:
    with gzip.open(path, "rt") if path.endswith(".gz") else open(path) as f:
        data = json.load(f)
    return data if isinstance(data, list) else data.get("traceEvents", [])


def load_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    with open(path) as f:
        return json.load(f)


def event_end(e: dict) -> float:
    return float(e.get("ts", 0)) + float(e.get("dur", 0) or 0)


def event_dur(e: dict) -> float:
    return float(e.get("dur", 0) or 0)


def fmt_us(us: float) -> str:
    if us >= 1000:
        return f"{us / 1000:.3f} ms"
    return f"{us:.1f} us"


def pct(part: float, total: float) -> str:
    return f"{part / total * 100:.2f}%" if total else "0.00%"


def safe_cell(s) -> str:
    s = "" if s is None else str(s)
    return s.replace("\n", " ").replace("|", "\\|")


def mermaid_text(s: str) -> str:
    return html.escape(str(s).replace('"', "'"), quote=False)


def get_arg(e: dict, *names):
    args = e.get("args") or {}
    for n in names:
        if n in args:
            return args[n]
    return None


def ext_id(e: dict):
    v = get_arg(e, "External id", "external id", "external_id", "External ID")
    return str(v) if v is not None else None


def corr_id(e: dict):
    v = get_arg(e, "correlation", "Correlation ID", "correlation_id")
    return str(v) if v is not None else None


def file_line_from_args(e: Optional[dict]) -> str:
    if not e:
        return ""
    args = e.get("args") or {}
    # Torch profiler variants may store one of these keys when with_stack=True.
    for key in ("File", "file", "Source Location", "source", "callsite", "Call site"):
        v = args.get(key)
        if v:
            return str(v)
    for key in ("Call stack", "Python stack", "Stack", "stack"):
        v = args.get(key)
        if not v:
            continue
        lines = v if isinstance(v, list) else str(v).split("\n")
        for line in lines:
            m = re.search(r"([^\s()]+\.py):(\d+)", str(line))
            if m:
                return f"{os.path.basename(m.group(1))}:{m.group(2)}"
    return ""


# ── profile / layer helpers ────────────────────────────────────────────────


def find_anchor_kernel(gpu_kernels: List[dict], profile: ModelProfile) -> str:
    if profile.anchor_kernel:
        return profile.anchor_kernel
    for c in ["mhc_post_tilelang", "flash_fwd_mla_combine", "AllReduce"]:
        if sum(1 for e in gpu_kernels if c in e.get("name", "")) >= 4:
            return c
    raise ValueError("Cannot auto-detect layer anchor kernel; specify --anchor-kernel.")


def detect_num_layers(anchor_indices, gpu_kernels, blocks_per_layer: int, default_num_layers: int = 1):
    n = min(200, len(anchor_indices) - 1)
    if n < 4:
        return default_num_layers
    durs = []
    for i in range(n):
        durs.append(sum(event_dur(gpu_kernels[j]) for j in range(anchor_indices[i], anchor_indices[i + 1])))
    best_period, best_score = None, 0
    for period in range(blocks_per_layer, min(120, n)):
        if period * 2 > n:
            break
        score = 0
        for i in range(min(period, n - period)):
            if durs[i] > 0 and durs[i + period] > 0:
                score += min(durs[i], durs[i + period]) / max(durs[i], durs[i + period])
        if score > best_score:
            best_period, best_score = period, score
    if best_period and best_score > best_period * 0.7:
        return best_period // blocks_per_layer
    return default_num_layers


def layer_type_label(layer_id: int, compress_ratios: List[int], num_layers: int, num_hash_layers: int) -> str:
    cr = compress_ratios[layer_id] if layer_id < len(compress_ratios) else -1
    if layer_id == 0:
        return "FIRST"
    if layer_id >= num_layers - num_hash_layers:
        return "HASH"
    if layer_id == num_layers - 1:
        return "FINAL"
    if cr == 0:
        return "FULL_ATTN"
    if cr == 4:
        return "C4_LIGHT"
    if cr == 128:
        return "C128_HEAVY"
    return f"CR{cr}"


def classify_kernel(name: str, profile: ModelProfile) -> Tuple[str, str]:
    for label, key, rule in profile.category_rules:
        if rule(name):
            return label, key
    return "Other", "other"


def infer_submodule(kernel_name: str, op_name: str, cat_key: str) -> str:
    text = f"{kernel_name} {op_name} {cat_key}".lower()
    if any(x in text for x in ["attention", "attn", "mla", "mqa", "flash", "paged"]):
        return "self_attn"
    if any(x in text for x in ["moe", "expert", "gemm", "ffn", "mlp", "silu", "router", "topk", "routing"]):
        return "mlp"
    if "norm" in text or "rms" in text:
        return "norm"
    if any(x in text for x in ["allreduce", "nccl", "reduce"]):
        return "comm"
    if any(x in text for x in ["rope", "rotary"]):
        return "rope"
    if any(x in text for x in ["quant", "fp8"]):
        return "quant"
    return "other"


# ── correlation helpers ───────────────────────────────────────────────────


def select_events(events: Iterable[dict], cats_or_names: Iterable[str]) -> List[dict]:
    keys = tuple(cats_or_names)
    out = []
    for e in events:
        cat = str(e.get("cat", ""))
        name = str(e.get("name", ""))
        if any(k in cat or k in name for k in keys):
            if e.get("ph") in (None, "X") and e.get("ts") is not None:
                out.append(e)
    return sorted(out, key=lambda e: (e.get("ts", 0), event_end(e)))


def build_correlation_maps(events: List[dict]):
    runtime_by_corr: Dict[str, dict] = {}
    cpu_by_ext: Dict[str, List[dict]] = defaultdict(list)
    runtime = []
    cpu_ops = []
    py_events = []
    for e in events:
        cat = str(e.get("cat", ""))
        name = str(e.get("name", ""))
        if "cuda_runtime" in cat or name.startswith("cuda"):
            runtime.append(e)
            c = corr_id(e)
            if c:
                runtime_by_corr[c] = e
        if "cpu_op" in cat or "Operator" in cat or name.startswith("aten::") or name.startswith("torch::"):
            cpu_ops.append(e)
            x = ext_id(e)
            if x:
                cpu_by_ext[x].append(e)
        if "python" in cat.lower() or "user_annotation" in cat or "record_function" in name.lower():
            py_events.append(e)
    return runtime_by_corr, cpu_by_ext, sorted(runtime, key=lambda e: e.get("ts", 0)), sorted(cpu_ops, key=lambda e: e.get("ts", 0)), sorted(py_events, key=lambda e: e.get("ts", 0))


def find_enclosing(events: List[dict], ts: float, end_ts: float, prefer_smallest=True) -> Optional[dict]:
    candidates = [e for e in events if e.get("ts", 0) <= ts and event_end(e) >= end_ts]
    if not candidates:
        # Fallback: containing start timestamp only.
        candidates = [e for e in events if e.get("ts", 0) <= ts <= event_end(e)]
    if not candidates:
        return None
    key = event_dur if prefer_smallest else (lambda e: -event_dur(e))
    return min(candidates, key=key)


def resolve_runtime_and_cpu(kernel: dict, runtime_by_corr, cpu_by_ext, runtime_events, cpu_ops):
    rt = None
    c = corr_id(kernel)
    if c and c in runtime_by_corr:
        rt = runtime_by_corr[c]
    if rt is None:
        rt = find_enclosing(runtime_events, kernel.get("ts", 0), event_end(kernel))
    cpu = None
    if rt is not None:
        x = ext_id(rt)
        if x and x in cpu_by_ext:
            # Usually the smallest enclosing CPU op with the same External id is the direct op.
            candidates = cpu_by_ext[x]
            cpu = find_enclosing(candidates, rt.get("ts", 0), event_end(rt)) or (candidates[-1] if candidates else None)
    if cpu is None:
        cpu = find_enclosing(cpu_ops, kernel.get("ts", 0), event_end(kernel))
    return rt, cpu


# ── tree model ─────────────────────────────────────────────────────────────


@dataclass
class Node:
    key: str
    label: str
    level: str
    file_line: str = ""
    inclusive_us: float = 0.0
    self_us: float = 0.0
    children: Dict[str, "Node"] = field(default_factory=dict)

    def child(self, key: str, label: str, level: str, file_line: str = "") -> "Node":
        if key not in self.children:
            self.children[key] = Node(key, label, level, file_line)
        elif file_line and not self.children[key].file_line:
            self.children[key].file_line = file_line
        return self.children[key]


def add_path(root: Node, path: List[Tuple[str, str, str, str]], dur_us: float):
    cur = root
    cur.inclusive_us += dur_us
    for key, label, level, file_line in path:
        cur = cur.child(key, label, level, file_line)
        cur.inclusive_us += dur_us
    cur.self_us += dur_us


def finalize_self(node: Node):
    for c in node.children.values():
        finalize_self(c)
    if node.children:
        child_sum = sum(c.inclusive_us for c in node.children.values())
        node.self_us = max(0.0, node.inclusive_us - child_sum)


# ── output ─────────────────────────────────────────────────────────────────


def write_csv(path: str, rows: List[dict], fieldnames: List[str]):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def make_default_output_dir(trace: str, fwd_pass: int, base: str = "outputs") -> str:
    stem = os.path.basename(trace)
    stem = re.sub(r"\.(json|gz|trace|pt)($|\.)", "_", stem)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._")[:48] or "trace"
    parent = os.path.join(os.getcwd(), base)
    os.makedirs(parent, exist_ok=True)
    for rank in range(10000):
        suffix = f"{stem}_fwd{fwd_pass}_r{rank:03d}"
        out = os.path.join(parent, suffix)
        if not os.path.exists(out):
            return out
    return os.path.join(parent, f"{stem}_fwd{fwd_pass}_{int(time.time())}")


def write_final_report(output_dir: str, status: str, reason: str, core_files: List[Tuple[str, str]], all_files: List[Tuple[str, str]], metadata: Dict[str, str]):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "final_report.md")
    with open(path, "w") as f:
        f.write("# FINAL_REPORT\n\n")
        f.write("## Status\n\n")
        f.write(f"- Status: **{status}**\n")
        if reason:
            f.write(f"- Reason: {reason}\n")
        f.write("\n## Task Metadata\n\n")
        for k, v in metadata.items():
            f.write(f"- {k}: `{v}`\n")
        f.write("\n## Core Important Files\n\n")
        if core_files:
            for rel, desc in core_files:
                f.write(f"- `{rel}`: {desc}\n")
        else:
            f.write("- None generated.\n")
        f.write("\n## Gate Check\n\n")
        missing = []
        for rel, _ in all_files:
            fp = os.path.join(output_dir, rel)
            if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
                missing.append(rel)
        if status == "成功" and not missing:
            f.write("- Result: PASS; all generated files exist and are non-empty.\n")
        elif missing:
            f.write("- Result: FAIL; missing or empty files:\n")
            for rel in missing:
                f.write(f"  - `{rel}`\n")
        else:
            f.write("- Result: FAIL.\n")
        f.write("\n## All Generated Files\n\n")
        if all_files:
            for rel, desc in all_files:
                f.write(f"- `{rel}`: {desc}\n")
            f.write("- `final_report.md`: final status, metadata, core files, all files, and gate result.\n")
        else:
            f.write("- `final_report.md`: failure report only.\n")
    return path


def gate_nonempty(output_dir: str, rel_files: List[str]) -> List[str]:
    bad = []
    for rel in rel_files:
        fp = os.path.join(output_dir, rel)
        if not os.path.isfile(fp) or os.path.getsize(fp) == 0:
            bad.append(rel)
    return bad



def write_call_tree_md(path: str, rows: List[dict], total_us: float):
    headers = ["Layer", "Submodule", "Python", "Py time", "ATen / Custom Op", "Op time", "CUDA API", "API time", "Kernel", "Kernel time", "Percent", "file:line"]
    with open(path, "w") as f:
        f.write("# Layer-aware Call Tree\n\n")
        f.write(f"Total kernel time: **{fmt_us(total_us)}**\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            vals = [
                r["Layer"], r["Submodule"], r["Python"], fmt_us(r["Py time"]),
                r["ATen / Custom Op"], fmt_us(r["Op time"]), r["CUDA API"], fmt_us(r["API time"]),
                r["Kernel"], fmt_us(r["Kernel time"]), pct(r["Kernel time"], total_us), r["file:line"],
            ]
            f.write("| " + " | ".join(safe_cell(v) for v in vals) + " |\n")


def gather_edges(node: Node, ids: Dict[str, str], edges: List[Tuple[str, str]], counter: List[int]):
    if node.key not in ids:
        ids[node.key] = f"N{counter[0]}"
        counter[0] += 1
    for c in node.children.values():
        if c.key not in ids:
            ids[c.key] = f"N{counter[0]}"
            counter[0] += 1
        edges.append((ids[node.key], ids[c.key]))
        gather_edges(c, ids, edges, counter)


def write_mermaid(path: str, root: Node, total_us: float, with_time: bool):
    ids: Dict[str, str] = {}
    edges: List[Tuple[str, str]] = []
    gather_edges(root, ids, edges, [0])
    key_to_node = {}

    def collect(n: Node):
        key_to_node[n.key] = n
        for c in n.children.values():
            collect(c)
    collect(root)

    with open(path, "w") as f:
        f.write("graph TD\n\n")
        for key, nid in ids.items():
            n = key_to_node[key]
            if with_time:
                label = f"{n.label}<br/>total: {fmt_us(n.inclusive_us)}<br/>self: {fmt_us(n.self_us)}<br/>{pct(n.inclusive_us, total_us)}"
                if n.file_line:
                    label += f"<br/>{n.file_line}"
            else:
                label = n.label
            f.write(f'{nid}["{mermaid_text(label)}"]\n')
        f.write("\n")
        for a, b in edges:
            f.write(f"{a} --> {b}\n")


# ── main ───────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description="Generate layer-aware call tree/graph outputs")
    ap.add_argument("--trace", required=True, help="Path to .trace.json(.gz)")
    ap.add_argument("--config", default=None, help="Path to model config.json")
    ap.add_argument("--profile", default=None, help="Model profile name; auto-inferred from config if omitted")
    ap.add_argument("--anchor-kernel", default=None, help="Override layer boundary anchor kernel substring")
    ap.add_argument("--fwd-pass", type=int, default=0, help="Forward pass index to analyze")
    ap.add_argument("--num-layers", type=int, default=None, help="Override number of layers")
    ap.add_argument("--output-dir", default=None, help="New directory for generated files; must not already exist. If omitted, create a unique ./outputs/<trace>_fwd<N>_rXXX directory.")
    ap.add_argument("--top-kernels-per-layer", type=int, default=20, help="Rows per layer in slowest_kernels_by_layer.csv")
    args = ap.parse_args()

    output_dir = args.output_dir or make_default_output_dir(args.trace, args.fwd_pass)
    metadata = {"trace": args.trace, "config": args.config or "", "profile_arg": args.profile or "auto", "fwd_pass": str(args.fwd_pass), "output_dir": output_dir}
    if os.path.exists(output_dir):
        # Per skill_rule: an explicitly existing output_dir is a task failure.
        write_final_report(
            output_dir,
            "失败",
            "output_dir already exists; refuse to mix or overwrite previous artifacts",
            [],
            [],
            metadata,
        )
        print(f"ERROR: output_dir already exists: {output_dir}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(os.path.join(output_dir, "call_tree"), exist_ok=False)
    os.makedirs(os.path.join(output_dir, "call_graph"), exist_ok=False)
    os.makedirs(os.path.join(output_dir, "mappings"), exist_ok=False)
    os.makedirs(os.path.join(output_dir, "rankings"), exist_ok=False)

    events = load_trace(args.trace)
    gpu = sorted([e for e in events if e.get("cat") == "kernel"], key=lambda e: e.get("ts", 0))
    if not gpu:
        write_final_report(output_dir, "失败", "no GPU kernel events found in trace", [], [], metadata)
        print("ERROR: no GPU kernel events found", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)
    profile = get_profile(args.profile) if args.profile else infer_profile(config)
    anchor = args.anchor_kernel or find_anchor_kernel(gpu, profile)
    anchor_indices = [i for i, e in enumerate(gpu) if anchor in e.get("name", "")]
    if len(anchor_indices) < 2:
        metadata.update({"profile": profile.name, "anchor": anchor})
        write_final_report(output_dir, "失败", f"anchor kernel '{anchor}' appears fewer than 2 times", [], [], metadata)
        print(f"ERROR: anchor kernel '{anchor}' appears fewer than 2 times", file=sys.stderr)
        sys.exit(1)

    compress_ratios = normalize_compress_ratios(config)
    num_hash_layers = config.get("num_hash_layers", 0)
    num_layers = args.num_layers or config.get("num_hidden_layers") or detect_num_layers(anchor_indices, gpu, profile.blocks_per_layer, profile.default_num_layers)
    blocks_per_pass = num_layers * profile.blocks_per_layer
    base = args.fwd_pass * blocks_per_pass
    if base + blocks_per_pass >= len(anchor_indices):
        metadata.update({"profile": profile.name, "anchor": anchor, "num_layers": str(num_layers)})
        write_final_report(output_dir, "失败", f"fwd_pass={args.fwd_pass} exceeds available anchor blocks", [], [], metadata)
        print(f"ERROR: fwd_pass={args.fwd_pass} exceeds available anchor blocks", file=sys.stderr)
        sys.exit(1)

    runtime_by_corr, cpu_by_ext, runtime_events, cpu_ops, py_events = build_correlation_maps(events)

    root = Node("model", f"Model Forward #{args.fwd_pass}", "Model")
    rows: List[dict] = []
    kernel_to_layer = []
    layer_stats = defaultdict(lambda: {"Time(ms)": 0.0, "file:line": "", "Type": ""})
    submodule_stats = defaultdict(float)
    submodule_file = {}
    kernels_by_layer = defaultdict(lambda: defaultdict(float))

    for layer_id in range(num_layers):
        layer_start_block = base + layer_id * profile.blocks_per_layer
        next_layer_start_block = base + (layer_id + 1) * profile.blocks_per_layer
        if next_layer_start_block >= len(anchor_indices):
            break
        start_i = anchor_indices[layer_start_block]
        end_i = anchor_indices[next_layer_start_block]
        layer_name = f"layer.{layer_id}"
        ltype = layer_type_label(layer_id, compress_ratios, num_layers, num_hash_layers)
        for j in range(start_i, end_i):
            k = gpu[j]
            kname = k.get("name", "")
            kdur = event_dur(k)
            rt, cpu = resolve_runtime_and_cpu(k, runtime_by_corr, cpu_by_ext, runtime_events, cpu_ops)
            py = find_enclosing(py_events, cpu.get("ts", k.get("ts", 0)) if cpu else k.get("ts", 0), event_end(cpu) if cpu else event_end(k))
            py_name = (py or cpu or {}).get("name", "<unknown python>")
            op_name = (cpu or {}).get("name", "<unknown op>")
            api_name = (rt or {}).get("name", "<unknown cuda api>")
            _, cat_key = classify_kernel(kname, profile)
            submodule = infer_submodule(kname, op_name, cat_key)
            fl = file_line_from_args(py) or file_line_from_args(cpu)

            api_dur = event_dur(rt) if rt else 0.0
            op_dur = event_dur(cpu) if cpu else 0.0
            py_dur = event_dur(py) if py else op_dur

            path = [
                (f"layer:{layer_name}", f"{layer_name} [{ltype}]", "Layer", fl),
                (f"sub:{layer_name}:{submodule}", submodule, "Layer Submodule", fl),
                (f"py:{layer_name}:{submodule}:{py_name}:{fl}", py_name, "Python Function", fl),
                (f"op:{layer_name}:{submodule}:{py_name}:{op_name}", op_name, "ATen / Custom Op", fl),
                (f"api:{layer_name}:{submodule}:{py_name}:{op_name}:{api_name}", api_name, "CUDA Runtime API", ""),
                (f"kernel:{layer_name}:{submodule}:{py_name}:{op_name}:{api_name}:{kname}", kname, "CUDA Kernel", ""),
            ]
            add_path(root, path, kdur)

            rows.append({
                "Layer": layer_name, "Submodule": submodule, "Python": py_name, "Py time": py_dur,
                "ATen / Custom Op": op_name, "Op time": op_dur, "CUDA API": api_name, "API time": api_dur,
                "Kernel": kname, "Kernel time": kdur, "file:line": fl,
            })
            kernel_to_layer.append({"Kernel": kname, "Layer": layer_name, "Submodule": submodule})
            layer_stats[layer_name]["Time(ms)"] += kdur / 1000.0
            layer_stats[layer_name]["Type"] = ltype
            if fl and not layer_stats[layer_name]["file:line"]:
                layer_stats[layer_name]["file:line"] = fl
            sub_key = (submodule, layer_name)
            submodule_stats[sub_key] += kdur / 1000.0
            if fl and sub_key not in submodule_file:
                submodule_file[sub_key] = fl
            kernels_by_layer[layer_name][kname] += kdur / 1000.0

    total_us = sum(r["Kernel time"] for r in rows)
    finalize_self(root)

    write_call_tree_md(os.path.join(output_dir, "call_tree", "layer_call_tree.md"), rows, total_us)
    write_mermaid(os.path.join(output_dir, "call_graph", "layer_call_graph.mmd"), root, total_us, with_time=False)
    write_mermaid(os.path.join(output_dir, "call_graph", "layer_call_graph_with_time.mmd"), root, total_us, with_time=True)
    write_csv(os.path.join(output_dir, "mappings", "kernel_to_layer.csv"), kernel_to_layer, ["Kernel", "Layer", "Submodule"])

    slow_layers = []
    for layer, info in sorted(layer_stats.items(), key=lambda kv: kv[1]["Time(ms)"], reverse=True):
        slow_layers.append({"Layer": layer, "Type": info["Type"], "Time(ms)": f'{info["Time(ms)"]:.6f}', "Percent": pct(info["Time(ms)"] * 1000, total_us), "file:line": info["file:line"]})
    write_csv(os.path.join(output_dir, "rankings", "slowest_layers.csv"), slow_layers, ["Layer", "Type", "Time(ms)", "Percent", "file:line"])

    slow_subs = []
    for (sub, layer), ms in sorted(submodule_stats.items(), key=lambda kv: kv[1], reverse=True):
        slow_subs.append({"Submodule": sub, "Layer": layer, "Time(ms)": f"{ms:.6f}", "Percent": pct(ms * 1000, total_us), "file:line": submodule_file.get((sub, layer), "")})
    write_csv(os.path.join(output_dir, "rankings", "slowest_submodules.csv"), slow_subs, ["Submodule", "Layer", "Time(ms)", "Percent", "file:line"])

    slow_kernels = []
    for layer in sorted(kernels_by_layer, key=lambda x: int(x.split(".")[-1]) if x.split(".")[-1].isdigit() else x):
        for kname, ms in sorted(kernels_by_layer[layer].items(), key=lambda kv: kv[1], reverse=True)[: args.top_kernels_per_layer]:
            slow_kernels.append({"Layer": layer, "Kernel": kname, "Time(ms)": f"{ms:.6f}", "Percent": pct(ms * 1000, total_us)})
    write_csv(os.path.join(output_dir, "rankings", "slowest_kernels_by_layer.csv"), slow_kernels, ["Layer", "Kernel", "Time(ms)", "Percent"])

    core_files = [
        ("call_tree/layer_call_tree.md", "Layer → Submodule → Python → ATen/Custom Op → CUDA API → Kernel markdown attribution table."),
        ("call_graph/layer_call_graph_with_time.mmd", "Mermaid execution DAG with total time, self time, percentage, and source location when available."),
        ("rankings/slowest_layers.csv", "Layers ranked by total CUDA kernel time."),
        ("rankings/slowest_submodules.csv", "Layer submodules ranked by total CUDA kernel time."),
    ]
    all_files = core_files + [
        ("call_graph/layer_call_graph.mmd", "Mermaid execution DAG without expanded timing labels."),
        ("mappings/kernel_to_layer.csv", "CUDA kernel to owning layer/submodule mapping."),
        ("rankings/slowest_kernels_by_layer.csv", "Top CUDA kernels grouped by layer."),
    ]
    bad = gate_nonempty(output_dir, [rel for rel, _ in all_files])
    status = "成功" if not bad else "失败"
    reason = "" if not bad else "Gate failed: missing or empty files: " + ", ".join(bad)
    metadata.update({"profile": profile.name, "anchor": anchor, "num_layers": str(num_layers), "total_kernel_time": fmt_us(total_us)})
    write_final_report(output_dir, status, reason, core_files, all_files, metadata)
    if bad:
        print(reason, file=sys.stderr)
        sys.exit(3)

    print(f"Wrote layer-aware call tree/graph outputs to: {output_dir}")
    print(f"Trace={args.trace}")
    print(f"Profile={profile.name}, anchor={anchor}, fwd_pass={args.fwd_pass}, layers={num_layers}, total_kernel_time={fmt_us(total_us)}")
    print(f"Final report: {os.path.join(output_dir, 'final_report.md')}")


if __name__ == "__main__":
    main()
