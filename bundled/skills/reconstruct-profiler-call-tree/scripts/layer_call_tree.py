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


# ── python frame parsing ──────────────────────────────────────────────────

# torch profiler with_stack=True encodes python frames as "path/to/file.py(123): func".
PY_FRAME_RE = re.compile(r"^(?P<file>.+?)\((?P<line>\d+)\):\s*(?P<func>.+)$")


def parse_py_frame(name: str) -> Optional[Tuple[str, str, str]]:
    """Return (file, line, func) when the event name carries a source location."""
    m = PY_FRAME_RE.match(str(name).strip())
    if not m:
        return None
    return m.group("file"), m.group("line"), m.group("func")


def frame_file_line(name: str) -> str:
    parsed = parse_py_frame(name)
    return f"{parsed[0]}:{parsed[1]}" if parsed else ""


def kernel_leaf(name: str) -> str:
    """Compact leaf symbol for a demangled CUDA kernel name."""
    s = re.sub(r"^void\s+", "", str(name).strip())
    # "(anonymous namespace)::foo<...>(...)" would otherwise truncate at the first paren.
    s = re.sub(r"\(anonymous namespace\)::", "", s)
    depth = 0
    for i, ch in enumerate(s):
        if ch == "<":
            depth += 1
        elif ch == ">":
            depth = max(0, depth - 1)
        elif ch == "(" and depth == 0:
            s = s[:i]
            break
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)$", s.split("<")[0].strip().rstrip(":"))
    return m.group(1) if m else s[:80]


def build_py_frame_index(py_events: List[dict]):
    """Index python_function events by thread and by 'Python id' for chain walking."""
    by_tid: Dict[object, List[dict]] = defaultdict(list)
    by_pyid: Dict[object, dict] = {}
    for e in py_events:
        if str(e.get("cat", "")) != "python_function":
            continue
        by_tid[e.get("tid")].append(e)
        pid = get_arg(e, "Python id")
        if pid is not None:
            by_pyid[pid] = e
    for lst in by_tid.values():
        lst.sort(key=lambda e: (e.get("ts", 0), -event_dur(e)))
    return by_tid, by_pyid


def deepest_frames_for_queries(frames: List[dict], queries: List[Tuple[float, int]]) -> Dict[int, dict]:
    """Sweep properly-nested frames to find the innermost frame containing each query ts.

    ``queries`` is a list of ``(ts, query_key)``; the returned dict maps query_key
    to the innermost containing frame event.
    """
    out: Dict[int, dict] = {}
    stack: List[dict] = []
    i = 0
    for ts, key in sorted(queries, key=lambda q: q[0]):
        while i < len(frames) and frames[i].get("ts", 0) <= ts:
            f = frames[i]
            while stack and event_end(stack[-1]) < f.get("ts", 0):
                stack.pop()
            stack.append(f)
            i += 1
        while stack and event_end(stack[-1]) < ts:
            stack.pop()
        if stack:
            out[key] = stack[-1]
    return out


def frame_chain(frame: Optional[dict], by_pyid: Dict[object, dict], max_depth: int = 64) -> List[dict]:
    """Return the ancestor chain from outermost to the given frame via Python parent id."""
    chain: List[dict] = []
    cur = frame
    seen = set()
    while cur is not None and len(chain) < max_depth:
        pid = get_arg(cur, "Python id")
        if pid in seen:
            break
        seen.add(pid)
        chain.append(cur)
        parent_id = get_arg(cur, "Python parent id")
        cur = by_pyid.get(parent_id) if parent_id is not None else None
    chain.reverse()
    return chain


def pick_dispatch_frame(chain: List[dict], source_filter: str) -> Optional[dict]:
    """Pick the deepest frame whose file path matches source_filter, else the deepest frame."""
    if not chain:
        return None
    if source_filter:
        for f in reversed(chain):
            parsed = parse_py_frame(f.get("name", ""))
            if parsed and source_filter in parsed[0]:
                return f
    for f in reversed(chain):
        if parse_py_frame(f.get("name", "")):
            return f
    return chain[-1]


def resolve_innermost_cpu_ops(kernels: List[dict], launch_by_corr: Dict[str, dict], cpu_ops: List[dict]) -> Dict[int, dict]:
    """Innermost enclosing cpu_op per kernel, anchored on the CPU-side launch event.

    Replaces a per-kernel linear scan over every cpu_op; both sides are properly
    nested per thread, so one sweep resolves all kernels.
    """
    by_tid: Dict[object, List[dict]] = defaultdict(list)
    for e in cpu_ops:
        by_tid[e.get("tid")].append(e)
    for lst in by_tid.values():
        lst.sort(key=lambda e: (e.get("ts", 0), -event_dur(e)))

    queries_by_tid: Dict[object, List[Tuple[float, int]]] = defaultdict(list)
    for k in kernels:
        c = corr_id(k)
        launch = launch_by_corr.get(c) if c else None
        if launch is None:
            continue
        queries_by_tid[launch.get("tid")].append((float(launch.get("ts", 0)), id(k)))

    out: Dict[int, dict] = {}
    for tid, queries in queries_by_tid.items():
        ops = by_tid.get(tid)
        if ops:
            out.update(deepest_frames_for_queries(ops, queries))
    return out


def resolve_dispatch_sites(kernels: List[dict], launch_by_corr: Dict[str, dict], py_events: List[dict], source_filter: str, chain_depth: int):
    """Map each kernel correlation id to its launching python dispatch site.

    A kernel's GPU timestamp does not overlap the CPU python frames, so the launch
    event (cudaLaunchKernel / cuLaunchKernelEx) is used as the CPU-side anchor.
    """
    by_tid, by_pyid = build_py_frame_index(py_events)
    queries_by_tid: Dict[object, List[Tuple[float, int]]] = defaultdict(list)
    launch_for_key: Dict[int, dict] = {}
    for k in kernels:
        c = corr_id(k)
        launch = launch_by_corr.get(c) if c else None
        if launch is None:
            continue
        key = id(k)
        launch_for_key[key] = launch
        queries_by_tid[launch.get("tid")].append((float(launch.get("ts", 0)), key))

    innermost: Dict[int, dict] = {}
    for tid, queries in queries_by_tid.items():
        frames = by_tid.get(tid)
        if not frames:
            continue
        innermost.update(deepest_frames_for_queries(frames, queries))

    resolved: Dict[int, dict] = {}
    for key, launch in launch_for_key.items():
        chain = frame_chain(innermost.get(key), by_pyid)
        dispatch = pick_dispatch_frame(chain, source_filter)
        parsed = parse_py_frame(dispatch.get("name", "")) if dispatch else None
        chain_names = [f.get("name", "") for f in chain if parse_py_frame(f.get("name", ""))]
        resolved[key] = {
            "py_name": dispatch.get("name", "") if dispatch else "",
            "py_dur": event_dur(dispatch) if dispatch else 0.0,
            "file_line": f"{parsed[0]}:{parsed[1]}" if parsed else "",
            "func": parsed[2] if parsed else "",
            "chain": chain_names[-chain_depth:] if chain_depth > 0 else chain_names,
            "innermost": innermost.get(key, {}).get("name", ""),
        }
    return resolved


# ── profile / layer helpers ────────────────────────────────────────────────


def config_layer_counts(config: dict) -> List[int]:
    """Plausible anchor-blocks-per-forward counts derived from a model config.

    Multimodal configs nest the language model under ``text_config``, and a model
    with next-token-prediction layers emits anchors for those too, so both the
    bare hidden-layer count and the count including nextn layers are candidates.
    """
    sections = [config]
    for key in ("text_config", "language_config", "llm_config"):
        section = config.get(key)
        if isinstance(section, dict):
            sections.append(section)
    counts: List[int] = []
    for section in sections:
        base = section.get("num_hidden_layers")
        if not isinstance(base, int) or base <= 0:
            continue
        nextn = section.get("num_nextn_predict_layers") or 0
        for candidate in (base, base + (nextn if isinstance(nextn, int) else 0)):
            if candidate > 0 and candidate not in counts:
                counts.append(candidate)
    return counts


def config_num_layers(config: dict) -> Optional[int]:
    """``num_hidden_layers`` from the config, looking inside nested sections."""
    counts = config_layer_counts(config)
    return counts[0] if counts else None


def detect_anchor_from_config(gpu_kernels: List[dict], config: dict) -> Optional[Tuple[str, int]]:
    """Pick a layer-boundary anchor whose occurrence count fits a config layer count.

    A once-per-layer kernel appears ``layers_per_forward * forward_passes`` times,
    so a candidate is accepted when its count divides evenly by a config-derived
    layer count and yields at least two complete forward passes.

    Counting is done on the compact leaf symbol, which is also what gets returned:
    counting full templated symbols but returning their shared leaf would select an
    anchor that matches far more kernels than were counted (several distinct GEMM
    instantiations collapse to one leaf) and slice layers at the wrong offsets.

    Among valid candidates the smallest count wins, since one occurrence per layer
    is the smallest count that still reaches every layer; a kernel launched twice
    per layer would otherwise halve the layer windows.
    """
    counts = config_layer_counts(config)
    if not counts:
        return None
    occurrences: Dict[str, int] = defaultdict(int)
    for event in gpu_kernels:
        leaf = kernel_leaf(str(event.get("name", "")))
        if leaf:
            occurrences[leaf] += 1
    best: Optional[Tuple[int, int, str]] = None
    for name, count in occurrences.items():
        for layers in counts:
            if count < layers * 2 or count % layers:
                continue
            # Prefer the finest layer partition, then the sparsest anchor.
            key = (-layers, count)
            if best is None or key < (-best[1], best[0]):
                best = (count, layers, name)
    if best is None:
        return None
    return best[2], best[1]


def find_anchor_kernel(gpu_kernels: List[dict], profile: ModelProfile, config: Optional[dict] = None) -> str:
    def occurrences(needle: str) -> int:
        return sum(1 for e in gpu_kernels if needle in e.get("name", ""))

    # An inferred profile can name an anchor this capture does not contain (e.g. a
    # config matches dsv3_mla but the deployment runs a sparse-attention backend).
    # Only trust the profile's anchor when it is actually present.
    if profile.anchor_kernel and occurrences(profile.anchor_kernel) >= 2:
        return profile.anchor_kernel
    if config:
        detected = detect_anchor_from_config(gpu_kernels, config)
        if detected and occurrences(detected[0]) >= 4:
            return detected[0]
    if profile.anchor_kernel:
        return profile.anchor_kernel
    for c in ["mhc_post_tilelang", "flash_fwd_mla_combine", "AllReduce"]:
        if occurrences(c) >= 4:
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
        if "cuda_runtime" in cat or "cuda_driver" in cat or name.startswith("cuda") or name.startswith("cuLaunch"):
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


def resolve_runtime_and_cpu(kernel: dict, runtime_by_corr, cpu_by_ext, runtime_events, cpu_ops, cpu_fallback: Optional[Dict[int, dict]] = None):
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
    if cpu is None and cpu_fallback is not None:
        cpu = cpu_fallback.get(id(kernel))
    elif cpu is None:
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
    ap.add_argument("--fwd-pass", type=int, default=0, help="Forward pass index to analyze; a negative value selects a steady-state pass automatically")
    ap.add_argument("--num-layers", type=int, default=None, help="Override number of layers")
    ap.add_argument("--output-dir", default=None, help="New directory for generated files; must not already exist. If omitted, create a unique ./outputs/<trace>_fwd<N>_rXXX directory.")
    ap.add_argument("--top-kernels-per-layer", type=int, default=20, help="Rows per layer in slowest_kernels_by_layer.csv")
    ap.add_argument("--source-filter", default="sglang/", help="Path substring identifying model/framework source; the deepest matching python frame is reported as the dispatch site. Empty string disables filtering.")
    ap.add_argument("--chain-depth", type=int, default=8, help="Number of innermost python frames kept in the recorded call chain")
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
    anchor = args.anchor_kernel or find_anchor_kernel(gpu, profile, config)
    anchor_indices = [i for i, e in enumerate(gpu) if anchor in e.get("name", "")]
    if len(anchor_indices) < 2:
        metadata.update({"profile": profile.name, "anchor": anchor})
        write_final_report(output_dir, "失败", f"anchor kernel '{anchor}' appears fewer than 2 times", [], [], metadata)
        print(f"ERROR: anchor kernel '{anchor}' appears fewer than 2 times", file=sys.stderr)
        sys.exit(1)

    compress_ratios = normalize_compress_ratios(config)
    num_hash_layers = config.get("num_hash_layers", 0)
    num_layers = args.num_layers or config_num_layers(config) or detect_num_layers(anchor_indices, gpu, profile.blocks_per_layer, profile.default_num_layers)
    if not args.num_layers and not args.anchor_kernel:
        # When the anchor came from config-based detection, its own layer count is
        # the one consistent with the anchor's occurrence count; a bare
        # num_hidden_layers can disagree with it (e.g. nextn layers also emit the
        # anchor) and would slice forwards at the wrong offsets.
        detected = detect_anchor_from_config(gpu, config) if config else None
        if detected and detected[0] == anchor:
            num_layers = detected[1]
    blocks_per_pass = num_layers * profile.blocks_per_layer
    fwd_pass = args.fwd_pass
    if fwd_pass < 0:
        # Automatic selection: skip the cold-start pass and the trailing one, which
        # can be truncated by when profiling stopped.
        available = len(anchor_indices) // blocks_per_pass
        fwd_pass = max(0, available - 2)
        metadata["fwd_pass_selection"] = f"auto (available={available})"
    metadata["fwd_pass"] = str(fwd_pass)
    base = fwd_pass * blocks_per_pass
    if base + blocks_per_pass >= len(anchor_indices):
        metadata.update({"profile": profile.name, "anchor": anchor, "num_layers": str(num_layers)})
        write_final_report(output_dir, "失败", f"fwd_pass={fwd_pass} exceeds available anchor blocks", [], [], metadata)
        print(f"ERROR: fwd_pass={fwd_pass} exceeds available anchor blocks", file=sys.stderr)
        sys.exit(1)

    runtime_by_corr, cpu_by_ext, runtime_events, cpu_ops, py_events = build_correlation_maps(events)

    # Kernels of the requested forward pass only; resolving dispatch sites for the
    # whole trace would scan far more python frames than needed.
    pass_start = anchor_indices[base]
    pass_end = anchor_indices[min(base + blocks_per_pass, len(anchor_indices) - 1)]
    dispatch_sites = resolve_dispatch_sites(
        gpu[pass_start:pass_end], runtime_by_corr, py_events, args.source_filter, args.chain_depth
    )
    cpu_fallback = resolve_innermost_cpu_ops(gpu[pass_start:pass_end], runtime_by_corr, cpu_ops)

    root = Node("model", f"Model Forward #{fwd_pass}", "Model")
    rows: List[dict] = []
    kernel_to_layer = []
    dispatch_rows: List[dict] = []
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
            rt, cpu = resolve_runtime_and_cpu(k, runtime_by_corr, cpu_by_ext, runtime_events, cpu_ops, cpu_fallback)
            site = dispatch_sites.get(id(k)) or {}
            py = None
            if not site.get("py_name"):
                # Only fall back to a linear containment scan when correlation-based
                # resolution failed; py_events can hold over a million frames.
                py = find_enclosing(py_events, cpu.get("ts", k.get("ts", 0)) if cpu else k.get("ts", 0), event_end(cpu) if cpu else event_end(k))
            py_name = site.get("py_name") or (py or cpu or {}).get("name", "<unknown python>")
            op_name = (cpu or {}).get("name", "<unknown op>")
            api_name = (rt or {}).get("name", "<unknown cuda api>")
            _, cat_key = classify_kernel(kname, profile)
            submodule = infer_submodule(kname, op_name, cat_key)
            fl = site.get("file_line") or file_line_from_args(py) or file_line_from_args(cpu) or frame_file_line(py_name)

            api_dur = event_dur(rt) if rt else 0.0
            op_dur = event_dur(cpu) if cpu else 0.0
            py_dur = site.get("py_dur") or (event_dur(py) if py else op_dur)

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
            dispatch_rows.append({
                "Layer": layer_name,
                "Submodule": submodule,
                "Kernel": kname,
                "Kernel time(us)": f"{kdur:.3f}",
                "file:line": fl,
                "Function": site.get("func", ""),
                "ATen / Custom Op": op_name,
                "CUDA API": api_name,
                "Python call chain": " -> ".join(site.get("chain", [])),
            })
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
    write_csv(
        os.path.join(output_dir, "mappings", "kernel_dispatch_sites.csv"),
        dispatch_rows,
        ["Layer", "Submodule", "Kernel", "Kernel time(us)", "file:line", "Function", "ATen / Custom Op", "CUDA API", "Python call chain"],
    )

    # Per-kernel-symbol cache so downstream analysis resolves a source location once
    # per kernel instead of once per launch.
    cache: Dict[str, dict] = {}
    for r in dispatch_rows:
        entry = cache.setdefault(r["Kernel"], {
            "kernel_leaf": kernel_leaf(r["Kernel"]),
            "file_line": r["file:line"],
            "function": r["Function"],
            "aten_op": r["ATen / Custom Op"],
            "cuda_api": r["CUDA API"],
            "python_call_chain": r["Python call chain"],
            "submodules": [],
            "layers": [],
            "launch_count": 0,
            "total_time_us": 0.0,
        })
        if not entry["file_line"] and r["file:line"]:
            entry.update({"file_line": r["file:line"], "function": r["Function"], "python_call_chain": r["Python call chain"]})
        if r["Submodule"] not in entry["submodules"]:
            entry["submodules"].append(r["Submodule"])
        if r["Layer"] not in entry["layers"]:
            entry["layers"].append(r["Layer"])
        entry["launch_count"] += 1
        entry["total_time_us"] += float(r["Kernel time(us)"])
    for entry in cache.values():
        entry["total_time_us"] = round(entry["total_time_us"], 3)
        entry["layer_count"] = len(entry["layers"])
        entry.pop("layers")
    resolved_syms = sum(1 for e in cache.values() if e["file_line"])
    with open(os.path.join(output_dir, "mappings", "dispatch_site_cache.json"), "w") as f:
        json.dump({
            "trace": args.trace,
            "fwd_pass": fwd_pass,
            "source_filter": args.source_filter,
            "kernel_symbols": len(cache),
            "kernel_symbols_with_source": resolved_syms,
            "launches": len(dispatch_rows),
            "kernels": cache,
        }, f, indent=2, ensure_ascii=False)

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
        ("mappings/kernel_dispatch_sites.csv", "Per-launch kernel → dispatch `file:line`, function, and python call chain."),
        ("mappings/dispatch_site_cache.json", "Per-kernel-symbol dispatch-site cache for reuse by downstream source mapping."),
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
    metadata.update({"profile": profile.name, "anchor": anchor, "num_layers": str(num_layers), "total_kernel_time": fmt_us(total_us), "source_filter": args.source_filter, "kernel_symbols_with_source": f"{resolved_syms}/{len(cache)}"})
    write_final_report(output_dir, status, reason, core_files, all_files, metadata)
    if bad:
        print(reason, file=sys.stderr)
        sys.exit(3)

    print(f"Wrote layer-aware call tree/graph outputs to: {output_dir}")
    print(f"Trace={args.trace}")
    print(f"Profile={profile.name}, anchor={anchor}, fwd_pass={fwd_pass}, layers={num_layers}, total_kernel_time={fmt_us(total_us)}")
    print(f"Final report: {os.path.join(output_dir, 'final_report.md')}")


if __name__ == "__main__":
    main()
