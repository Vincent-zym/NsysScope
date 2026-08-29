#!/usr/bin/env python3
"""Resolve dispatch code snippets for a dispatch_site_cache.json against a source tree.

``layer_call_tree.py`` records each kernel's python call chain from the trace. The
line number torch profiler stores in a frame name is the function's ``def`` line,
not the statement that launches CUDA work, so this script locates each function by
name and then finds the statement inside it that calls the next frame in the chain.

Function names are treated as authoritative and reported line numbers as hints:
source may have been edited after the trace was captured, so drift is recorded per
entry instead of silently trusted.
"""

import argparse
import json
import os
import re
from typing import Dict, List, Optional, Tuple

FRAME_RE = re.compile(r"^(?P<file>.+?)\((?P<line>\d+)\):\s*(?P<func>.+)$")
DEF_RE = "def {name}("
BRANCH_KEYWORDS = ("if ", "elif ", "if(", "elif(")


def parse_frame(name: str) -> Optional[Tuple[str, int, str]]:
    m = FRAME_RE.match(str(name).strip())
    if not m:
        return None
    return m.group("file"), int(m.group("line")), m.group("func")


def resolve_path(rel: str, roots: List[str], strip_prefixes: List[str]) -> Optional[str]:
    candidates = [rel] + [rel[len(p):] for p in strip_prefixes if rel.startswith(p)]
    for root in roots:
        for cand in candidates:
            p = os.path.join(root, cand)
            if os.path.isfile(p):
                return p
    return None


_file_cache: Dict[str, List[str]] = {}


def read_lines(path: str) -> List[str]:
    if path not in _file_cache:
        with open(path, errors="replace") as f:
            _file_cache[path] = f.read().splitlines()
    return _file_cache[path]


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip())


def find_def_line(lines: List[str], func: str, hint: int) -> Optional[int]:
    """Return the 0-based index of ``def func(``, preferring the hinted line."""
    needle = DEF_RE.format(name=func)
    hint_idx = hint - 1
    if 0 <= hint_idx < len(lines) and needle in lines[hint_idx]:
        return hint_idx
    matches = [i for i, l in enumerate(lines) if needle in l.lstrip()]
    if not matches:
        return None
    # Several overloads/nested defs can share a name; pick the one closest to the hint.
    return min(matches, key=lambda i: abs(i - hint_idx))


def function_body(lines: List[str], def_idx: int) -> Tuple[int, int]:
    """Return the [start, end) line range of the function body after its def line.

    The signature can span many lines, so the body starts after the line where the
    signature's parentheses close and a trailing ``:`` appears.
    """
    base = indent_of(lines[def_idx])
    depth = 0
    start = def_idx + 1
    for j in range(def_idx, min(def_idx + 80, len(lines))):
        s = lines[j]
        depth += sum(s.count(c) for c in "([{") - sum(s.count(c) for c in ")]}")
        if depth <= 0 and s.rstrip().endswith(":"):
            start = j + 1
            break
    end = start
    for j in range(start, len(lines)):
        s = lines[j]
        if s.strip() and indent_of(s) <= base and not s.lstrip().startswith((")", "#")):
            break
        end = j + 1
    return start, end


def enclosing_branch(lines: List[str], idx: int, stop: int) -> str:
    """Nearest enclosing if/elif condition above ``idx``, when one exists."""
    target = indent_of(lines[idx])
    for j in range(idx - 1, stop - 1, -1):
        s = lines[j]
        if not s.strip():
            continue
        if indent_of(s) < target:
            stripped = s.lstrip()
            if stripped.startswith(BRANCH_KEYWORDS):
                return s.rstrip()
            target = indent_of(s)
    return ""


def statement_at(lines: List[str], idx: int, max_lines: int = 6) -> str:
    """Capture a statement, extending across an unbalanced open paren."""
    out = [lines[idx].rstrip()]
    depth = sum(lines[idx].count(c) for c in "([{") - sum(lines[idx].count(c) for c in ")]}")
    j = idx + 1
    while depth > 0 and j < len(lines) and len(out) < max_lines:
        out.append(lines[j].rstrip())
        depth += sum(lines[j].count(c) for c in "([{") - sum(lines[j].count(c) for c in ")]}")
        j += 1
    return "\n".join(out)


def skip_docstring(lines: List[str], start: int, end: int) -> int:
    """Return the first body line index after a leading docstring, if any."""
    i = start
    while i < end and not lines[i].strip():
        i += 1
    if i >= end:
        return start
    stripped = lines[i].lstrip()
    for quote in ('"""', "'''"):
        if stripped.startswith(quote):
            if stripped.count(quote) >= 2 and len(stripped) > 3:
                return i + 1
            for j in range(i + 1, end):
                if quote in lines[j]:
                    return j + 1
            return start
    return start


def find_call(lines: List[str], start: int, end: int, targets: List[str]) -> Optional[int]:
    """Find the statement that calls one of ``targets``, most specific token first."""
    for token in targets:
        if not token:
            continue
        for i in range(start, min(end, len(lines))):
            s = lines[i]
            stripped = s.lstrip()
            if stripped.startswith(("#", "import ", "from ", '"""', "'''")):
                continue
            if token not in s:
                continue
            # Require a call or attribute reference, not a bare word in a docstring
            # or type annotation.
            if re.search(re.escape(token) + r"\s*\(", s) or f".{token}" in s or f"{token}." in s:
                return i
    return None


def call_targets(chain: List[str], dispatch_idx: int, entry: dict) -> List[str]:
    """Tokens that identify the dispatch statement, most specific first."""
    targets = []
    for frame in chain[dispatch_idx + 1:]:
        parsed = parse_frame(frame)
        if parsed:
            targets.append(parsed[2])
    op = str(entry.get("aten_op") or "")
    if op and "unknown" not in op:
        targets.append(op)
        targets.append(op.split("::")[-1])
    leaf = str(entry.get("kernel_leaf") or "")
    if leaf:
        targets.append(leaf)
        base = re.sub(r"_kernel(_v\d+)?(_persist)?$", "", leaf)
        if base != leaf:
            targets.append(base)
        # Kernel symbols are often prefixed by an arch/impl tag such as
        # "sm100_fp8_fp4_mega_moe_impl"; the python-side callee usually keeps the
        # middle words, so also try progressively trimmed forms.
        parts = base.split("_")
        for cut in range(1, min(3, len(parts))):
            targets.append("_".join(parts[cut:]))
        base_noimpl = re.sub(r"_impl$", "", base)
        if base_noimpl != base:
            targets.append(base_noimpl)
    # De-duplicate while keeping specificity order.
    seen = set()
    return [t for t in targets if t and not (t in seen or seen.add(t))]


def resolve_entry(entry: dict, roots: List[str], strip_prefixes: List[str]) -> dict:
    chain = [f for f in str(entry.get("python_call_chain") or "").split(" -> ") if f]
    result = {
        "resolved_path": "",
        "reported_line": "",
        "resolved_def_line": "",
        "line_drift": None,
        "dispatch_code_snippet": "",
        "snippet_line": "",
        "enclosing_branch": "",
        "dispatch_function_body": "",
        "body_line_range": "",
        "evidence": "",
    }
    parsed = parse_frame(entry.get("py_name") or "") if entry.get("py_name") else None
    if parsed is None:
        fl = str(entry.get("file_line") or "")
        if ":" in fl:
            f, ln = fl.rsplit(":", 1)
            parsed = (f, int(ln) if ln.isdigit() else 0, str(entry.get("function") or ""))
    if parsed is None:
        result["evidence"] = "no python frame recorded for this kernel"
        return result

    rel, hint, func = parsed
    result["reported_line"] = hint
    path = resolve_path(rel, roots, strip_prefixes)
    if path is None:
        result["evidence"] = f"source file not found under provided roots: {rel}"
        return result
    result["resolved_path"] = path

    lines = read_lines(path)
    def_idx = find_def_line(lines, func, hint)
    if def_idx is None:
        result["evidence"] = f"function '{func}' not found in {rel}; source likely predates or postdates the trace"
        return result
    result["resolved_def_line"] = def_idx + 1
    result["line_drift"] = (def_idx + 1) != hint

    start, end = function_body(lines, def_idx)
    dispatch_idx = len(chain) - 1
    for i, frame in enumerate(chain):
        p = parse_frame(frame)
        if p and p[2] == func and p[0] == rel:
            dispatch_idx = i
            break
    targets = call_targets(chain, dispatch_idx, entry)
    call_idx = find_call(lines, start, end, targets)
    if call_idx is None:
        # No callee token to match against (the dispatch frame is the deepest python
        # frame, so the next callee is native code). Emit the whole function body as a
        # bounded search window instead of guessing a statement.
        body_start = skip_docstring(lines, start, end)
        body = [l.rstrip() for l in lines[body_start:min(end, body_start + 25)]]
        while body and not body[-1].strip():
            body.pop()
        result["dispatch_function_body"] = "\n".join(body)
        result["body_line_range"] = f"{body_start + 1}-{body_start + len(body)}"
        result["evidence"] = (
            "function located by name; dispatch frame is the deepest python frame so the "
            "launching statement cannot be matched by callee name — function body provided "
            "as the search window"
        )
        return result

    result["snippet_line"] = call_idx + 1
    result["dispatch_code_snippet"] = statement_at(lines, call_idx)
    result["enclosing_branch"] = enclosing_branch(lines, call_idx, start)
    result["evidence"] = "function matched by name; dispatch statement matched by callee token"
    return result


def main():
    ap = argparse.ArgumentParser(description="Attach source dispatch snippets to a dispatch_site_cache.json")
    ap.add_argument("--cache", required=True, help="mappings/dispatch_site_cache.json from layer_call_tree.py")
    ap.add_argument("--source-root", action="append", required=True, help="Source tree root; repeat for multiple roots")
    ap.add_argument("--strip-prefix", action="append", default=None, help="Path prefix to strip from trace-recorded paths (default: sglang/python/)")
    ap.add_argument("--output", default=None, help="Output JSON path (default: <cache dir>/dispatch_site_cache_with_snippets.json)")
    args = ap.parse_args()

    strip_prefixes = args.strip_prefix or ["sglang/python/"]
    roots = [os.path.abspath(r) for r in args.source_root]
    for r in roots:
        if not os.path.isdir(r):
            raise SystemExit(f"source root is not a directory: {r}")

    with open(args.cache) as f:
        cache = json.load(f)
    kernels = cache.get("kernels") or {}

    counts = {"total": 0, "file_resolved": 0, "func_resolved": 0, "snippet_resolved": 0, "body_window_only": 0, "line_drifted": 0, "unresolved": 0}
    for name, entry in kernels.items():
        counts["total"] += 1
        res = resolve_entry(entry, roots, strip_prefixes)
        entry.update(res)
        if res["resolved_path"]:
            counts["file_resolved"] += 1
        if res["resolved_def_line"]:
            counts["func_resolved"] += 1
        if res["dispatch_code_snippet"]:
            counts["snippet_resolved"] += 1
        elif res["dispatch_function_body"]:
            counts["body_window_only"] += 1
        else:
            counts["unresolved"] += 1
        if res["line_drift"]:
            counts["line_drifted"] += 1

    cache["source_resolution"] = {
        "source_roots": roots,
        "strip_prefixes": strip_prefixes,
        "counts": counts,
        "note": "Function names are authoritative; reported_line is a hint. line_drift=true means the local checkout differs from the traced build for that function.",
    }
    out = args.output or os.path.join(os.path.dirname(os.path.abspath(args.cache)), "dispatch_site_cache_with_snippets.json")
    with open(out, "w") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

    print(f"Wrote {out}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
