#!/usr/bin/env python3
"""Emit a per-slot skeleton for one repeating unit, prefilled from the trace.

The mechanical half of slot-level semantic mapping is transcription: the kernel's
Python call chain, its dispatch statement, its launch geometry and the integer
template parameters that give away a GEMM's N/K all exist already -- in the nsys
SQLite and in the dispatch-site cache produced by `reconstruct-profiler-call-tree`.
Copying them by hand is the single most expensive step of an analysis and adds no
judgement.

This script does the copying. It does NOT decide anything: `module`,
`functional_module`, `category`, `introduction` and `mapping_reason` are left
empty on purpose, because those are the analysis. Treat the output as a draft to
edit, never as findings.

The cache is built from a torch profiler capture, i.e. a *different run* than the
nsys trace under analysis. A slot therefore carries `cache_match`:

  exact   the demangled kernel name matched a cache key verbatim
  leaf    only the compact leaf symbol matched; the template instantiation differs
  missing no cache entry -- resolve this slot the normal way

Verify before citing: a `leaf` match means the call chain came from a different
template instantiation of the same kernel, which is usually but not always the
same call site.
"""
from __future__ import annotations

import argparse
import json
import re
import sqlite3
import statistics
from collections import Counter
from pathlib import Path
from typing import Any


KERNEL_QUERY = """
SELECT k.start, k.end, k.streamId, k.gridX, k.gridY, k.gridZ,
       k.blockX, k.blockY, k.blockZ, sh.value, dm.value, k.correlationId
FROM CUPTI_ACTIVITY_KIND_KERNEL k
JOIN StringIds sh ON sh.id = k.shortName
LEFT JOIN StringIds dm ON dm.id = k.demangledName
WHERE k.deviceId = ?
ORDER BY k.start
"""


def load_kernels(db: Path, device: int) -> list[dict[str, Any]]:
    con = sqlite3.connect(str(db))
    try:
        rows = con.execute(KERNEL_QUERY, (device,)).fetchall()
    finally:
        con.close()
    return [
        {
            "idx": i,
            "start": r[0],
            "end": r[1],
            "duration_us": round((r[1] - r[0]) / 1000.0, 3),
            "stream": r[2],
            "grid": [r[3], r[4], r[5]],
            "block": [r[6], r[7], r[8]],
            "short": r[9],
            "demangled": r[10] or r[9],
            "correlation": r[11],
        }
        for i, r in enumerate(rows)
    ]


def kernel_leaf(name: str) -> str:
    """Compact leaf symbol of a demangled CUDA kernel name.

    Kept compatible with reconstruct-profiler-call-tree's kernel_leaf, plus one
    spelling nsys introduces: it renders an anonymous namespace as `<unnamed>::`
    where the torch profiler renders `(anonymous namespace)::`. Left in place, the
    `<` of `<unnamed>` opens a template depth that never closes, so the top-level
    `(` is never seen and the whole signature is returned as the leaf.
    """
    s = re.sub(r"^void\s+", "", str(name).strip())
    s = re.sub(r"\(anonymous namespace\)::", "", s)
    s = re.sub(r"<unnamed>::", "", s)
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


TEMPLATE_INT_RE = re.compile(r"(?<![A-Za-z0-9_])(\d{2,7})[uU]?(?![\d.])")


def template_ints(demangled: str) -> list[int]:
    """Integer template parameters, in order, as shape candidates.

    A GEMM's N and K are normally among these -- `..._impl<..., 16384u, 2048u, ...>`
    is q_b_proj's (N, K). Deliberately a candidate list, not an inference: which
    values are the shape depends on the kernel's template signature, which only
    the source can tell you.
    """
    start = demangled.find("<")
    if start < 0:
        return []
    values = [int(m.group(1)) for m in TEMPLATE_INT_RE.finditer(demangled[start:])]
    seen, out = set(), []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out[:12]


def unit_windows(
    kernels: list[dict], anchor: str, layers_per_unit: int,
    confirm: str | None = None, confirm_window: int = 3,
) -> list[tuple[int, int]]:
    """[begin, end) kernel-index ranges of every complete candidate unit.

    An anchor kernel commonly fires more than once per layer -- a fused
    RMSNorm+quant kernel serves both input_layernorm and post_attention_layernorm.
    `confirm` disambiguates: keep only the occurrences followed within
    `confirm_window` kernels by a kernel that fires exactly once per layer.
    """
    starts = [k["idx"] for k in kernels if k["short"] == anchor]
    if confirm:
        shorts = [k["short"] for k in kernels]
        starts = [
            idx for idx in starts
            if confirm in shorts[idx + 1: idx + 1 + confirm_window + 1]
        ]
    if len(starts) < layers_per_unit + 1:
        raise SystemExit(
            f"anchor {anchor!r} yields {len(starts)} layer starts; need at least "
            f"{layers_per_unit + 1} to bound one complete unit"
        )
    return [
        (starts[i], starts[i + layers_per_unit])
        for i in range(len(starts) - layers_per_unit)
    ]


def representative_window(
    kernels: list[dict], windows: list[tuple[int, int]], skip: int,
) -> tuple[int, int]:
    """The candidate whose kernel-count and wall span are both typical.

    Picking the median avoids the first (cold) and any outlier unit without
    hard-coding which forward pass is steady state. Selection is deterministic so
    two runs on the same trace get the same draft.
    """
    pool = windows[skip:] or windows
    counts = Counter(end - begin for begin, end in pool)
    modal_count = counts.most_common(1)[0][0]
    typical = [w for w in pool if w[1] - w[0] == modal_count] or pool
    spans = [
        (max(kernels[j]["end"] for j in range(b, e))
         - min(kernels[j]["start"] for j in range(b, e)), (b, e))
        for b, e in typical
    ]
    spans.sort(key=lambda item: item[0])
    median_span = statistics.median(span for span, _ in spans)
    return min(spans, key=lambda item: abs(item[0] - median_span))[1]


BOOL_CAST_RE = re.compile(r"\(bool\)\s*([01])")
INT_CAST_RE = re.compile(
    r"\((?:unsigned\s+)?(?:int|long\s+long|long|short|char)\)\s*"
)
UINT_SUFFIX_RE = re.compile(r"(\d)[uU]+(?![A-Za-z0-9_])")


def normalize_symbol(name: str) -> str:
    """Spelling-independent form of a demangled kernel symbol.

    nsys and the torch profiler demangle the same instantiation differently:
    `<unnamed>::foo<(unsigned int)16, (bool)1>` versus
    `(anonymous namespace)::foo<16u, true>`. Without normalizing, almost every
    templated kernel would degrade to a leaf-level match and lose the guarantee
    that the call chain belongs to *this* instantiation.
    """
    s = re.sub(r"^void\s+", "", str(name).strip())
    s = s.replace("(anonymous namespace)::", "").replace("<unnamed>::", "")
    s = BOOL_CAST_RE.sub(lambda m: "true" if m.group(1) == "1" else "false", s)
    s = INT_CAST_RE.sub("", s)
    s = UINT_SUFFIX_RE.sub(r"\1", s)
    return re.sub(r"\s+", "", s)


CACHE_FIELDS = (    "function", "file_line", "python_call_chain", "aten_op", "cuda_api",
    "dispatch_code_snippet", "snippet_line", "enclosing_branch",
    "dispatch_function_body", "body_line_range", "line_drift", "resolved_path",
    "resolved_def_line", "evidence", "submodules",
)


def cache_indexes(cache: dict) -> tuple[dict, dict, dict]:
    """(exact demangled, normalized symbol, leaf) lookups over the cache."""
    entries = cache.get("kernels") or {}
    by_norm: dict[str, dict] = {}
    by_leaf: dict[str, dict] = {}
    for name, entry in entries.items():
        by_norm.setdefault(normalize_symbol(name), entry)
        leaf = entry.get("kernel_leaf") or kernel_leaf(name)
        # Keep the busiest instantiation per leaf: it is the one whose call site
        # a reader is most likely to recognise.
        current = by_leaf.get(leaf)
        if current is None or (entry.get("launch_count") or 0) > (current.get("launch_count") or 0):
            by_leaf[leaf] = entry
    return entries, by_norm, by_leaf


def symbol_for(
    kernel: dict, exact: dict, by_norm: dict, by_leaf: dict, body_lines: int = 40,
) -> dict[str, Any]:
    entry = exact.get(kernel["demangled"])
    match = "exact"
    if entry is None:
        entry = by_norm.get(normalize_symbol(kernel["demangled"]))
        match = "normalized" if entry is not None else match
    if entry is None:
        # nsys' shortName is already the leaf symbol and avoids any demangling
        # spelling difference, so try it before parsing the full signature.
        entry = by_leaf.get(kernel["short"]) or by_leaf.get(kernel_leaf(kernel["demangled"]))
        match = "leaf" if entry is not None else "missing"
    symbol: dict[str, Any] = {
        "kernel_short": kernel["short"],
        "kernel_demangled": kernel["demangled"],
        "template_int_candidates": template_ints(kernel["demangled"]),
        "cache_match": match,
    }
    for field in CACHE_FIELDS:
        symbol[field] = (entry or {}).get(field, "" if field != "submodules" else [])
    # A body window is only a fallback for locating the dispatch statement. Keeping
    # it once the statement is known trades a large share of the draft's size for
    # nothing, and a draft too big to read is a draft nobody checks.
    if symbol["dispatch_code_snippet"]:
        symbol["dispatch_function_body"] = ""
    elif symbol["dispatch_function_body"] and body_lines > 0:
        lines = str(symbol["dispatch_function_body"]).splitlines()
        if len(lines) > body_lines:
            symbol["dispatch_function_body"] = "\n".join(lines[:body_lines]) + (
                f"\n# ... {len(lines) - body_lines} more lines; read the file for the rest"
            )
    return symbol


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--sqlite", required=True, type=Path, help="exported nsys SQLite")
    ap.add_argument("--cache", type=Path, help="dispatch_site_cache_with_snippets.json")
    ap.add_argument("--anchor", required=True, help="layer-start kernel shortName")
    ap.add_argument(
        "--anchor-confirm",
        help="kernel that fires exactly once per layer; keeps only the anchor "
             "occurrences followed by it (use when the anchor repeats within a layer)",
    )
    ap.add_argument("--confirm-window", type=int, default=3)
    ap.add_argument("--layers-per-unit", type=int, default=1)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--skip-units", type=int, default=1,
        help="candidate units to drop before choosing (default 1: the cold first unit)",
    )
    ap.add_argument(
        "--body-lines", type=int, default=40,
        help="cap on the fallback function-body window per slot (0 disables the cap)",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    kernels = load_kernels(args.sqlite, args.device)
    if not kernels:
        raise SystemExit(f"no kernels on device {args.device} in {args.sqlite}")
    windows = unit_windows(
        kernels, args.anchor, args.layers_per_unit,
        args.anchor_confirm, args.confirm_window,
    )
    begin, end = representative_window(kernels, windows, args.skip_units)

    cache = json.loads(args.cache.read_text()) if args.cache else {}
    exact, by_norm, by_leaf = cache_indexes(cache)

    # One entry per distinct kernel symbol, referenced by the slots. A 4-layer unit
    # here is 184 slots over 49 symbols, so inlining the evidence per slot would
    # quadruple the draft for no added information.
    symbols: dict[str, dict[str, Any]] = {}
    symbol_ids: dict[str, str] = {}
    slots = []
    for position, kernel in enumerate(kernels[begin:end]):
        key = kernel["demangled"]
        symbol_id = symbol_ids.get(key)
        if symbol_id is None:
            short = kernel["short"]
            symbol_id = short if short not in symbols else f"{short}#{len(symbols)}"
            symbols[symbol_id] = symbol_for(
                kernel, exact, by_norm, by_leaf, args.body_lines,
            )
            symbol_ids[key] = symbol_id
        slots.append({
            # --- from the trace: verify, do not retype ---
            "slot": position,
            "symbol": symbol_id,
            "duration_us": kernel["duration_us"],
            "stream": kernel["stream"],
            "grid": kernel["grid"],
            "block": kernel["block"],
            # --- yours to decide: left empty on purpose ---
            "module": "",
            "functional_module": "",
            "category": "",
            "shape": None,
            "introduction": "",
            "mapping_reason": "",
        })

    matches = Counter(symbols[slot["symbol"]]["cache_match"] for slot in slots)
    payload = {
        "schema_version": "1.0",
        "status": "draft -- module/functional_module/category/shape/introduction/"
                  "mapping_reason are unfilled by design",
        "sqlite": str(args.sqlite),
        "cache": str(args.cache) if args.cache else None,
        "device": args.device,
        "anchor": args.anchor,
        "anchor_confirm": args.anchor_confirm,
        "layers_per_unit": args.layers_per_unit,
        "selected_window": {
            "kernel_index_begin": begin,
            "kernel_index_end": end,
            "candidate_units": len(windows),
            "skipped_units": args.skip_units,
            "wall_span_us": round(
                (max(k["end"] for k in kernels[begin:end])
                 - min(k["start"] for k in kernels[begin:end])) / 1000.0, 3),
            "selection_rule": "modal kernel count, then median wall span",
        },
        "coverage": {
            "slots": len(slots),
            "symbols": len(symbols),
            "slots_cache_exact": matches.get("exact", 0),
            "slots_cache_normalized": matches.get("normalized", 0),
            "slots_cache_leaf": matches.get("leaf", 0),
            "slots_cache_missing": matches.get("missing", 0),
            "missing_symbols": sorted(
                sid for sid, sym in symbols.items() if sym["cache_match"] == "missing"
            ),
            "leaf_symbols": sorted(
                sid for sid, sym in symbols.items() if sym["cache_match"] == "leaf"
            ),
        },
        "caveats": [
            "The cache comes from a torch profiler capture, not from this nsys trace.",
            "cache_match=leaf means a different template instantiation supplied the "
            "call chain; confirm the call site before citing it.",
            "cache_match=missing symbols have no prefilled evidence at all.",
            "template_int_candidates are raw template integers, not a resolved shape.",
            "The selected window is one representative unit; per-slot duration_us is "
            "that occurrence only, not the stable-sample average.",
        ],
        "symbols": symbols,
        "slots": slots,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n")
    print(
        f"Wrote {args.output}\n"
        f"  slots: {len(slots)} over {len(symbols)} symbols "
        f"(kernel idx {begin}..{end - 1}, "
        f"wall {payload['selected_window']['wall_span_us']} us)\n"
        f"  slot cache match: exact={matches.get('exact', 0)} "
        f"normalized={matches.get('normalized', 0)} leaf={matches.get('leaf', 0)} "
        f"missing={matches.get('missing', 0)}"
    )


if __name__ == "__main__":
    main()



