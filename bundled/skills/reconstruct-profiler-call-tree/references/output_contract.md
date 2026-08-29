# Output Contract Reference

Use this reference for artifact layouts, file meanings, and gate expectations. The main skill workflow should only point here.

## Layer call tree outputs

```text
output_dir/
├── final_report.md
├── call_tree/
│   └── layer_call_tree.md
├── call_graph/
│   ├── layer_call_graph.mmd
│   └── layer_call_graph_with_time.mmd
├── mappings/
│   ├── kernel_to_layer.csv
│   ├── kernel_dispatch_sites.csv
│   └── dispatch_site_cache.json
└── rankings/
    ├── slowest_layers.csv
    ├── slowest_submodules.csv
    └── slowest_kernels_by_layer.csv
```

| Relative path | Purpose |
|---|---|
| `final_report.md` | Status (`成功`/`失败`), metadata, core files, all files, and gate result |
| `call_tree/layer_call_tree.md` | Layer → Submodule → Python → ATen/Custom Op → CUDA API → Kernel markdown attribution table |
| `call_graph/layer_call_graph.mmd` | Mermaid execution DAG with compact labels |
| `call_graph/layer_call_graph_with_time.mmd` | Mermaid execution DAG with total time, self time, percent, and source location when available |
| `mappings/kernel_to_layer.csv` | CUDA kernel to owning layer/submodule mapping |
| `mappings/kernel_dispatch_sites.csv` | Per-launch kernel → dispatch `file:line`, function, and full python call chain |
| `mappings/dispatch_site_cache.json` | Per-kernel-symbol dispatch-site cache; resolve a source location once per kernel symbol instead of once per launch |
| `rankings/slowest_layers.csv` | Layers ranked by total CUDA kernel time |
| `rankings/slowest_submodules.csv` | Submodules ranked by total CUDA kernel time |
| `rankings/slowest_kernels_by_layer.csv` | Top CUDA kernels grouped by layer |

## Dispatch site resolution

`layer_call_tree.py` anchors each kernel to its launch event via `correlation`, then
finds the innermost `python_function` frame containing that launch timestamp and walks
`Python parent id` upward for the full chain. The kernel's own GPU timestamp must not be
used as the anchor: it does not overlap the CPU-side python frames.

`--source-filter` (default `sglang/`) selects the deepest frame inside the model/framework
source as the reported dispatch site, skipping third-party and interpreter frames.

The line number torch profiler stores in a frame name is the function's `def` line, not
the launching statement. To obtain the statement, run
`scripts/resolve_dispatch_snippets.py` with one or more `--source-root` paths. It locates
each function **by name** (treating the recorded line as a hint), then finds the statement
that calls the next frame in the chain. Its output adds:

| Field | Meaning |
|---|---|
| `resolved_path` / `resolved_def_line` | Where the function actually lives in the supplied source tree |
| `line_drift` | `true` when the local checkout's `def` line differs from the traced build |
| `dispatch_code_snippet` / `snippet_line` | The matched launching statement |
| `enclosing_branch` | Nearest enclosing `if`/`elif` condition explaining why the statement ran |
| `dispatch_function_body` / `body_line_range` | Bounded search window used when the dispatch frame is the deepest python frame, so no callee name exists to match |
| `evidence` | Which of the above paths produced the result |

When the dispatch frame is the deepest python frame, the launching call goes straight into
native code and no callee token exists to match; the function body is reported as a
search window rather than guessing a statement.

## `layer_call_tree.md` columns

| Column | Meaning |
|---|---|
| `Layer` | Owning layer, e.g. `layer.0` |
| `Submodule` | Best-effort inferred submodule, e.g. `self_attn`, `mlp`, `norm`, `comm` |
| `Python` | Enclosing Python or profiler annotation event when available |
| `Py time` | Enclosing Python event duration when available |
| `ATen / Custom Op` | Enclosing CPU operator or custom op event when available |
| `Op time` | CPU op duration when available |
| `CUDA API` | Correlated/enclosing CUDA runtime API event when available |
| `API time` | CUDA runtime API duration when available |
| `Kernel` | CUDA kernel event name |
| `Kernel time` | CUDA kernel duration |
| `Percent` | Kernel time as percent of total selected forward-pass kernel time |
| `file:line` | Source location from stack/callsite metadata when the trace provides it |

## Gate expectations

Before reporting success, verify every expected file exists and is non-empty. If any file is missing or empty, regenerate it when possible; otherwise write `final_report.md` with status `失败` and the failing relative paths.
