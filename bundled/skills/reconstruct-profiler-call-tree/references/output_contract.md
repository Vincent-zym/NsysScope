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
│   └── kernel_to_layer.csv
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
| `rankings/slowest_layers.csv` | Layers ranked by total CUDA kernel time |
| `rankings/slowest_submodules.csv` | Submodules ranked by total CUDA kernel time |
| `rankings/slowest_kernels_by_layer.csv` | Top CUDA kernels grouped by layer |

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
