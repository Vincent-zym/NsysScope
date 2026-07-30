# Kernel mapping and position-aware statistics

## Contents

1. Description-grounded module mapping
2. Python call-chain attribution
3. Dual-stream grouping
4. Mapping files
5. Stable statistics

## Description-grounded module mapping

Read model description/design notes before kernel names. Extract canonical labels,
aliases, nesting, order, variants and boundaries. Validate against source, config
and deployment wiring.

Map each selected kernel using:

1. architecture-taxonomy position and variant
2. described submodule boundary
3. active source call/branch
4. timestamp span and stream occurrence
5. adjacent kernels and signature

Annotate every row with `unit_position`, `unit_id` and `unit_variant`. Keep the
same functional-module label in two variants only when its semantics are truly
shared; the aggregation key still includes the structural position and variant.
Never use a generic parent such as `Attention` to erase distinct KDA, MLA,
state-space, recurrent, dense or sparse execution paths.

Prefer stable high-level labels such as `self_attn`, `mlp/router`,
`mlp/experts`, `attention/pre_norm`, `attention/post_merge` and
`ffn/post_merge`, but use model-specific names when documented. Put finer
dispatch details in `python_function`.

Every kernel must be covered. Never ship `unknown/<index>`, `misc` or `other`;
use the nearest defensible parent and record uncertainty.

## Python call-chain attribution

Start at the decoder/layer `forward` and recursively follow every wrapper/helper
until the statement that dispatches CUDA/C++/Triton work. Record every frame:

```text
DecoderLayer.forward -> self.self_attn(...) -> Attention.forward ->
attn_backend.forward(...) -> Backend.forward @ path/backend.py:120-135
```

The cited path/range belongs to the deepest frame. Prefer the concrete call or
selected branch, usually under 15 lines. Ranges over 25 lines require an
`uncertain_mappings` explanation. `mapping_reason` must separately explain the
module classification and call-site evidence.

## Dual-stream grouping

One module occurrence maps to exactly one CUDA stream. Use explicit
`occurrence`/`group` identities when a label repeats or work overlaps across
streams. Sort whole occurrence groups by their earliest timestamp; within each
group sort by kernel timestamp/raw index. Keep groups contiguous.

Raw start order remains authoritative for repeating-template matching; module
group order is only the readable CSV presentation order.

## Mapping files

`module_map.json` keys may be global kernel indices or offsets relative to the
selected window:

```json
{
  "0": {"module": "self_attn", "occurrence": "layer8.attn"},
  "1": {"module": "self_attn", "occurrence": "layer8.attn"}
}
```

`python_function_map.json` and `mapping_reason_map.json` accept global indices,
relative offsets or module labels, with per-row keys taking precedence. Use
`__layer_total__` for the total call site/reason. A `*` reason fallback is only
acceptable for drafts.

## Stable statistics

Treat the selected raw window as a full template. Statistical identity must
include CSV position, raw index, module and operator name; never aggregate by
operator name alone.

First declare whether statistics target a fixed network-layer position or every
occurrence of a structural layer-variant cycle. For every included rank/device:

1. sort raw kernels by start time after `stable_start_ns`
2. match the full template inside complete CUDA-graph instances; accept all
   equivalent structural-cycle positions, or only the fixed graph position when
   fixed-layer statistics were requested
3. accept only chunks whose operator-name sequence matches every position
4. project raw offsets into module-group CSV order
5. collect per-position duration and full-window wall-span

Populate representative duration plus min/max/diff/avg. Compute each operator
average percentage against the average total wall-span. The total row uses the
first-kernel start to last-kernel end, or a reliable NVTX range, never summed
kernel duration.

Write a stats sidecar containing scheme, stable start, unit length, devices,
accepted sample count, per-device counts and per-position samples/min/max/avg.
If no additional match exists, use a clearly marked single-sample fallback.

For composite units also record layer count, ordered variants, occurrences per
graph instance, cycle wall span and normalized average-per-layer wall span.
Record per-position wall spans as well. A normalized cycle average is a
supplementary metric and must not be presented as the duration of a specific
variant.
