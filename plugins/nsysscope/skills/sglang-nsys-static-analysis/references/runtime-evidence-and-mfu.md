# Runtime evidence, repeating-unit timing, and MFU

## Contents

1. Runtime-material precedence
2. Timing scope
3. MFU evidence
4. CPU attribution

## Runtime-material precedence

Run `scripts/audit_runtime_evidence.py` before mapping. Treat captured
`server_args`, environment, CUDA graph structure and kernel sequence as runtime
truth. A deployment script records requested flags; the server may rewrite or
disable them. Source-derived branch claims require a commit match to captured
`SGLANG_BUILD_COMMIT`, or must be marked unverified.

Keep separate manifest objects for captured runtime, launch intent, supplied
source identity and conflicts. Never merge them into one apparently consistent
configuration.

## Timing scope

Declare one of:

- `fixed_layer_position`: the same network-layer/graph position across steps
- `structural_cycle`: every complete occurrence of the smallest layer-variant
  cycle
- `explicit_subtype`: one user-requested layer subtype

For a hybrid pattern, the default is the full structural cycle. Record:

- distinct layer variants
- ordered composition and network layer IDs when proven
- representative and average cycle wall span
- layer count
- normalized average-per-layer wall span
- accepted occurrence count per graph instance/device

Do not call a subtype or multi-layer cycle simply `单层耗时`.

## MFU evidence

For each eligible GEMM record:

- logical shape `M,N,K`
- representative/average duration used
- activation and weight formats
- accumulator behavior
- Tensor Core `compute_dtype`
- dense per-GPU peak and source
- formula and resulting percentage

Use `references/hardware-peaks.json` for supported hardware. Its values are
derived from NVIDIA's official HGX system specifications and converted to
per-GPU dense peaks.

For grouped MoE, distinguish logical routed rows from padded physical rows. If
the trace exposes only logical work, report logical MFU and say so. Do not
invent expert padding.

## CPU attribution

GPU idle is not CPU delay. To claim a CPU launch gap, join the next CUDA work's
`correlationId` to `CUPTI_ACTIVITY_KIND_RUNTIME` and compare the relevant
launch/API end timestamp with the GPU-ready boundary. If the launch completed
before prior GPU work ended, the later idle interval is not a CPU late-submit
gap. Label unresolved time as GPU idle/queue/dependency gap.
