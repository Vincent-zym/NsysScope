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

The scope of that downgrade is narrow. An unverified or mismatched commit means
source **defaults and branch assertions** stop counting as runtime truth; it does
not make the source tree unusable as call-site evidence. Keep citing `file:line`,
keep quoting the dispatching statement, and keep deriving GEMM M/N/K from config
plus captured token counts, marking source-dependent claims unverified. Replacing
those columns with `source commit unverified` prose is a validation failure, not a
conservative choice: the package then cannot answer the question it exists for.
`scripts/validate_analysis_package.py` fails a package whose origin rows mostly
lack a `file:line` call site, a real dispatch snippet or a Chinese functional
description, and one whose core-compute table has no shape at all.

## Sampling the repeating unit

Accept at least three complete occurrences from the steady state. A single
occurrence is rejected: on a serving capture the same layer varies by up to 2x
between steps (MLA layer 30ms vs 57ms in one prefill trace), and the capture's
first forward is the least representative of all. `single_sample_fallback` is a
diagnostic flag, never an acceptable final state.

Pick the rank whose layer mix reproduces the declared unit. Under pipeline
parallelism each rank owns a different slice, and the first rank additionally
spends most of its wall time in `SendRecv` waits for its neighbours, so a layer
window measured there is dominated by a bubble rather than by layer work.

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

### Mixed-input GEMMs

`dense_tflops_per_gpu` is keyed by one dtype, so the peak is resolved
automatically only when a GEMM's tensor inputs share a dtype. When they do not --
fp8 activations against mxfp4 weights, for instance -- set `mfu_peak_tflops`
explicitly on the rule and cite `mixed_input_rules` in
`references/hardware-peaks.json`.

The rule there is that on Blackwell the tcgen05 `mxf8f6f4` path runs fp8/fp6/fp4
operands at a single rate, so a mixed-input GEMM runs at the rate of its *widest*
operand, not at the narrow operand's headline rate. SGLang's MegaMoE grouped GEMM
on B200 is the worked example: fp8 x mxfp4 peaks at 4500 TFLOPS/GPU (fp8), not
9000 (fp4). Do not let the narrow operand set the peak just because it appears in
the kernel name.

Two consequences to apply without rediscovering them:

- A fused kernel whose narrow operand is sub-byte and which folds several GEMMs
  into one launch has no expressible `(M,N,K)` byte count. Declare
  `storage_dtypes.b` as an explicit unmodeled marker such as
  `"mxfp4-subbyte-unmodeled"` and leave `mbu` empty. Publishing MFU with an
  empty MBU is correct here; publishing both contradicts the semantic map.
- Any field you add to a semantic rule must be threaded through your own emit
  pipeline. `mfu_peak_tflops` and `peak_note` are the ones that get silently
  dropped: the rule keeps the value, the CSV loses it, and MFU comes out empty
  along with it. After adding a field, diff one emitted row against its rule
  before running the validators.

## MBU evidence (approximate)

For each eligible GEMM record, alongside MFU, also estimate accessed bytes and
report `mbu` as a peak-relative percentage, in the same form as MFU:

```text
accessed_bytes = M*K*bytes_a + K*N*bytes_b + M*N*bytes_c
achieved_gb_per_s = accessed_bytes / duration_seconds / 1e9
mbu = achieved_gb_per_s / hbm_bandwidth_gb_per_s * 100
```

Each operand gets its own width, because a GEMM's three matrices routinely
differ: a bf16 activation times an fp32 factor matrix into an fp32 output is
normal, and the A term dominates whenever K is large. Declare them per rule:

```json
"storage_dtypes": {"a": "bf16", "b": "fp32", "c": "fp32"}
```

These are *storage* widths — how wide the operand is in HBM — not the compute
format. `tf32` is a tensor-core compute format with no memory representation of
its own: a `sm100_tf32_*` kernel commonly reads bf16 and upconverts in
registers, so costing its operands at 4 bytes doubles the byte count. When the
only dtype evidence is a compute format, leave `mbu` empty and record the reason
rather than assuming a width. `hbm_bandwidth_gb_per_s` is the matched profile's
per-GPU HBM peak in `references/hardware-peaks.json`; if the matched profile has
no bandwidth peak, leave `mbu` empty rather than emitting raw bytes/second.

Counting every element once assumes perfect reuse, which makes this a *lower*
bound on real HBM traffic: tiling can only re-read an operand, and cache reuse
only pushes real traffic down toward the bound. So an MBU above 100% is
arithmetically impossible and is never explained by a cache-friendly shape — it
means an input is wrong, nearly always an operand width taken from the compute
dtype. Treat it exactly like an MFU above 100%: blank the cell, record the
rejected value and the shape in the evidence sidecar, and fix the dtype
declaration. Within 0-100% the estimate is still coarse — it ignores cache
reuse, tiling and intermediate quantization/dequantization traffic — so read it
as evidence of whether a kernel is bandwidth-bound, not as an exact utilization.

## CPU attribution

GPU idle is not CPU delay. To claim a CPU launch gap, join the next CUDA work's
`correlationId` to `CUPTI_ACTIVITY_KIND_RUNTIME` and compare the relevant
launch/API end timestamp with the GPU-ready boundary. If the launch completed
before prior GPU work ended, the later idle interval is not a CPU late-submit
gap. Label unresolved time as GPU idle/queue/dependency gap.
