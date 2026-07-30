# Nsight export and repeating-unit selection

## Contents

1. Export rules
2. SQLite inspection
3. Device and window selection
4. Repeating-unit and layer-ID evidence

## Export rules

- Accept `.sqlite` directly.
- For `.nsys-rep`, prefer the user-provided `nsys`; otherwise resolve it from
  `PATH`.
- Default the export beside the report as `<stem>.sqlite`.
- Do not overwrite an existing SQLite export unless refresh was requested.
- If export fails, report the exact command/stdout/stderr and stop.

The bundled extractor accepts either `--sqlite-input` or `--nsys-rep` plus an
optional `--sqlite` and `--nsys`.

## SQLite inspection

Use Python `sqlite3`. Require:

- `CUPTI_ACTIVITY_KIND_KERNEL`
- `StringIds`

Inspect whether these exist:

- `NVTX_EVENTS`
- `CUPTI_ACTIVITY_KIND_RUNTIME`
- `PROCESSES`

Read table schemas before querying because Nsight versions vary. Typical kernel
fields include `start`, `end`, `deviceId`, `streamId`, `shortName`,
`demangledName`, `correlationId`, `globalPid`, and launch dimensions.

Join name IDs through `StringIds`. Prefer `demangledName`, falling back to
`shortName` only when needed.

## Device and window selection

Group kernels by device/process and inspect counts and time ranges. Pick a model
rank/device, preserve its raw start-time order for template matching, and exclude
warmup unless specifically requested.

Identify candidate boundaries using, in descending confidence:

1. explicit user/NVTX layer range
2. runtime correlation and known layer loop
3. model-config-matching full signature sequence
4. steady-state repeated kernel motifs plus source order

The repeating unit is the smallest complete sequence that repeats. It may be a
single layer or a composite such as alternating CSA/HCA or
`[C128,C4,C4,C4,C4]`. Emit one window and one origin timeline for the full unit.
Keep a common module label only when its semantics are truly shared. Record
every structural position and subtype in row-level unit fields, not only in the
manifest. Give variant-specific paths distinct architecture-defined labels;
do not hide materially different cores behind generic Attention names.

Verify window completeness against the active, architecture-defined forward
path. For Transformer-like layers this may include pre-attention norm,
attention, post-attention merge, pre-FFN norm, MLP/MoE and final merge; for
state-space, hybrid, encoder-decoder or other blocks use their own declared
ordered modules. Confirm the next kernel begins the next unit.

Do not choose one convenient layer from an interleaved architecture. A
`KDA,KDA,KDA,MLA` pattern is a four-layer structural unit unless the user
explicitly requests one subtype. Record the layer count so downstream UI does
not call the full cycle or one subtype a generic single-layer duration.

## Layer-ID evidence

Use exact layer IDs only when supported by:

- NVTX/runtime metadata
- a user-provided layer hint
- a complete timeline sequence matched to config/source layer signatures

Do not infer exact network IDs solely from local period/modulo position. For a
composite unit, assign each contained layer its own ID and use a range in the
total row. Otherwise leave IDs blank and document ambiguity.

Record device/process, start/end raw indices and timestamps, candidate sequence,
expected signature source, selected offsets, boundary evidence, expected/actual
tail module, next kernel, and completeness status.
