# Architecture taxonomy contract

## Contents

1. Purpose
2. Required evidence
3. JSON contract
4. Functional-module granularity
5. Mapping and aggregation
6. Fusion and overlap
7. Validation failures

## Purpose

Create `<prefix>_architecture_taxonomy.json` before mapping kernels. This file is
the task-local model architecture contract. It prevents labels from a previous
model from becoming the naming authority for a new architecture.

Treat a repeating unit as an ordered sequence of structural units. A structural
unit may be a decoder layer, encoder block, state-space block, attention variant,
pipeline stage or another model-described component. Do not assume every model
has homogeneous Transformer layers.

## Required evidence

List the current task's design notes, config, active source branch and captured
runtime evidence. Use this precedence:

1. captured runtime and trace
2. model design/config
3. verified source branch
4. launch intent

Every variant needs a source anchor, ordered functional modules and
discriminators. Discriminators may be config layer lists, source branch
predicates, NVTX labels or trace kernel motifs. Kernel-name similarity alone is
not sufficient when better evidence exists.

## JSON contract

```json
{
  "schema_version": "1.0",
  "model": "ExampleModel",
  "evidence": [
    {"kind": "config", "path": "/path/config.json"},
    {"kind": "source", "path": "/path/model.py:100-240"}
  ],
  "repeating_unit": {
    "kind": "composite",
    "positions": [
      {
        "position": 1,
        "unit_id": "layer.4",
        "unit_variant": "Variant-A",
        "layer_id": 4,
        "discriminator": "config list + trace motif"
      }
    ]
  },
  "functional_module_policy": {
    "target_min": 5,
    "target_max": 8,
    "detail_column": "module"
  },
  "variants": [
    {
      "name": "Variant-A",
      "source_evidence": "ModelLayer selects VariantA when ...",
      "discriminators": ["config.variant_a_layers", "variant_a_kernel"],
      "ordered_functional_modules": [
        "Input aggregation",
        "Variant-A core",
        "Output merge"
      ],
      "distinctive_functional_modules": ["Variant-A core"]
    }
  ],
  "fusion_groups": [
    {
      "name": "Fused front",
      "functional_module": "Input aggregation",
      "logical_owners": ["router", "shared branch", "routed branch"],
      "attribution_policy": "indivisible"
    }
  ]
}
```

Positions must be contiguous from 1. Resolve each selected kernel to exactly one
position by `layer_id`, a narrow `module_regex`, or explicit semantic-map fields.
Use `unit_id` for the concrete occurrence and `unit_variant` for the
architecture-defined subtype.

## Functional-module granularity

`module` and `功能模块` serve different purposes:

- `module` is the fine-grained execution/source attribution path;
- `功能模块` is a coarse architecture stage used for comparison and rollup.

Do not promote every projection, normalization, cache update, gate, dispatch
step or epilogue into its own functional module. A variant should normally have
5–8 ordered functional modules. More than 8 is rejected unless that variant
contains a non-empty `granularity_exception` explaining the current-model
evidence and why the extra boundaries are independently actionable.

Merge adjacent fine modules into one functional module when they form one
producer-consumer phase and have no independently meaningful branch output,
architecture boundary or optimization decision. Use these defaults:

- input norm, projections, gates, RoPE and cache preparation belong to
  `Attention 输入与投影` unless the model exposes a separately actionable path;
- attention/state-space backend belongs to one variant-specific core stage, but
  only the backend's actual core execution (for example one fused attention
  kernel) should receive the core label;
- backend output reconstruction, LSE/value recovery, postprocessing and
  architecture collectives belong in an adjacent output-reconstruction or
  communication stage. Never make a broad `MLA core` bucket that absorbs these
  surrounding operators: the fine `module` column must still show them and the
  coarse core stage must remain a meaningful single computation boundary;
- output projection, tensor-parallel communication and residual merge belong to
  `Attention 输出与通信`;
- router logits, top-k, packing, quantization and dispatch belong to
  `MoE 输入与路由`;
- shared and routed expert math belong to `MoE Experts 计算` unless the trace
  proves separately scheduled branches that the requested analysis must compare;
- combine, reconstruction, normalization and expert-side collectives belong to
  `MoE 输出与通信`;
- the final residual/branch merge is `层输出合并`.

These labels are defaults, not a fixed Transformer ontology. Derive equivalent
coarse stages from the current model. Preserve fine distinctions in `module`,
operator names, mapping reasons and fusion-group metadata.

## Mapping and aggregation

Annotate every non-total origin row with:

```text
unit_position,unit_id,unit_variant
```

Annotate every human-facing operator and stage row with:

```text
单元位置,单元ID,单元类型
```

Aggregate a functional module by:

```text
(unit_position, unit_id, unit_variant, functional_module)
```

Never aggregate only by `functional_module` in a heterogeneous or composite
unit. A cycle-wide rollup is a separate view, not a replacement for
position/variant results.

For the secondary stage/core/auxiliary analysis views, also emit an explicit
pattern-level rollup when the selected repeating unit contains multiple
positions. That rollup groups identical `功能模块` names across layers and is
the default input for functional-module charts. Keep the position-aware rows
in the CSV as audit/detail rows; do not use layer-qualified stage keys in the
pattern-level view. Fine operator rows must still retain their original layer,
unit and variant fields for drill-down.

For a homogeneous repeated pattern, preserve individual positions when layer
position changes routing load, shapes, communication or branches. Add a variant
average only as a secondary summary.

## Fusion and overlap

Do not split the measured time of one fused CUDA kernel among its logical
owners. Assign it to one named fusion group, point `functional_module` at one
declared coarse stage, list all logical owners, and set
`attribution_policy: indivisible`. The fusion-group name may remain more
specific than the coarse functional module.

Report these module metrics separately:

- stable position-aware kernel-duration sum
- representative interval union
- representative first-start-to-last-end wall span

The first is GPU work, the second removes same-module overlap, and the third
includes internal gaps. None is automatically critical-path contribution.

## Validation failures

Fail the package when:

- model evidence is absent;
- a declared variant lacks source evidence, ordered modules or discriminators;
- a composite position cannot be resolved for every selected kernel;
- final tables omit a declared variant or position;
- a variant-specific required module is absent;
- a variant declares more than eight functional modules without a
  current-model `granularity_exception`;
- `功能模块` is effectively a one-to-one copy of the fine-grained `module`
  taxonomy;
- heterogeneous results are presented as generic single-layer duration;
- a fused kernel is fractionally attributed without trace-level split evidence;
- a previous model's labels are retained without current-model justification.
