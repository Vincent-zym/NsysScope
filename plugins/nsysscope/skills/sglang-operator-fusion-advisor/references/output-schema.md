# optimization.json output schema

## Contents

1. Top-level shape
2. `suggestions[]` fields
3. `options[]` fields
4. Validation invariants

## Top-level shape

```json
{
  "schemaVersion": "1.0",
  "generatedAt": "2026-08-07T10:00:00Z",
  "scope": {
    "unitPosition": 1,
    "unitId": "layer.4",
    "unitVariant": "KDA+LatentMoE"
  },
  "suggestions": []
}
```

- `schemaVersion` is always the literal string `"1.0"` for this version of
  the skill.
- `generatedAt` is an ISO-8601 UTC timestamp.
- `scope` states exactly which repeating unit was analyzed. If the
  analyzed job's repeating unit is homogeneous (single variant across all
  positions), `unitVariant` may be a single shared value; never merge two
  genuinely different variants into one scope object.
- `suggestions` is `[]` when no candidate group cleared the bar in
  candidate-selection.md — this is a valid, expected result, not an error.

## `suggestions[]` fields

Each entry:

```json
{
  "id": "s1",
  "targetOperators": [11, 12, 13],
  "targetOperatorNames": ["per_token_group_quant_flat_kernel", "..."],
  "unitPosition": 1,
  "unitId": "layer.4",
  "unitVariant": "KDA+LatentMoE",
  "groupDurationUs": 8.4,
  "groupDurationPctOfUnit": 1.6,
  "options": []
}
```

- `id`: short stable string unique within this file (`s1`, `s2`, ...).
- `targetOperators`: the exact `index` values from `analysis.json` for
  every kernel in the candidate group, in timeline order.
- `targetOperatorNames`: the corresponding `kernelName` values, for human
  readability — must stay in the same order as `targetOperators`.
- `unitPosition`/`unitId`/`unitVariant`: copied from `scope` for
  convenience when suggestions from multiple runs are later merged by
  the frontend; must match `scope` exactly in this version (single-unit
  scope only).
- `groupDurationUs`: sum of the group's `durationUs` values from
  `analysis.json`.
- `groupDurationPctOfUnit`: that sum divided by
  `summary.normalizedLayerDurationUs` (or the equivalent total available
  in the manifest), as a percentage.
- `options`: 1 to 3 entries, sorted by `estimatedGainPct` descending.

## `options[]` fields

Each entry:

```json
{
  "approach": "将量化算子融合进后续 GEMM 的 epilogue",
  "rationale": "同代码库 moe/fused_moe_kernel.py 的 use_fp8_w8a8 分支已实现相同融合，参见 sglang/kernels/moe/fused_moe_kernel.py:210-245",
  "estimatedGainPct": 12.5,
  "estimatedGainBasis": "访存字节数估算：量化算子读写共 X MB，融合后省去一次中间写回和读取，按显存带宽估算减少约 12.5% 组内耗时",
  "referenceLinks": ["sglang/kernels/moe/fused_moe_kernel.py:210-245"],
  "confidence": "high"
}
```

- `approach`: one sentence, concrete, naming the actual fusion/optimization
  (not "optimize this operator").
- `rationale`: the mechanism, per research-and-estimation.md — must name
  either a source citation, a web citation, or an explicit first-principles
  argument.
- `estimatedGainPct`: a number, 0–100. Must not exceed what the group's own
  `groupDurationUs` allows.
- `estimatedGainBasis`: the arithmetic or citation behind the number — never
  leave this generic ("会更快").
- `referenceLinks`: file:line citations from the supplied source tree,
  and/or URLs actually retrieved via web search this run. Empty array when
  the option is pure first-principles reasoning (do not invent a link to
  fill this field).
- `confidence`: one of `high` (existing implementation found, same
  codebase or verified upstream), `medium` (web-search-confirmed pattern
  in a different but comparable context), `low` (first-principles only).

## Validation invariants

- `suggestions[].options` has between 1 and 3 entries.
- `options[]` within one suggestion are sorted by `estimatedGainPct`
  descending.
- Every `targetOperators` index actually exists in the source
  `analysis.json`'s `operators[].index` for the stated
  `unitPosition`/`unitId`/`unitVariant`.
- No suggestion targets a `category: communication` operator as its sole
  member.
- Every option has a non-empty `estimatedGainBasis`.
- `referenceLinks` entries, when present, correspond to citations actually
  used in that option's `rationale`.
