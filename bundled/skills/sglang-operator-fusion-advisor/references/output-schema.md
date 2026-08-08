# optimization.json output schema

## Contents

1. Top-level shape
2. `activePatterns[]` fields
3. `suggestions[]` fields
4. `options[]` fields
5. Verdict vocabulary
6. Validation invariants

## Top-level shape

```json
{
  "schemaVersion": "1.1",
  "generatedAt": "2026-08-08T10:00:00Z",
  "scope": {
    "unitPosition": 1,
    "unitId": "layer.5",
    "unitVariant": "KDA-MoE",
    "unitTotalDurationUs": 31381.326
  },
  "prescan": {
    "candidatesFile": "candidates.json",
    "registryProvenance": { "derivedFrom": "...", "importedAt": "2026-08-08", "sglangHead": "..." },
    "registryActionable": 1,
    "registryActive": 0,
    "unexplainedClusters": 5
  },
  "activePatterns": [],
  "suggestions": [],
  "limits": []
}
```

- `schemaVersion` is the literal string `"1.1"`.
- `scope` states exactly which repeating unit was analyzed, copied from
  `candidates.json`'s `scope`. Never merge two genuinely different variants.
- `prescan` carries provenance from the deterministic prescan so a reader can
  tell which registry snapshot produced the verdicts. `registryProvenance` is
  copied verbatim from `candidates.json` → `inputs.registryProvenance`.
- `activePatterns` records registry patterns that are **already in effect** in
  this unit (`* direct` status). These are not optimization targets; they exist
  so a reader can see what is already covered and does not need re-proposing.
- `suggestions` is `[]` when nothing cleared the bar — a valid, expected result.
- `limits` is copied from `candidates.json` → `limits`, plus any additional
  caveat specific to this run (for example: no web search was available).

## `activePatterns[]` fields

```json
{
  "registryPattern": "Fused MoE grouped-topk / gate kernels",
  "registryPatternZh": "MoE grouped-topk / gate 融合",
  "verdict": "已有实现已生效",
  "matchedOperators": [34, 35],
  "familySharePctOfUnit": 5.46
}
```

## `suggestions[]` fields

```json
{
  "id": "s1",
  "source": "registry",
  "verdict": "已有实现未启用",
  "registryPattern": "Fused MoE grouped-topk / gate kernels",
  "registryPatternZh": "MoE grouped-topk / gate 融合",
  "candidatePaths": ["python/sglang/srt/layers/moe/topk.py"],
  "targetOperators": [32, 34],
  "targetOperatorNames": ["route_radix_kernel<...>", "_pack_topk_ids_triton_kernel"],
  "unitPosition": 1,
  "unitId": "layer.5",
  "unitVariant": "KDA-MoE",
  "groupDurationUs": 8.4,
  "groupDurationPctOfUnit": 0.23,
  "familySharePctOfUnit": 5.46,
  "witnessSpreadOps": 2,
  "options": []
}
```

- `id`: short stable string unique within this file (`s1`, `s2`, ...).
- `source`: `"registry"` when a registry pattern explained the group,
  `"cluster"` when it came from an unexplained adjacency cluster.
- `verdict`: one value from the closed vocabulary below. This is the single most
  important field for a reader: it separates "upstream already did this, you are
  not using it" from "this is a genuinely new idea".
- `registryPattern` / `registryPatternZh`: the matched registry family, or
  `null` for `source: "cluster"`. Never name a pattern the prescan did not report.
- `candidatePaths`: where the fused implementation lives, taken from the
  prescan's resolved paths. This is what makes a suggestion actionable — a
  reader should be able to open the file. Empty list is allowed only for
  `source: "cluster"`.
- `targetOperators`: exact `index` values from `analysis.json`, timeline order.
- `targetOperatorNames`: matching `kernelName` values, same order.
- `groupDurationUs` / `groupDurationPctOfUnit`: the concrete group's own cost.
- `familySharePctOfUnit`: for registry matches, the share of every operator the
  pattern's keyword groups matched in this unit. This is the number the
  registry's `minSharePct` / `likelySharePct` thresholds are evaluated against,
  and it is usually larger than `groupDurationPctOfUnit`.
- `witnessSpreadOps`: operator-position distance across the matched witnesses,
  from the prescan. A small number means the split kernels really do run
  back-to-back.
- `options`: 1 to 3 entries, sorted by `estimatedGainPct` descending.

## `options[]` fields

```json
{
  "approach": "将 route_radix 与 _pack_topk_ids 合并进 grouped-topk 融合实现",
  "rationale": "同代码库 python/sglang/srt/layers/moe/topk.py 已提供 grouped-topk 融合入口，当前 trace 走的是拆分路径",
  "estimatedGainPct": 12.5,
  "estimatedGainBasis": "访存字节数估算：两个 kernel 共读写约 X MB，融合后省去一次中间写回和读取，按显存带宽估算减少约 12.5% 组内耗时",
  "referenceLinks": ["python/sglang/srt/layers/moe/topk.py:820"],
  "confidence": "high"
}
```

- `approach`: one sentence, concrete, naming the actual fusion. Write in Chinese.
- `rationale`: the mechanism, per research-and-estimation.md — a source
  citation, a web citation, or an explicit first-principles argument.
- `estimatedGainPct`: number in 0–100, a reduction of the **group's own**
  duration. Must not exceed what `groupDurationUs` allows.
- `estimatedGainBasis`: the arithmetic or citation behind the number.
- `referenceLinks`: file:line citations from the supplied source tree and/or
  URLs actually retrieved this run. Empty array for pure first-principles.
- `confidence`: `high` (implementation found in the analysed tree or verified
  upstream), `medium` (comparable context confirmed), `low` (first-principles
  only). Note this describes **evidence strength for the mechanism**, not the
  size of the win — a high-confidence suggestion can still be a small win.

## Verdict vocabulary

Actionable (may appear in `suggestions`):

- `已有实现未启用` — registry `mainline split`: SGLang mainline has this fusion
  and the path resolved in the analysed tree, but this trace ran the split form.
- `上游已实现待迁移` — registry `upstream split`: another framework
  (vLLM / TensorRT-LLM / TokenSpeed) ships it; migrating is the work.
- `在飞PR待跟进` — registry `inflight split`: an open upstream PR covers it.
- `疑似新机会` — an adjacency cluster no registry row explains.
- `需人工确认` — matched, but the registry's implementation path could not be
  resolved in the analysed source tree, so "just enable it" cannot be asserted.

Informational (may appear only in `activePatterns`):

- `已有实现已生效` / `上游实现已生效` / `在飞PR已生效`

## Validation invariants

Checked by `scripts/validate_optimization_package.py`:

- `schemaVersion` is `"1.1"`; `limits` is non-empty; `prescan.registryProvenance`
  is present.
- `suggestions[].verdict` is in the actionable set; informational verdicts in
  `suggestions` are an error.
- `suggestions[].source` is `registry` or `cluster`.
- `source: "registry"` requires a non-empty `registryPattern`, and that pattern
  must appear in the prescan's `registryMatches` when `--candidates-json` is
  supplied. Inventing a registry match is an error.
- `source: "cluster"` requires `registryPattern` to be null.
- `suggestions[].options` has between 1 and 3 entries, sorted by
  `estimatedGainPct` descending.
- Every `targetOperators` index exists in the source `analysis.json`.
- Every option has a non-empty `estimatedGainBasis`.
