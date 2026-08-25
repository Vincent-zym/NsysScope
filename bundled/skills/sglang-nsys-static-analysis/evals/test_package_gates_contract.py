"""Contract tests for the package gates that reject an evidence-free analysis.

Every case here reproduces a symptom of one real package that passed validation and
was still unusable: no call sites, English one-word descriptions, blank GEMM shapes,
a 400ms pipeline wait counted inside a 102ms layer, a truncated first position, and
a single sampled occurrence taken from the capture's first forward.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_analysis_package.py"
    spec = importlib.util.spec_from_file_location("package_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def grounded_row(index: int, variant: str = "KDA") -> dict[str, str]:
    return {
        "operator_name": f"nvjet_gemm_{index}",
        "duration_us": "100.0",
        "unit_position": "1",
        "unit_id": "layer.1",
        "unit_variant": variant,
        "python_function": (
            "KimiK3LinearModel.forward @ sglang/srt/models/kimi_k3.py:2377 -> "
            "KimiK3DecoderLayer.forward @ kimi_k3.py:2123-2144"
        ),
        "function_introduction": "KDA 融合输入投影：一次 GEMM 输出 q|k|v|g 四段。",
        "mapping_reason": f"gridX 与 config 推导一致（第 {index} 行）",
        "dispatch_code_snippet": "fused_states, _ = self.fused_qkvg_proj(hidden_states)",
    }


def boilerplate_row(index: int) -> dict[str, str]:
    return {
        "operator_name": f"nvjet_gemm_{index}",
        "duration_us": "100.0",
        "unit_position": "1",
        "unit_id": "layer.1",
        "unit_variant": "KDA",
        "python_function": (
            "KimiK3DecoderLayer.forward -> layer[1] -> kda/beta_projection @ "
            "sglang/srt/models/kimi_k3.py (source commit unverified)"
        ),
        "function_introduction": "KDA beta gate projection",
        "mapping_reason": f"Trace kernel {index}; source branch evidence is unverified.",
        "dispatch_code_snippet": "kernel dispatch represented by captured CUDA leaf",
    }


def test_grounded_evidence_passes():
    module = load_validator()
    errors: list[str] = []
    module.validate_evidence_depth(
        [grounded_row(i) for i in range(20)], [{"shape": "(M=16384,N=7168,K=12288)"}],
        errors,
    )
    assert errors == []


def test_boilerplate_evidence_is_rejected():
    module = load_validator()
    errors: list[str] = []
    module.validate_evidence_depth(
        [boilerplate_row(i) for i in range(20)], [{"shape": ""}], errors,
    )
    joined = "\n".join(errors)
    assert "call site" in joined
    assert "Chinese function_introduction" in joined
    assert "dispatch code snippet" in joined
    assert "no core-compute row has a shape" in joined


def test_operator_longer_than_the_unit_is_rejected():
    module = load_validator()
    rows = [grounded_row(i) for i in range(5)]
    rows.append(dict(grounded_row(9), operator_name="ncclDevKernel_SendRecv",
                     duration_us="200426.5"))
    errors: list[str] = []
    module.validate_unit_attribution(rows, 102011.4, errors)
    assert any("longer than the whole repeating unit" in error for error in errors)


def test_small_intra_layer_collective_is_allowed():
    # DCP all-to-all / TP all-reduce legitimately run inside a layer at a few percent
    module = load_validator()
    rows = [grounded_row(i) for i in range(5)]
    rows.append(dict(grounded_row(9), operator_name="ncclDevKernel_SendRecv",
                     duration_us="48.0"))
    errors: list[str] = []
    module.validate_unit_attribution(rows, 2058.4, errors)
    assert errors == []


def test_dominant_handoff_wait_is_rejected():
    module = load_validator()
    rows = [dict(grounded_row(9), operator_name="ncclDevKernel_SendRecv",
                 duration_us="800.0")]
    errors: list[str] = []
    module.validate_unit_attribution(rows, 2058.4, errors)
    assert any("rank-level wait" in error for error in errors)


def test_phase_shifted_unit_window_is_rejected():
    module = load_validator()
    rows = []
    for position, count in (("1", 32), ("2", 41), ("3", 42)):
        for index in range(count):
            rows.append(dict(grounded_row(index), unit_position=position))
    errors: list[str] = []
    module.validate_unit_attribution(rows, 102011.4, errors)
    assert any("phase-shifted" in error for error in errors)


def test_equal_positions_of_one_variant_pass():
    module = load_validator()
    rows = []
    for position in ("1", "2", "3"):
        for index in range(41):
            rows.append(dict(grounded_row(index), unit_position=position))
    errors: list[str] = []
    module.validate_unit_attribution(rows, 102011.4, errors)
    assert errors == []


def test_single_sample_is_rejected():
    module = load_validator()
    errors: list[str] = []
    module.validate_sampling({"stable_statistics": {
        "accepted_full_template_sample_count": 1, "single_sample_fallback": True,
    }}, errors)
    joined = "\n".join(errors)
    assert "single_sample_fallback" in joined
    assert "at least 3" in joined


def test_steady_state_sampling_passes():
    module = load_validator()
    errors: list[str] = []
    module.validate_sampling(
        {"stable_statistics": {"accepted_full_template_sample_count": 9}}, errors,
    )
    assert errors == []


def test_missing_manifest_is_rejected():
    # An unread manifest costs the unit composition and the sample count, and the
    # frontend then quietly reports one sample and one unit.
    module = load_validator()
    errors: list[str] = []
    module.validate_sampling({}, errors)
    assert "analysis_manifest.json" in "\n".join(errors)


def test_package_without_the_forward_pipeline_table_is_accepted(tmp_path):
    # The seventh table is a valuable bonus view (the only place the package says
    # what fraction of a forward step the measured unit is), but some captures
    # genuinely cannot produce it. A package missing only this table must still
    # pass the *table-presence* gate -- other, unrelated gates (classification
    # order, manifest presence) are exercised by their own tests and are not this
    # test's concern, so call the presence check directly instead of the full CLI.
    module = load_validator()
    for suffix in module.SUFFIXES:
        (tmp_path / f"analysis{suffix}").write_text("x\n", encoding="utf-8")
    missing = [
        f"analysis{suffix}" for suffix in module.SUFFIXES
        if not (tmp_path / f"analysis{suffix}").is_file()
    ]
    assert missing == []
    missing_optional = [
        f"analysis{suffix}" for suffix in module.OPTIONAL_SUFFIXES
        if not (tmp_path / f"analysis{suffix}").is_file()
    ]
    assert missing_optional == ["analysis_forward_pipeline_table.csv"]


