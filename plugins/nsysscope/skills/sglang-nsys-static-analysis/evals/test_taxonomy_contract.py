from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_validator():
    path = ROOT / "scripts" / "validate_architecture_taxonomy.py"
    spec = importlib.util.spec_from_file_location("taxonomy_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def taxonomy() -> dict:
    return {
        "schema_version": "1.0",
        "model": "SyntheticMixedModel",
        "evidence": [
            {"kind": "config", "path": "/evidence/config.json"},
            {"kind": "source", "path": "/evidence/model.py:10-80"},
        ],
        "repeating_unit": {
            "kind": "composite",
            "positions": [
                {
                    "position": 1,
                    "unit_id": "layer.4",
                    "unit_variant": "KDA",
                    "layer_id": 4,
                },
                {
                    "position": 2,
                    "unit_id": "layer.7",
                    "unit_variant": "MLA",
                    "layer_id": 7,
                },
            ],
        },
        "functional_module_policy": {
            "target_min": 5,
            "target_max": 8,
            "detail_column": "module",
        },
        "variants": [
            {
                "name": "KDA",
                "source_evidence": "layer 4 selects KDA",
                "discriminators": ["config.kda_layers", "kda_core"],
                "ordered_functional_modules": ["Input", "KDA core"],
                "distinctive_functional_modules": ["KDA core"],
            },
            {
                "name": "MLA",
                "source_evidence": "layer 7 selects MLA",
                "discriminators": ["config.mla_layers", "mla_core"],
                "ordered_functional_modules": ["Input", "MLA core"],
                "distinctive_functional_modules": ["MLA core"],
            },
        ],
        "fusion_groups": [],
    }


def test_validator_rejects_heterogeneous_variant_without_discriminator():
    value = taxonomy()
    value["variants"][1]["discriminators"] = []
    errors = load_validator().validate(value)
    assert any("needs discriminators" in error for error in errors)


def test_validator_rejects_overly_fine_functional_modules_without_exception():
    value = taxonomy()
    value["variants"][0]["ordered_functional_modules"] = [
        f"stage-{index}" for index in range(1, 10)
    ]
    errors = load_validator().validate(value)
    assert any("requires a current-model granularity_exception" in error for error in errors)

    value["variants"][0]["granularity_exception"] = (
        "Nine independently scheduled branches are required by current-model evidence."
    )
    errors = load_validator().validate(value)
    assert not any("granularity_exception" in error for error in errors)

    value["functional_module_policy"]["target_max"] = 20
    errors = load_validator().validate(value)
    assert any("target_max <= 8" in error for error in errors)


def test_builder_never_merges_same_named_stage_across_variants(tmp_path: Path):
    origin = tmp_path / "origin.csv"
    fieldnames = [
        "序号", "module", "operator_name", "duration_us", "start_ns", "end_ns",
        "device", "stream", "layer_id", "duration_avg_us", "python_function",
        "function_introduction", "mapping_reason",
    ]
    rows = [
        {
            "序号": 1, "module": "layers.4/input", "operator_name": "input_kernel",
            "duration_us": 10, "start_ns": 0, "end_ns": 10000, "device": 0,
            "stream": 1, "layer_id": 4, "duration_avg_us": 10,
        },
        {
            "序号": 2, "module": "layers.4/kda_core", "operator_name": "kda_kernel",
            "duration_us": 20, "start_ns": 10000, "end_ns": 30000, "device": 0,
            "stream": 1, "layer_id": 4, "duration_avg_us": 20,
        },
        {
            "序号": 3, "module": "layers.7/input", "operator_name": "input_kernel",
            "duration_us": 30, "start_ns": 30000, "end_ns": 60000, "device": 0,
            "stream": 1, "layer_id": 7, "duration_avg_us": 30,
        },
        {
            "序号": 4, "module": "layers.7/mla_core", "operator_name": "mla_kernel",
            "duration_us": 40, "start_ns": 60000, "end_ns": 100000, "device": 0,
            "stream": 1, "layer_id": 7, "duration_avg_us": 40,
        },
        {
            "序号": 5, "module": "__layer_total__", "operator_name": "total",
            "duration_us": 100, "start_ns": 0, "end_ns": 100000, "device": 0,
            "stream": 1, "duration_avg_us": 100,
        },
    ]
    with origin.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    semantic = {
        "rules": [
            {"module_regex": "/input$", "functional_module": "Input"},
            {"module_regex": "kda_core$", "functional_module": "KDA core"},
            {"module_regex": "mla_core$", "functional_module": "MLA core"},
        ],
    }
    taxonomy_path = tmp_path / "taxonomy.json"
    semantic_path = tmp_path / "semantic.json"
    taxonomy_path.write_text(json.dumps(taxonomy()))
    semantic_path.write_text(json.dumps(semantic))

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_static_analysis_tables.py"),
            "--origin-csv", str(origin),
            "--output-dir", str(tmp_path),
            "--prefix", "mixed",
            "--semantic-map", str(semantic_path),
            "--taxonomy", str(taxonomy_path),
            "--stage", "decode",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with (tmp_path / "mixed_stage_table.csv").open(newline="") as handle:
        stages = list(csv.DictReader(handle))
    with (tmp_path / "mixed_opreator_table.csv").open(newline="") as handle:
        overview_reader = csv.DictReader(handle)
        overview_fields = overview_reader.fieldnames
        overview = list(overview_reader)

    # Numbered rows are per-position; the P-prefixed rows are the cross-position
    # rollup for one variant and deliberately leave position/variant blank.
    inputs = [
        row for row in stages
        if row["功能模块"] == "Input" and row["序号"].isdigit()
    ]
    assert [(row["单元位置"], row["单元类型"], row["模块耗时(us)"]) for row in inputs] == [
        ("1", "KDA", "10.000"),
        ("2", "MLA", "30.000"),
    ]
    assert {
        row["功能模块"]: row["模块耗时(us)"]
        for row in stages if row["序号"].startswith("P")
    } == {"Input": "40.000", "MLA core": "40.000", "KDA core": "20.000"}
    assert {row["单元类型"] for row in stages if row["序号"].isdigit()} == {
        "KDA", "MLA",
    }
    assert stages[-1]["序号"] == "总计"
    assert stages[-1]["模块耗时(us)"] == "100.000"
    assert overview[-1]["序号"] == "总计"
    assert overview[-1]["算子耗时(us)"] == "100.000"
    assert overview_fields[overview_fields.index("算子名称") - 1] == "module"

    for suffix in (
        "_operator_origin_table.csv",
        "_opreator_table.csv",
        "_core_compute_table.csv",
        "_auxiliary_operator_table.csv",
        "_op_classification_table.csv",
        "_stage_table.csv",
    ):
        with (tmp_path / f"mixed{suffix}").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if suffix == "_operator_origin_table.csv":
            assert rows[-1]["module"] == "__layer_total__"
        else:
            assert rows[-1]["序号"] == "总计"
