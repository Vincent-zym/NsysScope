"""Contract tests for the forward-pipeline table on a fabricated nsys export.

The trace is synthetic on purpose: it encodes the two failure modes that were
found on real captures -- a marker that fires once per repeating unit instead of
once per forward, and a layer breakdown that has to close against the step -- so
the guards can be tested without shipping a multi-hundred-MB sqlite file.
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sqlite3
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILDER = ROOT / "scripts" / "build_forward_pipeline_table.py"

STEPS = 6
UNITS_PER_STEP = 2           # two KKKM cycles per forward
PATTERN = ("KDA", "KDA", "KDA", "MLA")
US = 1000                    # ns per us


def load_validator():
    path = ROOT / "scripts" / "validate_analysis_package.py"
    spec = importlib.util.spec_from_file_location("package_validator", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def fabricate_trace(path: Path) -> None:
    """Write a minimal CUPTI-shaped sqlite with STEPS identical forward steps."""
    con = sqlite3.connect(path)
    con.executescript(
        "create table StringIds (id integer primary key, value text);"
        "create table CUPTI_ACTIVITY_KIND_KERNEL ("
        "  start integer, end integer, deviceId integer, shortName integer,"
        "  gridX integer, graphId integer, streamId integer);"
    )
    names: dict[str, int] = {}

    def name_id(value: str) -> int:
        if value not in names:
            names[value] = len(names) + 1
            con.execute("insert into StringIds values (?, ?)", (names[value], value))
        return names[value]

    rows: list[tuple[int, int, int, int, int, None, int]] = []
    cursor = 10 * US

    def emit(value: str, duration_us: int, grid: int = 1) -> None:
        nonlocal cursor
        rows.append((cursor, cursor + duration_us * US, 0, name_id(value), grid, None, 7))
        cursor += duration_us * US

    for _ in range(STEPS):
        emit("step_marker_kernel", 5)          # once per forward
        for _ in range(UNITS_PER_STEP):
            emit("unit_gate_kernel", 4)        # once per repeating unit -- the trap
            for variant in PATTERN:
                emit("attn_res_fused_tma_kernel", 2)
                emit("kda_core" if variant == "KDA" else "mla_core", 40)
                emit("attn_res_fused_tma_kernel", 2)
                emit("moe_experts_kernel", 30)
        emit("lm_head_kernel", 12)             # forward tail, becomes 其他
        cursor += 80 * US                      # inter-step bookkeeping hole

    con.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?, ?)", rows,
    )
    con.commit()
    con.close()


def write_taxonomy(path: Path) -> None:
    """Minimal taxonomy: the per-unit guard needs the declared unit ratio."""
    path.write_text(json.dumps({
        "schema_version": "1.0",
        "model": "SyntheticCycleModel",
        "repeating_unit": {
            "kind": "composite",
            "boundary_evidence": {"layer_start_kernel": ["attn_res_fused_tma_kernel"]},
            "positions": [
                {"position": i + 1, "unit_id": f"layer.{i}", "unit_variant": variant}
                for i, variant in enumerate(PATTERN)
            ],
        },
        "variants": [
            {"name": "KDA", "trace_marker_kernels": ["kda_core"]},
            {"name": "MLA", "trace_marker_kernels": ["mla_core"]},
        ],
    }))


def build(tmp_path: Path, *extra: str, taxonomy: bool = True) -> tuple[Path, dict]:
    trace = tmp_path / "trace.sqlite"
    if not trace.exists():
        fabricate_trace(trace)
    manifest = tmp_path / "manifest.json"
    command = [
        sys.executable, str(BUILDER),
        "--sqlite", str(trace),
        "--output-dir", str(tmp_path),
        "--prefix", "synthetic",
        "--device", "0",
        "--variant-marker", "KDA=kda_core",
        "--variant-marker", "MLA=mla_core",
        "--manifest-out", str(manifest),
    ]
    if taxonomy:
        taxonomy_path = tmp_path / "taxonomy.json"
        write_taxonomy(taxonomy_path)
        command += ["--taxonomy", str(taxonomy_path)]
    completed = subprocess.run(command + list(extra), capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(manifest.read_text())["forward_pipeline"]
    return tmp_path / "synthetic_forward_pipeline_table.csv", payload


def test_per_unit_marker_is_replaced_by_a_per_forward_one(tmp_path: Path):
    # unit_gate_kernel fires UNITS_PER_STEP times per step, so its launch count
    # equals the number of repeating units -- accepting it would report a
    # structural period as a forward step.
    _, manifest = build(tmp_path, "--step-marker", "unit_gate_kernel")
    assert manifest["step_marker"]["pattern"] != "unit_gate_kernel"
    assert manifest["step_marker"]["step_count"] == STEPS
    assert manifest["marker_auto_selected"]["reason"] == "previous marker was per-unit"


def test_per_unit_guard_needs_a_taxonomy(tmp_path: Path):
    # Documented limitation: without a taxonomy there is no declared unit ratio to
    # divide by, so the per-unit marker cannot be recognised. The runner always
    # passes one; a manual invocation should be aware of this.
    _, manifest = build(
        tmp_path, "--step-marker", "unit_gate_kernel", taxonomy=False,
    )
    assert manifest["step_marker"]["pattern"] == "unit_gate_kernel"
    assert manifest["step_marker"]["step_count"] == STEPS * UNITS_PER_STEP


def test_layer_breakdown_closes_and_counts_every_unit(tmp_path: Path):
    table, manifest = build(tmp_path)
    errors: list[str] = []
    load_validator().validate_forward_pipeline(table, errors)
    assert errors == []

    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    variants = {row["环节"]: row for row in rows if row["环节类型"] == "variant"}
    assert int(variants["KDA 层"]["层数"]) == 3 * UNITS_PER_STEP
    assert int(variants["MLA 层"]["层数"]) == 1 * UNITS_PER_STEP
    assert manifest["layers_per_step"] == len(PATTERN) * UNITS_PER_STEP
    # single device, so the per-rank shard warning must stay off
    assert manifest["layer_shard_note"] is None


def test_chunk_size_is_carried_from_the_launch_command(tmp_path: Path):
    _, manifest = build(tmp_path, "--chunk-size", "4096")
    assert manifest["chunk_size"] == 4096


def test_gap_only_counts_holes_above_the_threshold(tmp_path: Path):
    # the fabricated hole between steps is 80us
    _, loose = build(tmp_path, "--gap-threshold-us", "50")
    strict_dir = tmp_path / "strict"
    strict_dir.mkdir()
    (strict_dir / "trace.sqlite").write_bytes((tmp_path / "trace.sqlite").read_bytes())
    _, strict = build(strict_dir, "--gap-threshold-us", "500")
    assert loose["gap_holes"] and not strict["gap_holes"]


# --- server-shaped capture ------------------------------------------------------
# A serving trace is not a benchmark loop: the step period jitters with the chunk
# fill, the marker's gridX follows the per-step token count, and each pipeline rank
# holds a different slice of the layer stack. All three broke the builder on a real
# prefill capture, so they get their own fabricated trace.

SERVER_STEPS = 8
JITTER = (1.0, 1.08, 0.93, 1.1, 0.95, 1.06, 0.92, 1.05)
# 2 populations that do *not* repeat as a draft/verify pattern
SERVER_GRIDS = (1, 1, 2, 1, 2, 1, 1, 1)
# rank 0 reproduces the declared unit, rank 1 does not -- and rank 1 is busier
RANK_PATTERNS = {0: PATTERN, 1: ("KDA", "KDA", "MLA", "MLA")}


def fabricate_server_trace(path: Path) -> None:
    con = sqlite3.connect(path)
    con.executescript(
        "create table StringIds (id integer primary key, value text);"
        "create table CUPTI_ACTIVITY_KIND_KERNEL ("
        "  start integer, end integer, deviceId integer, shortName integer,"
        "  gridX integer, graphId integer, streamId integer);"
    )
    names: dict[str, int] = {}

    def name_id(value: str) -> int:
        if value not in names:
            names[value] = len(names) + 1
            con.execute("insert into StringIds values (?, ?)", (names[value], value))
        return names[value]

    rows: list[tuple[int, int, int, int, int, None, int]] = []
    for device, pattern in RANK_PATTERNS.items():
        cursor = 10 * US

        def emit(value: str, duration_us: float, grid: int = 1) -> None:
            nonlocal cursor
            span = int(duration_us * US)
            rows.append((cursor, cursor + span, device, name_id(value), grid, None, 7))
            cursor += span

        for step in range(SERVER_STEPS):
            scale = JITTER[step]
            emit("compute_position_kernel", 5 * scale, SERVER_GRIDS[step])
            for _ in range(UNITS_PER_STEP):
                for variant in pattern:
                    emit("attn_res_fused_tma_kernel", 2)
                    emit("kda_core" if variant == "KDA" else "mla_core", 40 * scale)
                    emit("attn_res_fused_tma_kernel", 2)
                    emit("moe_experts_kernel", 30 * scale)
                    if device:  # make the mismatching rank the busiest one
                        emit("filler_kernel", 5 * scale)
            emit("ncclDevKernel_SendRecv", 3)
            emit("lm_head_kernel", 12 * scale)

    con.executemany(
        "insert into CUPTI_ACTIVITY_KIND_KERNEL values (?, ?, ?, ?, ?, ?, ?)", rows,
    )
    con.commit()
    con.close()


def build_server(tmp_path: Path, *extra: str) -> tuple[Path, dict]:
    """Same as build(), without pinning a device: rank choice is under test."""
    trace = tmp_path / "trace.sqlite"
    if not trace.exists():
        fabricate_server_trace(trace)
    taxonomy_path = tmp_path / "taxonomy.json"
    write_taxonomy(taxonomy_path)
    manifest = tmp_path / "manifest.json"
    completed = subprocess.run([
        sys.executable, str(BUILDER),
        "--sqlite", str(trace),
        "--output-dir", str(tmp_path),
        "--prefix", "server",
        "--variant-marker", "KDA=kda_core",
        "--variant-marker", "MLA=mla_core",
        "--taxonomy", str(taxonomy_path),
        "--manifest-out", str(manifest),
        *extra,
    ], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(manifest.read_text())["forward_pipeline"]
    return tmp_path / "server_forward_pipeline_table.csv", payload


def test_jittery_step_period_is_accepted_on_kernel_consensus(tmp_path: Path):
    # No kernel is evenly spaced enough for the strict cv gate, but several
    # independent per-forward kernels agree on the launch count, so the marker is
    # still trustworthy -- rejecting it dropped the table from a real prefill job.
    _, manifest = build_server(tmp_path)
    auto = manifest["marker_auto_selected"]
    assert manifest["step_marker"]["step_count"] == SERVER_STEPS
    assert auto["cv"] > 0.05 and auto["cv_relaxed"] is True
    assert auto["agreeing_kernels"] >= 3


def test_varying_marker_grid_is_not_speculative(tmp_path: Path):
    # gridX tracks the per-step token count in prefill; only a fixed
    # K-draft-then-verify pattern may be read as speculative decoding.
    table, manifest = build_server(tmp_path)
    assert manifest["speculative"] is False
    assert manifest["speculative_tokens"] == 0
    with table.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert not [row for row in rows if row["环节"] == "draft 模型"]


def test_pipeline_rank_is_chosen_by_variant_mix(tmp_path: Path):
    # Rank 1 has more kernels but a 2:2 KDA/MLA mix; only rank 0 reproduces the
    # declared 3:1 unit, so labelling rank 1 with this taxonomy would be wrong.
    _, manifest = build_server(tmp_path)
    assert manifest["device"] == 0
    assert manifest["device_candidates"][0] == 0
    assert manifest["gpu_count"] == len(RANK_PATTERNS)
    assert "device 0" in manifest["layer_shard_note"]
