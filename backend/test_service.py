from __future__ import annotations

import csv
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest import mock

import pytest

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.models import BUILTIN_MODEL_CONFIGS, JobCreate
from backend.runner import AGENT_CSV_SUFFIXES, JobRunner
from backend.store import JobStore


PROJECT = Path(__file__).resolve().parents[1]
SKILL = PROJECT / "bundled" / "skills" / "sglang-nsys-static-analysis"
# The frontend converter lives in the Skill so a standalone Skill run produces the
# same analysis.json the tool does. The Skill directory name is not an importable
# module path, so load it by file.
_converter_spec = importlib.util.spec_from_file_location(
    "nsysscope_build_analysis_json", SKILL / "scripts" / "build_analysis_json.py",
)
assert _converter_spec is not None and _converter_spec.loader is not None
build_analysis_json = importlib.util.module_from_spec(_converter_spec)
_converter_spec.loader.exec_module(build_analysis_json)
build_operator_payload = build_analysis_json.build_operator_payload
included_devices = build_analysis_json.included_devices
is_total_row = build_analysis_json.is_total_row
metadata_directory = build_analysis_json.metadata_directory
parallel_config = build_analysis_json.parallel_config
stable_sample_count = build_analysis_json.stable_sample_count
# A real seven-table package to exercise import/conversion against. It is too
# large to commit, so point NSYSSCOPE_TEST_PACKAGE at one; the tests that need it
# skip loudly rather than passing vacuously when it is absent.
PACKAGE = Path(
    os.getenv("NSYSSCOPE_TEST_PACKAGE", str(PROJECT / "does-not-exist" / "package"))
).expanduser()


def require_package() -> None:
    if not PACKAGE.exists():
        pytest.skip(
            "set NSYSSCOPE_TEST_PACKAGE to a seven-table result package to run this test",
        )


def package_tables() -> list[Path]:
    """The package's CSV tables, wherever the producing run put them."""
    root = PACKAGE / "csv" if (PACKAGE / "csv").is_dir() else PACKAGE
    return sorted(root.glob("*_table.csv"))


def package_prefix() -> str:
    """The prefix the package was produced with, read off the origin table."""
    for table in package_tables():
        if table.name.endswith("_operator_origin_table.csv"):
            return table.name[: -len("_operator_origin_table.csv")]
    raise AssertionError(f"no *_operator_origin_table.csv in {PACKAGE}")


def package_operator_count() -> int:
    """Operator rows in the origin table, excluding its total row.

    The origin table marks its total with `module=__layer_total__` rather than
    `序号=总计`, so is_total_row alone would count it as an operator.
    """
    origin = next(
        table for table in package_tables()
        if table.name.endswith("_operator_origin_table.csv")
    )
    with origin.open(newline="", encoding="utf-8") as handle:
        return sum(
            1 for row in csv.DictReader(handle)
            if not is_total_row(row) and row.get("module") != "__layer_total__"
        )


def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        allowed_roots=(PACKAGE.parent, tmp_path),
        api_token="test-token",
        cors_origins=("http://localhost:3000",),
        max_workers=1,
        codex_enabled=False,
        codex_bin="codex",
        comate_enabled=False,
        comate_bin="",
        comate_username="",
        comate_model="",
        comate_platform="internal",
        comate_timeout_seconds=120,
        comate_store_dir=tmp_path / "comate-store",
        agent_heartbeat_seconds=30,
        agent_stall_timeout_seconds=0,
        job_log_max_bytes=1024 * 1024,
        job_log_line_max_bytes=16 * 1024,
        nsys_bin="nsys",
        skill_dir=PROJECT / "bundled" / "skills" / "sglang-nsys-static-analysis",
        call_tree_skill_dir=(
            PROJECT / "bundled" / "skills" / "reconstruct-profiler-call-tree"
        ),
        converter=SKILL / "scripts/build_analysis_json.py",
        xlsx_converter=SKILL / "scripts/csv_to_xlsx.py",
        subprocess_timeout_seconds=600,
        popo_username="",
        popo_upload_script=PROJECT / "does-not-exist" / "upload.py",
        builtin_model_configs=dict(BUILTIN_MODEL_CONFIGS),
    )


def package_copy(tmp_path: Path, *, tables_only: bool = False) -> Path:
    """A flat copy of the fixture package, whatever layout it shipped in.

    A package produced by the agent keeps its tables in csv/ and its sidecars in
    metadata/; an older one is flat. Flattening here is what lets these tests run
    against any real result directory instead of one specific one. `tables_only`
    keeps the manifest, without which validate_sampling rejects the package, and
    drops the statistics/semantic/validation sidecars.
    """
    target = tmp_path / ("tables-only" if tables_only else "package")
    target.mkdir()
    for table in package_tables():
        shutil.copy2(table, target / table.name)
    sidecars = [*PACKAGE.glob("*.json"), *(PACKAGE / "metadata").glob("*.json")]
    if tables_only:
        sidecars = [
            path for path in sidecars if path.name.endswith("_analysis_manifest.json")
        ]
    for source in sidecars:
        if source.is_file():
            shutil.copy2(source, target / source.name)
    return target


def test_existing_package_job(tmp_path: Path) -> None:
    require_package()
    package = package_copy(tmp_path)
    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(package),
        "prefix": package_prefix(),
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    analysis = client.get(job["analysis_url"], headers=headers)
    assert analysis.status_code == 200
    payload = analysis.json()
    assert payload["schemaVersion"] == "1.0"
    assert len(payload["operators"]) == package_operator_count()
    assert payload["summary"]["stableSamples"] >= 1


def test_importing_a_canonical_skill_only_package_succeeds(tmp_path: Path) -> None:
    # The Skill is meant to be usable standalone, with the result imported afterwards
    # -- so a directory that finalize_package.py already laid out (csv/, xlsx/,
    # metadata/, trace/, analysis.json, final_report.md, nsysscope-package.json, and
    # crucially NO logs/ or metadata/context.json, since those are job-only) must be
    # accepted by "existing_package" mode exactly as a job-produced one is.
    require_package()
    package = package_copy(tmp_path)  # already has csv/, xlsx/, metadata/, analysis.json
    # package_copy does not carry final_report.md (most callers don't need it);
    # this test is specifically about not overwriting one that exists.
    shutil.copy2(PACKAGE / "final_report.md", package / "final_report.md")
    completed = subprocess.run(
        [
            sys.executable, str(SKILL / "scripts" / "finalize_package.py"),
            str(package), "--prefix", package_prefix(),
        ],
        capture_output=True, text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (package / "logs").exists()
    # The fixture package is job-produced and keeps its context.json; a genuinely
    # skill-only run would not have one. What matters here is that
    # finalize_package.py does not invent a "log" manifest entry for a log that
    # does not exist in this copy.
    manifest = json.loads((package / "nsysscope-package.json").read_text())
    assert "log" not in manifest

    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(package),
        "prefix": package_prefix(),
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    analysis = client.get(job["analysis_url"], headers=headers)
    assert analysis.status_code == 200
    assert analysis.json()["schemaVersion"] == "1.0"
    # ensure_final_report never overwrites an existing report -- confirm the import
    # kept the fixture's own file rather than regenerating a fresh skeleton over it.
    assert (package / "final_report.md").read_text() == (
        PACKAGE / "final_report.md"
    ).read_text()


def test_an_unresolvable_prefix_fails_the_job_instead_of_silently_succeeding(
    tmp_path: Path,
) -> None:
    # An existing_package import whose csv/ holds no table set matching any prefix
    # used to be swallowed: detect_prefix's RuntimeError was caught, the whole
    # repackaging/validation block was skipped, and the job reported "succeeded"
    # having done nothing to the pre-existing analysis.json.
    package = tmp_path / "unresolvable"
    csv_dir = package / "csv"
    csv_dir.mkdir(parents=True)
    for suffix in AGENT_CSV_SUFFIXES:
        (csv_dir / f"model_a{suffix}").write_text("a,b\n1,2\n", encoding="utf-8")
        (csv_dir / f"model_b{suffix}").write_text("a,b\n1,2\n", encoding="utf-8")
    (package / "analysis.json").write_text(
        json.dumps({"schemaVersion": "1.0", "operators": []}) + "\n", encoding="utf-8",
    )

    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(package),
        "prefix": "requested",
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "failed", job
    assert "cannot resolve table prefix" in (job.get("error") or "")


def test_auth_and_path_boundary(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert "NsysScope" in dashboard.text
    assert dashboard.headers["cache-control"] == "no-store"
    assert "window.__NSYSSCOPE_LOCAL__ = true" in dashboard.text
    assert client.get("/api/health").status_code == 200
    assert client.get("/analyzer-api/api/health").status_code == 200
    unauthorized = client.get("/api/jobs")
    assert unauthorized.status_code == 401
    rejected = client.post("/api/jobs", headers={"X-NsysScope-Token": "test-token"}, json={
        "mode": "existing_package",
        "model_name": "test",
        "stage": "prefill",
        "hardware": "test",
        "existing_package_path": "/etc",
    })
    assert rejected.status_code == 422


def test_existing_analysis_json_directory_import(tmp_path: Path) -> None:
    package = tmp_path / "analysis-only"
    package.mkdir()
    shutil.copy2(PROJECT / "public/demo-analysis.json", package / "analysis.json")
    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "Imported analysis",
        "stage": "prefill",
        "hardware": "Unknown",
        "existing_package_path": str(package),
        "prefix": "does-not-exist",
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job


def test_existing_six_tables_import_without_sidecars(tmp_path: Path) -> None:
    require_package()
    package = package_copy(tmp_path, tables_only=True)
    tables = len(list(package.glob("*.csv")))
    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(package),
        "prefix": package_prefix(),
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    assert len(list((package / "xlsx").glob("*.xlsx"))) == tables
    payload = json.loads((package / "analysis.json").read_text())
    assert payload["metadata"]["model"] == "GLM5.2"
    # The sample count survives without the statistics sidecar because the
    # manifest carries it; the converter only falls back to 1 when neither is
    # present, and validate_sampling rejects a package missing the manifest.
    assert payload["summary"]["stableSamples"] >= 1
    assert sum(item["count"] for item in payload["classifications"]) == (
        package_operator_count()
    )
    by_name = {
        operator["kernelName"]: operator["category"]
        for operator in payload["operators"]
    }
    assert set(by_name.values()) <= {"core", "communication", "auxiliary"}
    # A quantisation or norm helper is never core compute, whatever else the
    # package contains: mis-classifying one inflates the core-compute table.
    helpers = [
        (name, category) for name, category in by_name.items()
        if name.startswith(("per_token_group_quant", "generalLayerNorm", "rmsnorm"))
    ]
    assert all(category == "auxiliary" for _, category in helpers), helpers


def test_external_skill_selection_uses_lightweight_pointer(tmp_path: Path) -> None:
    external = tmp_path / "external-skill"
    shutil.copytree(PROJECT / "bundled/skills/sglang-nsys-static-analysis", external)
    environment = {
        **os.environ,
        "NSYSSCOPE_CONFIG_DIR": str(tmp_path / "config"),
    }
    manager = PROJECT / "scripts/skill_manager.py"
    selected = subprocess.run(
        ["python3", str(manager), "use", str(external)],
        check=True, text=True, capture_output=True, env=environment,
    )
    assert "已切换到外部 Skill" in selected.stdout
    resolved = subprocess.run(
        ["python3", str(manager), "resolve"],
        check=True, text=True, capture_output=True, env=environment,
    )
    assert Path(resolved.stdout.strip()) == external
    config = json.loads((tmp_path / "config/config.json").read_text())
    assert config["skill_dir"] == str(external)
    assert len(config["skill_sha256"]) == 64


def test_zip_six_table_import_writes_only_to_selected_result(tmp_path: Path) -> None:
    require_package()
    archive_path = tmp_path / "shared-result.zip"
    tables = package_tables()
    with zipfile.ZipFile(archive_path, "w") as archive:
        for source in tables:
            archive.write(source, f"shared/csv/{source.name}")
    result_path = tmp_path / "imported-result"
    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(archive_path),
        "result_path": str(result_path),
        "prefix": package_prefix(),
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    assert len(list((result_path / "csv").glob("*.csv"))) == len(tables)
    assert len(list((result_path / "xlsx").glob("*.xlsx"))) == len(tables)
    assert (result_path / "analysis.json").exists()
    assert (result_path / "nsysscope-package.json").exists()
    assert not list(tmp_path.glob("import-*"))


def test_zip_import_of_a_canonical_skill_package_keeps_its_report(
    tmp_path: Path,
) -> None:
    # Zip up exactly what finalize_package.py produces (csv/, xlsx/, metadata/,
    # trace/, analysis.json, final_report.md, nsysscope-package.json, no logs/) and
    # confirm import_zip_package carries the report through instead of silently
    # dropping it -- the ZIP path never globs for *.md, so this only works if the
    # report is explicitly picked up.
    require_package()
    source_dir = flat_analysis_dir(tmp_path / "source", prefix=package_prefix())
    for table in package_tables():
        shutil.copy2(table, source_dir / table.name)
    (source_dir / "final_report.md").write_text(
        "# a real report, not a skeleton\n", encoding="utf-8",
    )
    subprocess.run(
        [
            sys.executable, str(SKILL / "scripts" / "finalize_package.py"),
            str(source_dir), "--prefix", package_prefix(), "--no-xlsx",
        ],
        check=True, capture_output=True, text=True,
    )
    assert not (source_dir / "logs").exists()

    archive_path = tmp_path / "skill-package.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for path in source_dir.rglob("*"):
            if path.is_file():
                archive.write(path, f"result/{path.relative_to(source_dir).as_posix()}")

    result_path = tmp_path / "imported-result"
    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(archive_path),
        "result_path": str(result_path),
        "prefix": package_prefix(),
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    assert (result_path / "final_report.md").read_text() == (
        "# a real report, not a skeleton\n"
    )
    manifest = json.loads((result_path / "nsysscope-package.json").read_text())
    # The job itself writes to logs/job.log before extraction (the "converting"
    # state transition), so the manifest legitimately gets a "log" key here --
    # unlike finalize_package.py run by hand, which never has one to point at.
    # What matters is that the key, when present, points at a real file.
    if "log" in manifest:
        assert (result_path / manifest["log"]).is_file()


def test_converter_accepts_current_stats_schema() -> None:
    stats = {
        "accepted_full_template_sample_count": 1188,
        "per_device_sample_counts": {"0": 147, "1": 156},
    }
    assert stable_sample_count(stats, {}) == 1188
    assert included_devices(stats, {}) == [0, 1]


def test_converter_recognizes_generated_total_rows() -> None:
    assert is_total_row({"序号": "总计", "算子名称": "总计"})
    assert not is_total_row({"序号": "113", "算子名称": "add3_kernel"})


def test_metadata_is_found_whatever_the_agent_named_the_table_dir(tmp_path: Path) -> None:
    # The agent picks its own layout: tables landed in result/ on one run and in csv/
    # on another. Inferring the metadata dir from the table dir's name lost PP/TP and
    # the input report without any error -- the page just stopped showing them.
    job = tmp_path / "job"
    (job / "metadata").mkdir(parents=True)
    (job / "metadata" / "context.json").write_text("{}", encoding="utf-8")
    for name in ("csv", "result", "tables"):
        (job / name).mkdir()
        assert metadata_directory(job / name) == job / "metadata"
    # Tables straight in the job dir, and a package with no metadata at all.
    assert metadata_directory(job) == job / "metadata"
    bare = tmp_path / "bare"
    bare.mkdir()
    assert metadata_directory(bare) == bare


def test_chunked_prefill_size_reads_a_shell_arithmetic_expression(tmp_path: Path) -> None:
    # p_start.sh writes `--chunked-prefill-size $((32 * 1024))`, not a bare integer;
    # the plain-digit regex silently returned None and Chunk Size vanished from the
    # frontend without any error.
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "sglang.launch_server \\\n"
        "    --chunked-prefill-size $((32 * 1024)) \\\n"
        "    --other-flag 1\n",
        encoding="utf-8",
    )
    (metadata_dir / "context.json").write_text(
        json.dumps({"launch_path": str(launch)}), encoding="utf-8",
    )
    assert JobRunner.chunked_prefill_size(metadata_dir) == 32768

    # A plain integer literal still works.
    launch.write_text("--chunked-prefill-size 16384\n", encoding="utf-8")
    assert JobRunner.chunked_prefill_size(metadata_dir) == 16384

    # Anything beyond arithmetic on literals (e.g. a shell variable) is refused
    # rather than evaluated.
    launch.write_text("--chunked-prefill-size $((SOME_VAR * 1024))\n", encoding="utf-8")
    assert JobRunner.chunked_prefill_size(metadata_dir) is None


def test_parallel_config_resolves_flags_set_from_earlier_variables(tmp_path: Path) -> None:
    # p_start.sh does not write literals: PP_SIZE/TP_SIZE are their own variables,
    # assigned from a still-earlier NUM_NODES, and the flag then quotes the variable
    # rather than a number. Matching only `--pp-size \d+` returned {} silently and
    # the frontend's PARALLELISM row just went missing.
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "NUM_NODES=2\n"
        "export NUM_NODES\n"
        "PP_SIZE=$((NUM_NODES * 1))\n"
        "export PP_SIZE\n"
        "TP_SIZE=$((NUM_NODES * 4))\n"
        "export TP_SIZE\n"
        "sglang.launch_server \\\n"
        '    --tp-size "$TP_SIZE" \\\n'
        '    --pp-size "$PP_SIZE"\n',
        encoding="utf-8",
    )
    (metadata_dir / "context.json").write_text(
        json.dumps({"launch_path": str(launch)}), encoding="utf-8",
    )
    assert parallel_config(metadata_dir) == {"PP": 2, "TP": 8}

    # A literal flag value still works, unresolved variables are dropped rather
    # than guessed, and the two behave the same within one script.
    launch.write_text(
        '--pp-size "$UNDECLARED_VAR" --tp-size 4\n', encoding="utf-8",
    )
    assert parallel_config(metadata_dir) == {"TP": 4}


def test_parallel_config_resolves_a_bare_variable_reference_chain(tmp_path: Path) -> None:
    # d_start.sh (kimi3_decode_analysis_0817_1) goes one indirection further than
    # p_start.sh: TP_SIZE is not its own $((...)) expression, it is a *bare*
    # reference to another variable (TP_SIZE=$TOTAL_GPUS), and TOTAL_GPUS is
    # itself combined with `export` on the same line
    # (`export TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))`). Both of these silently
    # produced {} before: resolve_shell_int only stripped the $((...)) wrapper, so
    # a bare `$VAR` token kept its leading `$` and failed the numeric-only check,
    # and shell_assignments' regex did not match a leading `export `.
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    launch = tmp_path / "launch.sh"
    launch.write_text(
        "NUM_NODES=2\n"
        "GPUS_PER_NODE=8\n"
        "export TOTAL_GPUS=$((NUM_NODES * GPUS_PER_NODE))\n"
        "DP_SIZE=$NUM_NODES\n"
        "export DP_SIZE\n"
        "TP_SIZE=$TOTAL_GPUS\n"
        "export TP_SIZE\n"
        "EP_SIZE=$TOTAL_GPUS\n"
        "export EP_SIZE\n"
        "sglang serve \\\n"
        '    --tp-size "$TP_SIZE" \\\n'
        '    --dp-size "$DP_SIZE" \\\n'
        '    --ep-size "$EP_SIZE"\n',
        encoding="utf-8",
    )
    (metadata_dir / "context.json").write_text(
        json.dumps({"launch_path": str(launch)}), encoding="utf-8",
    )
    assert parallel_config(metadata_dir) == {"TP": 16, "DP": 2, "EP": 16}


def test_parallel_config_recognizes_dcp_size(tmp_path: Path) -> None:
    # d_start.sh also passes --dcp-size (decode context parallel), which the
    # PP/TP/DP/EP-only flags dict used to silently drop.
    metadata_dir = tmp_path / "metadata"
    metadata_dir.mkdir()
    launch = tmp_path / "launch.sh"
    launch.write_text('--tp-size 16 --dcp-size 8\n', encoding="utf-8")
    (metadata_dir / "context.json").write_text(
        json.dumps({"launch_path": str(launch)}), encoding="utf-8",
    )
    assert parallel_config(metadata_dir) == {"TP": 16, "DCP": 8}


def test_frontend_payload_repairs_legacy_semantic_operator_alias() -> None:
    raw = {
        "序号": "7",
        "module": "attention/q_projection",
        "operator_name": "void deep_gemm::very_long_raw_cuda_symbol<int>()",
        "duration_min_us": "10",
        "duration_max_us": "14",
        "duration_diff_us": "4",
        "start_ns": "1000",
        "end_ns": "1300",
        "device": "0",
        "stream": "7",
        "python_function": "model.forward -> q_proj",
        "mapping_reason": "source evidence",
    }
    overview = {
        "功能模块": "QKV 投影",
        "算子名称": "Q-B 投影",
        "算子耗时(us)": "12",
        "算子耗时占比(%)": "1.2",
        "shape": "(M=1,N=2,K=3)",
        "mfu": "60%",
        "功能介绍": "简化后的算子说明",
    }

    operator = build_operator_payload(raw, overview, {"category": "core"})

    assert operator["name"] == "Q-B 投影"
    assert operator["kernelName"] == "very_long_raw_cuda_symbol<int>"
    assert operator["stage"] == "QKV 投影"
    assert operator["fullName"] == raw["operator_name"]


def test_frontend_payload_preserves_validated_six_table_category() -> None:
    base = {
        "序号": "7",
        "module": "self_attn/indexer",
        "duration_min_us": "10",
        "duration_max_us": "14",
        "duration_diff_us": "4",
        "start_ns": "1000",
        "end_ns": "1300",
        "device": "0",
        "stream": "7",
        "python_function": "model.forward -> indexer",
        "mapping_reason": "source evidence",
    }
    view = {
        "功能模块": "NSA Indexer",
        "算子耗时(us)": "12",
        "算子耗时占比(%)": "1.2",
        "shape": "",
        "mfu": "",
        "功能介绍": "helper",
    }
    for kernel in (
        "per_token_group_quant_8bit_kernel",
        "void flashinfer::norm::generalLayerNorm<float>(float*)",
        "kernel_cutlass_kda_decode_mtp_kernel_TiledCopy_CopyAtom",
    ):
        operator = build_operator_payload(
            {**base, "operator_name": kernel},
            {**view, "算子名称": kernel},
            {"category": "core"},
        )
        assert operator["category"] == "core"


def test_frontend_payload_preserves_structural_unit_identity() -> None:
    raw = {
        "序号": "44",
        "module": "layers.7/attention/mla_core",
        "operator_name": "mla_decode_kernel",
        "duration_min_us": "170",
        "duration_max_us": "190",
        "duration_diff_us": "20",
        "start_ns": "1000",
        "end_ns": "181000",
        "device": "0",
        "stream": "7",
        "layer_id": "7",
        "unit_position": "4",
        "unit_id": "layer.7",
        "unit_variant": "MLA+LatentMoE",
        "python_function": "KimiK3MLAAttention.forward",
        "mapping_reason": "current-model source and trace",
    }
    view = {
        "单元位置": "4",
        "单元ID": "layer.7",
        "单元类型": "MLA+LatentMoE",
        "功能模块": "MLA Decode 核心",
        "算子名称": "mla_decode_kernel",
        "算子耗时(us)": "180",
        "算子耗时占比(%)": "8.7",
        "shape": "",
        "mfu": "",
        "功能介绍": "MLA decode attention core",
    }

    operator = build_operator_payload(raw, view, {"category": "core"})

    assert operator["unitPosition"] == 4
    assert operator["unitId"] == "layer.7"
    assert operator["unitVariant"] == "MLA+LatentMoE"
    assert operator["layerId"] == 7
    assert operator["stageKey"] == "MLA Decode 核心"


def test_validate_analysis_rejects_collapsed_heterogeneous_cycle(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    path = tmp_path / "analysis.json"
    path.write_text(json.dumps({
        "schemaVersion": "1.0",
        "summary": {
            "operatorCount": 1,
            "devices": [0],
            "stableSamples": 1,
            "heterogeneous": True,
            "distinctUnitVariants": ["KDA", "MLA"],
            "durationLabel": "平均单层耗时",
        },
        "operators": [{
            "category": "core",
            "unitPosition": 1,
            "unitId": "layer.4",
            "unitVariant": "KDA",
        }],
    }))

    try:
        runner.validate_analysis(path)
    except RuntimeError as exc:
        assert "loses heterogeneous unit variants" in str(exc)
    else:
        raise AssertionError("collapsed heterogeneous analysis should be rejected")


def test_log_pagination_and_conversion_retry(tmp_path: Path) -> None:
    require_package()
    package = package_copy(tmp_path)
    application = create_app(settings(tmp_path))
    client = TestClient(application)
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(package),
        "prefix": package_prefix(),
    })
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job

    log_path = application.state.runner.job_log_path(Path(job["output_dir"]))
    with log_path.open("a") as handle:
        handle.writelines(f"line-{index}\n" for index in range(7))
    page = client.get(
        f"/api/jobs/{job['id']}/logs?after=0&limit=2", headers=headers,
    ).json()
    assert page["next"] > 0
    assert page["has_more"] is True
    assert len(page["lines"]) == 2
    next_page = client.get(
        f"/api/jobs/{job['id']}/logs?after={page['next']}&limit=2",
        headers=headers,
    ).json()
    assert next_page["after"] == page["next"]
    assert next_page["lines"]

    Path(job["output_dir"], "analysis.json").unlink()
    application.state.store.update(
        job["id"], status="failed", progress=100, message="simulated converter failure",
    )
    retried = client.post(
        f"/api/jobs/{job['id']}/retry-conversion", headers=headers,
    )
    assert retried.status_code == 200, retried.text
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    assert Path(job["output_dir"], "analysis.json").exists()


def test_comate_provider_end_to_end_with_fake_zulu(tmp_path: Path) -> None:
    require_package()
    # The fake agent copies a flattened package into its --cwd, so this test does
    # not care how the fixture package arranges its tables and sidecars.
    staged = package_copy(tmp_path)
    fake_zulu = tmp_path / "zulu"
    fake_zulu.write_text(textwrap.dedent(f"""\
        #!/usr/bin/env python3
        import json
        import os
        import shutil
        import sys
        from pathlib import Path

        if sys.argv[1] == "status":
            print("状态：已登录")
            raise SystemExit(0)
        if sys.argv[1:4] == ["model", "list", "--ids"]:
            print("* Auto（自动选择）  (auto)")
            print("  Quality Model  (quality-model-id)")
            print("  Fast Model  (fast-model-id)")
            raise SystemExit(0)
        if sys.argv[1] != "run":
            raise SystemExit(2)
        if any(key in os.environ for key in (
            "HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy",
            "ALL_PROXY", "all_proxy",
        )):
            raise SystemExit(3)
        if os.environ.get("PLATFORM") != "internal":
            raise SystemExit(4)
        cwd = Path(sys.argv[sys.argv.index("--cwd") + 1])
        (cwd / "zulu-args.json").write_text(json.dumps(sys.argv))
        for source in sorted(Path({str(staged)!r}).iterdir()):
            if source.is_file():
                shutil.copy2(source, cwd / source.name)
        print(json.dumps({{"type": "task-json", "message": "x" * 300000}}))
    """))
    fake_zulu.chmod(0o755)

    material = tmp_path / "material"
    source = material / "source"
    source.mkdir(parents=True)
    report = material / "report.sqlite"
    config = material / "config.json"
    launch = material / "launch.yaml"
    report.write_text("")
    config.write_text("{}")
    launch.write_text("model: test\n")

    configured = replace(
        settings(tmp_path),
        comate_enabled=True,
        comate_bin=str(fake_zulu),
    )
    application = create_app(configured)
    client = TestClient(application)
    headers = {"X-NsysScope-Token": "test-token"}
    catalog = client.get(
        "/api/providers/comate/models", headers=headers,
    )
    assert catalog.status_code == 200, catalog.text
    assert catalog.json()["models"] == [
        {"id": "quality-model-id", "label": "Quality Model"},
        {"id": "fast-model-id", "label": "Fast Model"},
    ]
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "codex_skill",
        "agent_provider": "comate",
        "agent_model": "quality-model-id",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "report_path": str(report),
        "config_path": str(config),
        "launch_path": str(launch),
        "source_path": str(source),
        "result_path": str(tmp_path / "result"),
        "prefix": package_prefix(),
    })
    assert response.status_code == 200, response.text
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job
    output = Path(job["output_dir"])
    assert not (output / ".comate").exists()
    assert "Use the `sglang-nsys-static-analysis` skill at:" in (
        output / "metadata/prompt.md"
    ).read_text()
    skill_info = json.loads((output / "metadata/skill.json").read_text())
    assert skill_info["name"] == "sglang-nsys-static-analysis"
    assert len(skill_info["sha256"]) == 64
    log_path = output / "logs/job.log"
    assert "<prompt>" in log_path.read_text()
    zulu_args = json.loads((output / "metadata/zulu-args.json").read_text())
    assert zulu_args[zulu_args.index("--model") + 1] == "quality-model-id"
    assert zulu_args[zulu_args.index("--display") + 1] == "task-json"
    job_log = log_path.read_text()
    assert "详细会话未写入日志" in job_log
    assert len(job_log.encode()) < 50_000
    expected_tables = len(package_tables())
    assert len(list((output / "csv").glob("*.csv"))) == expected_tables
    workbooks = list((output / "xlsx").glob("*.xlsx"))
    assert len(workbooks) == expected_tables
    for workbook in workbooks:
        with zipfile.ZipFile(workbook) as archive:
            assert "xl/worksheets/sheet1.xml" in archive.namelist()
    assert (output / "trace/report.sqlite").exists()
    assert (output / "nsysscope-package.json").exists()
    analysis = client.get(job["analysis_url"], headers=headers).json()
    assert analysis["schemaVersion"] == "1.0"
    assert len(analysis["operators"]) == package_operator_count()


def test_codex_model_catalog_from_local_cache(tmp_path: Path, monkeypatch) -> None:
    fake_codex = tmp_path / "codex"
    fake_codex.write_text(textwrap.dedent("""\
        #!/usr/bin/env sh
        if [ "$1" = "login" ] && [ "$2" = "status" ]; then
          echo "Logged in using ChatGPT"
          exit 0
        fi
        exit 2
    """))
    fake_codex.chmod(0o755)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "config.toml").write_text('model = "quality-codex"\n')
    (codex_home / "models_cache.json").write_text(json.dumps({
        "models": [
            {"slug": "quality-codex", "display_name": "Quality Codex"},
            {"slug": "fast-codex", "display_name": "Fast Codex"},
        ],
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    configured = replace(
        settings(tmp_path),
        codex_enabled=True,
        codex_bin=str(fake_codex),
    )
    client = TestClient(create_app(configured))
    response = client.get(
        "/api/providers/codex/models",
        headers={"X-NsysScope-Token": "test-token"},
    )
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "provider": "codex",
        "default_model": "quality-codex",
        "models": [
            {"id": "quality-codex", "label": "Quality Codex"},
            {"id": "fast-codex", "label": "Fast Codex"},
        ],
    }


def test_codex_model_catalog_prefers_cli_catalog(tmp_path: Path, monkeypatch) -> None:
    fake_codex = tmp_path / "codex"
    fake_codex.write_text(textwrap.dedent("""\
        #!/usr/bin/env python3
        import json
        import sys

        if sys.argv[1:] == ["login", "status"]:
            print("Logged in using ChatGPT")
            raise SystemExit(0)
        if sys.argv[1:] == ["debug", "models"]:
            print(json.dumps({"models": [
                {"slug": "live-codex", "display_name": "Live Codex", "visibility": "list"},
                {"slug": "hidden-codex", "display_name": "Hidden", "visibility": "hide"},
            ]}))
            raise SystemExit(0)
        raise SystemExit(2)
    """))
    fake_codex.chmod(0o755)
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "models_cache.json").write_text(json.dumps({
        "models": [{"slug": "stale-codex", "display_name": "Stale Codex"}],
    }))
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    configured = replace(
        settings(tmp_path),
        codex_enabled=True,
        codex_bin=str(fake_codex),
    )
    client = TestClient(create_app(configured))
    response = client.get(
        "/api/providers/codex/models",
        headers={"X-NsysScope-Token": "test-token"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["models"] == [
        {"id": "live-codex", "label": "Live Codex"},
    ]


def test_codex_selected_model_is_forwarded(tmp_path: Path) -> None:
    configured = replace(settings(tmp_path), codex_enabled=True)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "job"
    source = tmp_path / "source"
    job_dir.mkdir()
    source.mkdir()
    report = tmp_path / "report.sqlite"
    config = tmp_path / "config.json"
    launch = tmp_path / "launch.yaml"
    report.write_text("")
    config.write_text("{}")
    launch.write_text("model: test\n")
    request = JobCreate(
        agent_provider="codex",
        agent_model="quality-codex",
        model_name="test",
        stage="prefill",
        hardware="test",
        report_path=str(report),
        config_path=str(config),
        launch_path=str(launch),
        source_path=str(source),
        result_path=str(job_dir),
        notes="只分析 GLM5.2 的单个非 shared Indexer 层",
    )
    captured: dict[str, object] = {}

    def capture_process(job_id, output, command, stdin=None, **kwargs):
        captured["command"] = command
        captured["stdin"] = stdin

    runner.state = lambda *args, **kwargs: None
    runner.run_process = capture_process
    runner.run_codex(
        "job-id", job_dir, request,
        {
            "report": report,
            "config": config,
            "launch": launch,
            "source": source,
            "design": None,
        },
        report,
    )
    command = captured["command"]
    assert command[command.index("--model") + 1] == "quality-codex"
    prompt = captured["stdin"]
    assert "Read that SKILL.md completely before analysis." in prompt
    assert "<user_acceptance_criteria>" in prompt
    assert "只分析 GLM5.2 的单个非 shared Indexer 层" in prompt
    assert "not the four-layer full/shared Indexer cycle" in prompt


def test_job_log_is_bounded_and_clips_large_lines(tmp_path: Path) -> None:
    configured = replace(
        settings(tmp_path),
        job_log_max_bytes=1024,
        job_log_line_max_bytes=128,
    )
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "bounded-log"
    job_dir.mkdir()
    for index in range(30):
        runner.log(job_dir, f"message-{index} " + "x" * 500)
    log_path = runner.job_log_path(job_dir)
    assert log_path.stat().st_size <= 1024
    content = log_path.read_text(errors="replace")
    assert "日志单条内容已截断" in content
    assert "日志已达到大小上限" in content


def test_run_process_emits_heartbeat(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "heartbeat"
    job_dir.mkdir()
    runner.run_process(
        "heartbeat-job", job_dir,
        ["/bin/sh", "-c", "sleep 1.2"],
        heartbeat_seconds=1,
        heartbeat_message="测试进程仍在运行",
    )
    assert "[heartbeat] 测试进程仍在运行" in runner.job_log_path(job_dir).read_text()


def test_run_process_returns_the_child_output_not_the_job_log(tmp_path: Path) -> None:
    # Crash signatures must be matched against the child's own output. The job log
    # also carries the command line -- whose path contains "zulu" -- and every other
    # step's output, which made the node-crash check fire on healthy runs.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "capture"
    job_dir.mkdir()

    output = runner.run_process(
        "capture-job", job_dir,
        ["/bin/sh", "-c", "echo ok", "/opt/zulu-cli/bin/zulu"],
    )

    assert output.strip() == "ok"
    assert "zulu" not in output
    assert "zulu" in runner.job_log_path(job_dir).read_text()


def test_cancel_terminates_agent_process_group(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "cancel"
    job_dir.mkdir()
    stopped = threading.Event()

    def run() -> None:
        try:
            runner.run_process(
                "cancel-job", job_dir,
                ["/bin/sh", "-c", "sleep 30"],
            )
        except RuntimeError:
            stopped.set()

    thread = threading.Thread(target=run)
    thread.start()
    for _ in range(100):
        if "cancel-job" in runner.processes:
            break
        time.sleep(0.01)
    assert runner.cancel("cancel-job") is True
    thread.join(timeout=3)
    assert stopped.is_set()
    assert not thread.is_alive()


def test_stall_timeout_kills_a_silent_agent(tmp_path: Path) -> None:
    # A dropped model request leaves the agent process alive with nothing pending:
    # no output, no new artifact. Waiting for comate_timeout_seconds then wastes
    # hours, so the heartbeat has to notice and fail the job instead.
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=1)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "stalled"
    job_dir.mkdir()

    failure: list[str] = []
    try:
        runner.run_process(
            "stall-job", job_dir,
            ["/bin/sh", "-c", "sleep 30"],
            heartbeat_seconds=1,
            heartbeat_message="测试进程仍在运行",
            stall_timeout_seconds=1,
        )
    except RuntimeError as error:
        failure.append(str(error))

    assert failure and "停滞" in failure[0]
    log = runner.job_log_path(job_dir).read_text()
    assert "[stalled]" in log
    assert "距上次进展" in log
    assert "stall-job" not in runner.processes


def test_new_artifacts_reset_the_stall_timer(tmp_path: Path) -> None:
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=3)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "working"
    job_dir.mkdir()

    runner.run_process(
        "working-job", job_dir,
        ["/bin/sh", "-c", "for i in 1 2 3 4 5; do touch step_$i; sleep 1; done"],
        heartbeat_seconds=1,
        heartbeat_message="测试进程仍在运行",
        stall_timeout_seconds=3,
    )
    assert (job_dir / "step_5").is_file()
    log = runner.job_log_path(job_dir).read_text()
    assert "[stalled]" not in log
    # The heartbeat has to name the artifact, otherwise the log shows no progress.
    assert "最近产出 step_" in log


def test_scratch_dumps_under_logs_count_as_progress(tmp_path: Path) -> None:
    # The agent drops its own kernel-sequence dumps into logs/ during a long
    # segmentation pass; only job.log itself must be ignored, or a working agent
    # gets killed for being quiet.
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=3)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "scratch"
    (job_dir / "logs").mkdir(parents=True)

    runner.run_process(
        "scratch-job", job_dir,
        ["/bin/sh", "-c", "for i in 1 2 3 4 5; do touch logs/dev4_seq_$i.txt; sleep 1; done"],
        heartbeat_seconds=1,
        heartbeat_message="测试进程仍在运行",
        stall_timeout_seconds=3,
    )
    log = runner.job_log_path(job_dir).read_text()
    assert "[stalled]" not in log
    assert "最近产出 logs/dev4_seq_" in log


def test_cpu_burning_agent_is_not_stalled(tmp_path: Path) -> None:
    # A long reasoning turn writes nothing and prints nothing while it keeps a core
    # busy; killing it would throw away the whole run, so CPU counts as progress.
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=3)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "thinking"
    job_dir.mkdir()

    runner.run_process(
        "thinking-job", job_dir,
        [sys.executable, "-c",
         "import time\ndeadline = time.time() + 7\nwhile time.time() < deadline: pass"],
        heartbeat_seconds=2,
        heartbeat_message="测试进程仍在运行",
        stall_timeout_seconds=3,
    )
    log = runner.job_log_path(job_dir).read_text()
    assert "[stalled]" not in log
    assert "累计 CPU" in log


def test_live_agent_session_prevents_a_kill(tmp_path: Path) -> None:
    # The direct liveness signal: the Comate engine keeps one conversation file per
    # run, keyed by the --cwd we passed, and appends to it on every message and tool
    # result. An agent whose conversation keeps advancing is working, even when it
    # writes nothing into the job dir, prints nothing and burns no measurable CPU --
    # exactly the shape of a long remote tool call that was killed as "stalled".
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=3)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "conversing"
    job_dir.mkdir()
    store = configured.comate_store_dir
    store.mkdir(parents=True, exist_ok=True)
    session = store / "chat_session_11111111-2222-3333-4444-555555555555"
    session.write_text(json.dumps({
        "sessionUuid": "11111111-2222-3333-4444-555555555555",
        "workspaceDirectory": str(job_dir),
        "messages": [],
    }), encoding="utf-8")

    runner.run_process(
        "conversing-job", job_dir,
        ["/bin/sh", "-c", f"for i in 1 2 3 4 5 6 7 8; do sleep 1; touch {session}; done"],
        heartbeat_seconds=1,
        heartbeat_message="测试进程仍在运行",
        stall_timeout_seconds=3,
        session_store=store,
    )
    log = runner.job_log_path(job_dir).read_text()
    assert "[stalled]" not in log
    assert f"已定位 Agent 会话 {session.name}" in log
    assert "会话 0 分钟前更新" in log


def test_agent_without_a_session_still_stalls(tmp_path: Path) -> None:
    # Liveness is an extra reprieve, not an excuse: a run with no conversation file
    # and no other signal is still killed, and the log says the file was missing so
    # the cause is visible.
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=1)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "sessionless"
    job_dir.mkdir()

    with pytest.raises(RuntimeError):
        runner.run_process(
            "sessionless-job", job_dir,
            ["/bin/sh", "-c", "sleep 30"],
            heartbeat_seconds=1,
            heartbeat_message="测试进程仍在运行",
            stall_timeout_seconds=1,
            session_store=configured.comate_store_dir,
        )
    log = runner.job_log_path(job_dir).read_text()
    assert "未找到 Agent 会话文件" in log
    assert "没有会话更新" in log


def test_a_foreign_conversation_is_not_taken_for_ours(tmp_path: Path) -> None:
    # Another job's conversation, and my own transcript quoting this job dir, must
    # not read as this job's liveness -- otherwise stall detection is defeated by
    # whatever else happens to be running on the host.
    configured = settings(tmp_path)
    store = configured.comate_store_dir
    store.mkdir(parents=True, exist_ok=True)
    job_dir = tmp_path / "mine"
    other = store / "chat_session_aaaaaaaa-0000-0000-0000-000000000000"
    other.write_text(json.dumps({
        "workspaceDirectory": str(tmp_path / "theirs"),
        "messages": [{"content": f"look at {job_dir} for me"}],
    }), encoding="utf-8")
    assert JobRunner.find_agent_session(store, job_dir, 0.0) is None

    mine = store / "chat_session_bbbbbbbb-0000-0000-0000-000000000000"
    mine.write_text(json.dumps({"workspaceDirectory": str(job_dir)}), encoding="utf-8")
    assert JobRunner.find_agent_session(store, job_dir, 0.0) == mine
    # A conversation from before this process started belongs to an earlier run.
    assert JobRunner.find_agent_session(store, job_dir, time.time() + 60) is None


def test_a_quiet_session_stalls_an_agent_that_still_burns_cpu(tmp_path: Path) -> None:
    # The case that hung a colleague's job for two hours: the agent had finished its
    # turn, so its conversation file stopped growing, but the CLI process stayed up
    # with an idle event loop that burnt enough CPU to reset the progress timer on
    # every heartbeat. A conversation that has gone quiet for longer than the timeout
    # now ends the wait on its own, whatever the other signals say.
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=2)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "parked"
    job_dir.mkdir()
    store = configured.comate_store_dir
    store.mkdir(parents=True, exist_ok=True)
    session = store / "chat_session_99999999-0000-0000-0000-000000000000"
    session.write_text(json.dumps({"workspaceDirectory": str(job_dir)}), encoding="utf-8")
    # The session is fresh, so it is located on the first heartbeat, and then never
    # touched again -- the shape of an agent whose turn is over. Meanwhile the process
    # keeps a core busy, which used to be enough to reset the progress timer forever.

    with pytest.raises(RuntimeError):
        runner.run_process(
            "parked-job", job_dir,
            [sys.executable, "-c",
             "import time\ndeadline = time.time() + 20\nwhile time.time() < deadline: pass"],
            heartbeat_seconds=1,
            heartbeat_message="测试进程仍在运行",
            stall_timeout_seconds=2,
            session_store=store,
        )
    log = runner.job_log_path(job_dir).read_text()
    assert "[stalled]" in log
    assert "没有会话更新" in log


def test_complete_outputs_end_the_wait_instead_of_failing(tmp_path: Path) -> None:
    # Same parked CLI, but this time everything the pipeline needs is already on disk.
    # Killing the process is then a success, not a stall: the caller carries on with
    # the package, and the job must not be reported as failed.
    configured = replace(settings(tmp_path), agent_stall_timeout_seconds=3)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "finished"
    package = job_dir / "result"
    package.mkdir(parents=True)
    for suffix in AGENT_CSV_SUFFIXES:
        (package / f"analysis{suffix}").write_text("序号\n1\n", encoding="utf-8")
    store = configured.comate_store_dir
    store.mkdir(parents=True, exist_ok=True)
    session = store / "chat_session_88888888-0000-0000-0000-000000000000"
    session.write_text(json.dumps({"workspaceDirectory": str(job_dir)}), encoding="utf-8")

    runner.run_process(
        "finished-job", job_dir,
        [sys.executable, "-c",
         "import time\ndeadline = time.time() + 20\nwhile time.time() < deadline: pass"],
        heartbeat_seconds=1,
        heartbeat_message="测试进程仍在运行",
        stall_timeout_seconds=3,
        session_store=store,
    )
    log = runner.job_log_path(job_dir).read_text()
    assert "产物已齐全" in log
    assert "[stalled]" not in log


def test_forward_pipeline_table_is_optional(tmp_path: Path) -> None:
    # The seventh table is a bonus view, not a gate: with neither the table nor a
    # trace to rebuild it from, the package still ships its six tables.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    job_dir = tmp_path / "incomplete"
    package = job_dir / "csv"
    package.mkdir(parents=True)

    runner.ensure_forward_pipeline("job", job_dir, package, "analysis", None)
    assert not (package / "analysis_forward_pipeline_table.csv").exists()

    # A package that ships the table needs no trace.
    (package / "analysis_forward_pipeline_table.csv").write_text(
        "环节,总耗时(us)\nforward step 总计,1.0\n", encoding="utf-8",
    )
    runner.ensure_forward_pipeline("job", job_dir, package, "analysis", None)


def test_packaging_survives_a_missing_seventh_table(tmp_path: Path) -> None:
    # validate_package logs the seventh table's absence and continues, so packaging
    # must too: it used to raise FileNotFoundError one step later, which read like a
    # broken pipeline rather than a missing optional view.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    result_dir = tmp_path / "result"
    package = result_dir / "tables"
    package.mkdir(parents=True)
    for suffix in AGENT_CSV_SUFFIXES:
        (package / f"analysis{suffix}").write_text("a,b\n1,2\n", encoding="utf-8")
    (result_dir / "analysis.json").write_text(json.dumps({
        "schemaVersion": "1.0",
        "summary": {"operatorCount": 1, "devices": [0], "stableSamples": 1},
        "operators": [{"category": "core"}],
    }) + "\n", encoding="utf-8")
    trace = result_dir / "capture.sqlite"
    trace.write_bytes(b"")

    runner.organize_result_package(result_dir, package, "analysis", trace)

    manifest = json.loads((result_dir / "nsysscope-package.json").read_text())
    assert len(manifest["tables"]) == len(AGENT_CSV_SUFFIXES)
    assert not any("forward_pipeline" in name for name in manifest["tables"])
    for name in manifest["tables"]:
        assert (result_dir / "csv" / name).is_file()


def flat_analysis_dir(root: Path, prefix: str = "analysis") -> Path:
    """A finished analysis as it is easiest to author it: one flat directory."""
    root.mkdir(parents=True)
    for suffix in AGENT_CSV_SUFFIXES:
        (root / f"{prefix}{suffix}").write_text("a,b\n1,2\n", encoding="utf-8")
    (root / f"{prefix}_analysis_manifest.json").write_text("{}\n", encoding="utf-8")
    (root / "validation_report.json").write_text("{}\n", encoding="utf-8")
    # Contract-valid on purpose: finalize_package refuses to write a manifest for an
    # analysis.json the frontend could not render.
    (root / "analysis.json").write_text(json.dumps({
        "schemaVersion": "1.0",
        "summary": {"operatorCount": 1, "devices": [0], "stableSamples": 1},
        "operators": [{"category": "core"}],
    }) + "\n", encoding="utf-8")
    (root / "final_report.md").write_text("# report\n", encoding="utf-8")
    (root / "scratch_probe.csv").write_text("x\n1\n", encoding="utf-8")
    (root / "capture.sqlite").write_bytes(b"")
    return root


def test_the_skill_and_the_service_lay_a_package_out_the_same_way(
    tmp_path: Path,
) -> None:
    # The whole point of the Skill owning the packager: a run done by hand and a job
    # run through the service must end in the same directory, so nobody has to know
    # which one produced a package they were handed.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))

    by_hand = flat_analysis_dir(tmp_path / "by-hand")
    subprocess.run(
        [
            sys.executable, str(SKILL / "scripts" / "finalize_package.py"),
            str(by_hand), "--prefix", "analysis",
            "--trace", str(by_hand / "capture.sqlite"),
        ],
        check=True, capture_output=True, text=True,
    )

    by_tool = flat_analysis_dir(tmp_path / "by-tool")
    runner.organize_result_package(
        by_tool, by_tool, "analysis", by_tool / "capture.sqlite",
    )

    def tree(root: Path) -> set[str]:
        return {
            path.relative_to(root).as_posix()
            for path in root.rglob("*") if path.is_file()
        }

    assert tree(by_hand) == tree(by_tool)
    assert "csv/analysis_stage_table.csv" in tree(by_hand)
    assert "xlsx/analysis_stage_table.xlsx" in tree(by_hand)
    assert "metadata/scratch_probe.csv" in tree(by_hand)
    assert "metadata/validation_report.json" in tree(by_hand)
    assert "trace/capture.sqlite" in tree(by_hand)
    # The frontend contract, the report and the manifest stay at the root.
    assert {"analysis.json", "final_report.md", "nsysscope-package.json"} <= tree(
        by_hand
    )


def test_packaging_repairs_a_derivable_contract_violation(tmp_path: Path) -> None:
    # analysis.json is derived from the tables, so a violation in it is repairable
    # without judgement. Failing a finished analysis over a derived file would waste
    # the whole run, so packaging rebuilds it and only fails if that does not help.
    require_package()
    result_dir = package_copy(tmp_path)
    (result_dir / "analysis.json").write_text(json.dumps({
        "schemaVersion": "1.0",
        # A partial write: most operators never made it, so the count disagrees and
        # the device scope is gone.
        "summary": {"operatorCount": 99, "devices": [], "stableSamples": 1},
        "operators": [{"category": "core"}],
        "metadata": {"model": "GLM5.2", "stage": "decode", "hardware": "Nvidia B200"},
    }) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable, str(SKILL / "scripts" / "finalize_package.py"),
            str(result_dir), "--prefix", package_prefix(),
        ],
        capture_output=True, text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "rebuilding it from" in completed.stderr
    assert (result_dir / "nsysscope-package.json").is_file()
    # The rejected document is kept as evidence rather than dropped.
    assert (result_dir / "metadata" / "analysis.rejected.json").is_file()
    repaired = json.loads((result_dir / "analysis.json").read_text())
    assert repaired["summary"]["operatorCount"] == len(repaired["operators"])
    assert repaired["metadata"]["model"] == "GLM5.2"


def test_packaging_refuses_what_it_cannot_repair(tmp_path: Path) -> None:
    # The one case that must still fail: the tables themselves lack what the
    # frontend needs, so rebuilding cannot conjure it. Silently shipping a cycle
    # collapsed into one averaged layer would mislead every reader of the dashboard.
    result_dir = flat_analysis_dir(tmp_path / "collapsed")
    (result_dir / "analysis.json").write_text(json.dumps({
        "schemaVersion": "1.0",
        "summary": {
            "operatorCount": 1,
            "devices": [0],
            "stableSamples": 1,
            "heterogeneous": True,
            "distinctUnitVariants": ["KDA", "MLA"],
            "durationLabel": "平均单层耗时",
        },
        "operators": [{
            "category": "core", "unitPosition": 1, "unitId": "layer.4",
            "unitVariant": "KDA",
        }],
    }) + "\n", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable, str(SKILL / "scripts" / "finalize_package.py"),
            str(result_dir), "--prefix", "analysis",
        ],
        capture_output=True, text=True,
    )

    assert completed.returncode != 0
    assert "frontend contract" in completed.stderr
    assert "after rebuilding it from the tables" in completed.stderr
    assert not (result_dir / "nsysscope-package.json").exists()


def test_a_renamed_prefix_is_followed_instead_of_reported_as_no_tables(
    tmp_path: Path,
) -> None:
    # An agent that names its tables after the model it found in the trace has done
    # the work; failing with "did not produce the six agent tables" hid that.
    package = tmp_path / "csv"
    package.mkdir()
    for suffix in AGENT_CSV_SUFFIXES:
        (package / f"glm5_decode{suffix}").write_text("a,b\n1,2\n", encoding="utf-8")

    assert JobRunner.find_package(tmp_path, "requested") == package
    assert JobRunner.detect_prefix(package, "requested") == "glm5_decode"


def test_bad_optional_torch_trace_does_not_fail_the_job(tmp_path: Path) -> None:
    # The torch trace only makes the analysis faster, so a path that no longer
    # resolves degrades to "not supplied" instead of costing the caller a run.
    # An evidence input in the same state still fails loudly.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    report = tmp_path / "capture.sqlite"
    report.write_bytes(b"")
    config = tmp_path / "config.json"
    config.write_text("{}", encoding="utf-8")
    launch = tmp_path / "launch.sh"
    launch.write_text("#!/bin/sh\n", encoding="utf-8")
    source = tmp_path / "source"
    source.mkdir()
    job_dir = tmp_path / "trace-job"
    job_dir.mkdir()

    def build(**overrides: str) -> JobCreate:
        fields = {
            "model_name": "test",
            "stage": "prefill",
            "hardware": "test",
            "report_path": str(report),
            "config_path": str(config),
            "launch_path": str(launch),
            "source_path": str(source),
            "result_path": str(tmp_path / "out"),
        }
        fields.update(overrides)
        return JobCreate(**fields)

    paths = runner.resolve_inputs(
        build(torch_trace_path=str(tmp_path / "missing-trace.json")), job_dir,
    )
    assert paths["report"] == report
    assert paths["torch_trace"] is None
    assert "missing-trace.json" in JobRunner.job_log_path(job_dir).read_text(
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        runner.resolve_inputs(
            build(source_path=str(tmp_path / "missing-source")), job_dir,
        )


def test_plugin_serves_the_bundled_skills() -> None:
    # The Codex plugin used to keep its own copy of the skills, which drifted 13
    # files behind bundled/ and never gained functional-modules.json at all --
    # two installs of one tool behaving differently is worse than a failure.
    bundled = PROJECT / "bundled" / "skills"
    exposed = PROJECT / "plugins" / "nsysscope" / "skills"
    expected = {path.name for path in bundled.iterdir() if path.is_dir()}
    assert {path.name for path in exposed.iterdir()} == expected
    for name in expected:
        assert (exposed / name).resolve() == (bundled / name).resolve(), (
            f"plugin skill {name} is a separate copy, not the bundled skill"
        )


def test_job_log_names_the_skill_and_warns_about_a_shadowing_copy(
    tmp_path: Path,
) -> None:
    # skill_manager resolves ~/.codex/skills before bundled/, so an old copy runs
    # instead of the repository's skill with nothing in the log to show it.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))

    bundled_dir = tmp_path / "bundled"
    bundled_dir.mkdir()
    runner.log_skill_source(bundled_dir, {
        "name": "sglang-nsys-static-analysis",
        "source": "bundled",
        "path": str(PROJECT / "bundled" / "skills" / "sglang-nsys-static-analysis"),
        "sha256": "a" * 64,
    })
    bundled_log = JobRunner.job_log_path(bundled_dir).read_text(encoding="utf-8")
    assert "sglang-nsys-static-analysis (bundled)" in bundled_log
    assert "注意" not in bundled_log

    shadowed_dir = tmp_path / "shadowed"
    shadowed_dir.mkdir()
    runner.log_skill_source(shadowed_dir, {
        "name": "sglang-nsys-static-analysis",
        "source": "codex",
        "path": "/home/someone/.codex/skills/sglang-nsys-static-analysis",
        "sha256": "b" * 64,
    })
    shadowed_log = JobRunner.job_log_path(shadowed_dir).read_text(encoding="utf-8")
    assert "(codex)" in shadowed_log
    assert "不是仓库自带的 skill" in shadowed_log


def test_report_version_decides_which_nsys_can_export(tmp_path: Path) -> None:
    # A .nsys-rep opens with the build that wrote it, and `nsys export` only reads
    # its own version or older -- so this header is what decides whether the local
    # nsys is usable at all.
    report = tmp_path / "capture.nsys-rep"
    report.write_bytes(
        b"NVIDIA Tegra Profiler Report 2026@4@1@191-264138605071v0."
        b"\n\x0bLocal (CLI)\x12\n"
    )
    assert JobRunner.report_tool_version(report) == (
        (2026, 4, 1, 191), "2026.4.1.191-264138605071v0",
    )

    not_a_report = tmp_path / "plain.nsys-rep"
    not_a_report.write_bytes(b"whatever")
    assert JobRunner.report_tool_version(not_a_report) is None


def test_nsys_package_is_picked_by_build_then_by_next_newer(tmp_path: Path) -> None:
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    index = textwrap.dedent("""\
        Package: nsight-systems-cli-2025.3.1
        Version: 2025.3.1.90-253135822126v0
        Filename: ./NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb

        Package: nsight-systems-cli-2026.4.1
        Version: 2026.4.1.191-264138605071v0
        Filename: ./NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb

        Package: nsight-systems-2026.4.1
        Version: 2026.4.1.191-264138605071v0
        Filename: ./nsight-systems-2026.4.1_2026.4.1.191-1_amd64.deb
    """)
    runner.download_text = lambda url: index  # type: ignore[method-assign]

    # Exact build wins, and never the multi-GB full package that shares its version.
    assert runner.nsys_package_filename(
        (2026, 4, 1, 191), "2026.4.1.191-264138605071v0",
    ) == "NsightSystems-linux-cli-public-2026.4.1.191-3860507.deb"
    # Unknown build: the lowest CLI that is still new enough to read the report.
    assert runner.nsys_package_filename(
        (2025, 3, 1, 90), "2025.3.1.90-unlisted",
    ) == "NsightSystems-linux-cli-public-2025.3.1.90-3582212.deb"
    with pytest.raises(RuntimeError):
        runner.nsys_package_filename((2099, 1, 1, 1), "2099.1.1.1-x")


def test_download_switches_proxy_and_resumes_from_what_it_has(tmp_path: Path) -> None:
    # A proxy can answer a HEAD instantly and then deliver nothing on a 200 MB body,
    # so reachability is not throughput: the transfer itself has to pick the path, and
    # a failed chunk must resume rather than restart.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    body = bytes(range(256)) * 40  # 10240 bytes
    attempted: list[tuple[object, int]] = []

    class Response:
        def __init__(self, payload: bytes, start: int) -> None:
            self.payload = payload
            self.status = 206
            self.headers = {
                "Content-Range": f"bytes {start}-{start + len(payload) - 1}/{len(body)}",
            }

        def read(self):
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(proxy):
        class Opener:
            def open(self, request, timeout=None):
                start = int(request.headers["Range"].split("=")[1].split("-")[0])
                attempted.append((proxy, start))
                # The first proxy stalls once, halfway through the file.
                if proxy == "http://first:1" and start == 4096:
                    raise TimeoutError("read timed out")
                return Response(body[start:start + 4096], start)
        return Opener()

    runner._opener = opener  # type: ignore[method-assign]
    artifact = tmp_path / "artifact.bin"
    # The probe is a throughput measurement, not part of the transfer under test.
    runner.fastest_download_path = lambda url, **kwargs: None  # type: ignore[method-assign]
    with mock.patch.object(
        JobRunner, "download_candidates",
        lambda self: ["http://first:1", "http://second:1"],
    ):
        runner.download("https://example.invalid/artifact.bin", artifact, chunk_bytes=4096)

    assert artifact.read_bytes() == body
    # The stalled chunk resumed at the same offset on the next proxy, not from zero.
    assert ("http://first:1", 4096) in attempted
    assert ("http://second:1", 4096) in attempted
    assert all(start % 4096 == 0 for _, start in attempted)

    with mock.patch.object(
        JobRunner, "download_candidates", lambda self: ["http://dead:1"],
    ):
        runner._opener = lambda proxy: mock.Mock(  # type: ignore[method-assign]
            open=mock.Mock(side_effect=TimeoutError("read timed out")),
        )
        with pytest.raises(RuntimeError, match="下载失败"):
            runner.download("https://example.invalid/artifact.bin", artifact)


def test_download_path_is_chosen_by_measured_throughput(tmp_path: Path) -> None:
    # Candidate order picks badly: the first *working* office proxy measured
    # 0.24 MB/s while the second did 0.95 MB/s, which turned a 4-minute download
    # into 13. Speed decides, not position.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))

    class Response:
        def __init__(self, payload: bytes, delay: float) -> None:
            self.payload = payload
            self.delay = delay

        def read(self):
            time.sleep(self.delay)
            return self.payload

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def opener(proxy):
        class Opener:
            def open(self, request, timeout=None):
                if proxy is None:
                    raise TimeoutError("read timed out")
                # The slow path answers first in candidate order.
                return Response(b"x" * 4096, 0.20 if proxy == "http://slow:1" else 0.02)
        return Opener()

    runner._opener = opener  # type: ignore[method-assign]
    with mock.patch.object(
        JobRunner, "download_candidates",
        lambda self: [None, "http://slow:1", "http://fast:1"],
    ):
        chosen = runner.fastest_download_path(
            "https://example.invalid/artifact.bin", probe_bytes=4096,
        )
    assert chosen == "http://fast:1"
    assert runner._download_proxy == "http://fast:1"


def test_nsys_fetch_reports_a_bar_with_rate_and_eta(tmp_path: Path) -> None:
    # A ten-minute download with no output looks like a hung job, so the fetch
    # reports progress into both the job status and the log.
    configured = settings(tmp_path)
    configured.prepare()
    runner = JobRunner(configured, JobStore(configured.data_dir / "jobs.sqlite"))
    # The reporter checks the job's status so a cancel that lands mid-download is not
    # overwritten; this test only cares about the rendered progress line.
    runner.is_cancelled = lambda job_id: False  # type: ignore[method-assign]
    reported: list[tuple[int, str]] = []
    runner.state = (  # type: ignore[method-assign]
        lambda job_id, job_dir, status, progress, message: reported.append(
            (progress, message),
        )
    )

    report = runner.nsys_progress_reporter("job", tmp_path, "2026.4.1.191")
    report(64 * 1048576, 200 * 1048576, 0.35 * 1048576)
    report(200 * 1048576, 200 * 1048576, 1.0 * 1048576)

    first, last = reported[0], reported[-1]
    assert "2026.4.1.191" in first[1] and "32%" in first[1]
    assert "64/200MB" in first[1] and "0.35MB/s" in first[1]
    assert "剩余约 6 分钟" in first[1]
    # Progress stays inside the slice reserved for the fetch, below the export.
    assert 10 <= first[0] < last[0] <= 16
    assert "100%" in last[1] and "剩余约 0 秒" in last[1]
