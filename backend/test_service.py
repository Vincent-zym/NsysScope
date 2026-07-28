from __future__ import annotations

import json
import shutil
import textwrap
import time
from dataclasses import replace
from pathlib import Path

from fastapi.testclient import TestClient

from backend.app import create_app
from backend.config import Settings
from backend.models import JobCreate
from backend.runner import JobRunner
from backend.store import JobStore
from scripts.build_analysis_json import included_devices, stable_sample_count


PROJECT = Path(__file__).resolve().parents[1]
PACKAGE = Path(
    "/home/users/zhongyuanming/nsys_statis_analysis/"
    "glm52_prefill_nonshared_indexer_20260728_run01"
)


def settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        allowed_roots=(Path("/home/users/zhongyuanming"), tmp_path),
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
        nsys_bin="nsys",
        skill_dir=Path("/root/.codex/skills/sglang-nsys-static-analysis"),
        converter=PROJECT / "scripts/build_analysis_json.py",
    )


def test_existing_package_job(tmp_path: Path) -> None:
    if not PACKAGE.exists():
        return
    client = TestClient(create_app(settings(tmp_path)))
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(PACKAGE),
        "prefix": "glm52",
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
    assert len(payload["operators"]) == 61
    assert payload["summary"]["stableSamples"] == 1164


def test_auth_and_path_boundary(tmp_path: Path) -> None:
    client = TestClient(create_app(settings(tmp_path)))
    assert client.get("/api/health").status_code == 200
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


def test_converter_accepts_current_stats_schema() -> None:
    stats = {
        "accepted_full_template_sample_count": 1188,
        "per_device_sample_counts": {"0": 147, "1": 156},
    }
    assert stable_sample_count(stats, {}) == 1188
    assert included_devices(stats, {}) == [0, 1]


def test_log_pagination_and_conversion_retry(tmp_path: Path) -> None:
    if not PACKAGE.exists():
        return
    application = create_app(settings(tmp_path))
    client = TestClient(application)
    headers = {"X-NsysScope-Token": "test-token"}
    response = client.post("/api/jobs", headers=headers, json={
        "mode": "existing_package",
        "model_name": "GLM5.2",
        "stage": "prefill",
        "hardware": "Nvidia B200",
        "existing_package_path": str(PACKAGE),
        "prefix": "glm52",
    })
    job = response.json()
    for _ in range(100):
        job = client.get(f"/api/jobs/{job['id']}", headers=headers).json()
        if job["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert job["status"] == "succeeded", job

    log_path = Path(job["output_dir"]) / "job.log"
    with log_path.open("a") as handle:
        handle.writelines(f"line-{index}\n" for index in range(7))
    page = client.get(
        f"/api/jobs/{job['id']}/logs?after=0&limit=2", headers=headers,
    ).json()
    assert page["next"] == 2
    assert page["has_more"] is True
    assert len(page["lines"]) == 2

    Path(job["output_dir"], "analysis.json").unlink()
    for source in [
        *PACKAGE.glob("glm52_*"),
        PACKAGE / "position_operator_stats.json",
        PACKAGE / "validation_report.json",
    ]:
        if source.is_file():
            shutil.copy2(source, Path(job["output_dir"]) / source.name)
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
    if not PACKAGE.exists():
        return
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
        package = Path({str(PACKAGE)!r})
        sources = list(package.glob("glm52_*")) + [
            package / "position_operator_stats.json",
            package / "validation_report.json",
        ]
        for source in sources:
            if source.is_file():
                shutil.copy2(source, cwd / source.name)
        print('{{"type":"assistant","message":"fake Comate completed"}}')
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
        "prefix": "glm52",
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
    staged = output / ".comate/skills/sglang-nsys-static-analysis/SKILL.md"
    assert staged.exists()
    assert "Use when the analysis agent receives" in staged.read_text()
    assert "<prompt>" in (output / "job.log").read_text()
    zulu_args = json.loads((output / "zulu-args.json").read_text())
    assert zulu_args[zulu_args.index("--model") + 1] == "quality-model-id"
    analysis = client.get(job["analysis_url"], headers=headers).json()
    assert analysis["schemaVersion"] == "1.0"
    assert len(analysis["operators"]) == 61


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
    assert response.json() == {
        "provider": "codex",
        "default_model": "quality-codex",
        "models": [
            {"id": "quality-codex", "label": "Quality Codex"},
            {"id": "fast-codex", "label": "Fast Codex"},
        ],
    }


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
