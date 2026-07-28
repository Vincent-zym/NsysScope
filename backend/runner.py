from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .config import Settings
from .models import JobCreate
from .store import JobStore


CSV_SUFFIXES = (
    "_operator_origin_table.csv",
    "_opreator_table.csv",
    "_core_compute_table.csv",
    "_auxiliary_operator_table.csv",
    "_op_classification_table.csv",
    "_stage_table.csv",
)


class JobRunner:
    def __init__(self, settings: Settings, store: JobStore):
        self.settings = settings
        self.store = store
        self.pool = ThreadPoolExecutor(
            max_workers=settings.max_workers, thread_name_prefix="nsysscope",
        )
        self.processes: dict[str, subprocess.Popen[str]] = {}
        self.lock = threading.Lock()

    def submit(self, job_id: str) -> None:
        self.pool.submit(self.run, job_id)

    def log(self, job_dir: Path, message: str) -> None:
        with (job_dir / "job.log").open("a", encoding="utf-8") as handle:
            handle.write(message.rstrip() + "\n")

    def state(self, job_id: str, job_dir: Path, status: str, progress: int, message: str) -> None:
        self.store.update(job_id, status=status, progress=progress, message=message)
        self.log(job_dir, f"[{progress:03d}%] {message}")

    def run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        request = JobCreate.model_validate(job["request"])
        job_dir = Path(job["output_dir"])
        try:
            if request.mode == "existing_package":
                package = self.settings.resolve_allowed(
                    request.existing_package_path or "", kind="existing package",
                )
                self.state(job_id, job_dir, "converting", 70, "正在转换已有六表分析包")
                prefix = self.detect_prefix(package, request.prefix)
                self.convert(package, job_dir / "analysis.json", prefix)
            else:
                paths = self.resolve_inputs(request)
                context = {key: str(value) if value else None for key, value in paths.items()}
                context.update(request.model_dump())
                (job_dir / "context.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                )
                sqlite_path = self.export_nsys(job_id, job_dir, paths["report"])
                context["sqlite_path"] = str(sqlite_path)
                (job_dir / "context.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                )
                self.run_codex(job_id, job_dir, request, paths, sqlite_path)
                self.state(job_id, job_dir, "converting", 88, "正在构建前端 analysis.json")
                package = self.find_package(job_dir, request.prefix)
                self.convert(package, job_dir / "analysis.json", request.prefix)

            self.state(job_id, job_dir, "validating", 95, "正在校验前端数据契约")
            payload = json.loads((job_dir / "analysis.json").read_text())
            if payload.get("schemaVersion") != "1.0" or not payload.get("operators"):
                raise RuntimeError("analysis.json schema or operators are invalid")
            self.state(job_id, job_dir, "succeeded", 100, "分析完成")
        except Exception as exc:
            self.log(job_dir, traceback.format_exc())
            if self.store.get(job_id)["status"] != "cancelled":
                self.store.update(
                    job_id, status="failed", progress=100,
                    message="分析失败", error=str(exc),
                )

    def resolve_inputs(self, request: JobCreate) -> dict[str, Path | None]:
        values = {
            "report": request.report_path,
            "config": request.config_path,
            "launch": request.launch_path,
            "source": request.source_path,
            "design": request.design_path,
        }
        return {
            key: self.settings.resolve_allowed(value, kind=key) if value else None
            for key, value in values.items()
        }

    def export_nsys(self, job_id: str, job_dir: Path, report: Path | None) -> Path:
        if report is None:
            raise RuntimeError("report path is missing")
        if report.suffix == ".sqlite":
            self.state(job_id, job_dir, "exporting", 20, "输入已经是 SQLite，跳过导出")
            return report
        if report.suffix != ".nsys-rep":
            raise RuntimeError("report must end with .nsys-rep or .sqlite")
        sqlite_path = job_dir / f"{report.stem}.sqlite"
        self.state(job_id, job_dir, "exporting", 10, "正在导出 Nsight Systems SQLite")
        self.run_process(
            job_id, job_dir,
            [self.settings.nsys_bin, "export", "--type", "sqlite", "--output", str(sqlite_path), str(report)],
        )
        if not sqlite_path.exists():
            raise RuntimeError("nsys export completed without producing SQLite")
        return sqlite_path

    def run_codex(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
    ) -> None:
        if not self.settings.codex_enabled:
            raise RuntimeError(
                "Codex analyzer is disabled. Set NSYSSCOPE_CODEX_ENABLED=true on an isolated runner."
            )
        self.state(job_id, job_dir, "analyzing", 30, "Codex Skill Agent 正在分析模型与时间线")
        prompt = self.build_prompt(job_dir, request, paths, sqlite_path)
        prompt_path = job_dir / "prompt.md"
        prompt_path.write_text(prompt)
        output_message = job_dir / "agent-final.txt"
        command = [
            self.settings.codex_bin, "--ask-for-approval", "never", "exec",
            "--ephemeral", "--json", "--sandbox", "workspace-write",
            "--skip-git-repo-check", "-C", str(job_dir),
        ]
        for directory in sorted({
            str(path if path.is_dir() else path.parent)
            for path in paths.values() if path is not None
        }):
            command.extend(["--add-dir", directory])
        command.extend(["--output-last-message", str(output_message), "-"])
        self.run_process(job_id, job_dir, command, stdin=prompt)

    def build_prompt(
        self, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
    ) -> str:
        return f"""Use the installed `sglang-nsys-static-analysis` skill.

Analyze this task without asking follow-up questions:
- nsys/sqlite: {sqlite_path}
- original report: {paths['report']}
- model: {request.model_name}
- stage: {request.stage}
- hardware: {request.hardware}
- config: {paths['config']}
- deployment YAML/script: {paths['launch']}
- model source root: {paths['source']}
- design notes: {paths['design'] or 'not supplied'}
- user notes: {request.notes or 'none'}

Write all artifacts only under:
{job_dir}

Requirements:
1. Read model evidence first and derive a task-specific functional taxonomy.
2. Select and prove one complete repeating unit.
3. Generate the normalized six CSV tables with prefix `{request.prefix}`.
4. Write `{request.prefix}_analysis_manifest.json`, semantic map, stable-statistics sidecar,
   and `validation_report.json`.
5. Validate every required invariant and finish only when validation passes.
6. Never edit input reports, config, launch files, design notes, or model source.
"""

    def run_process(
        self, job_id: str, job_dir: Path, command: list[str], stdin: str | None = None,
    ) -> None:
        redacted = " ".join(command)
        self.log(job_dir, f"$ {redacted}")
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE if stdin is not None else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=job_dir,
            env={**os.environ, "NO_COLOR": "1"},
        )
        with self.lock:
            self.processes[job_id] = process
        try:
            assert process.stdout is not None
            if stdin is not None and process.stdin is not None:
                process.stdin.write(stdin)
                process.stdin.close()
            for line in process.stdout:
                if line.strip():
                    self.log(job_dir, line)
            code = process.wait()
            if code:
                raise RuntimeError(f"process exited with code {code}: {command[0]}")
        finally:
            with self.lock:
                self.processes.pop(job_id, None)

    def convert(self, package: Path, output: Path, prefix: str) -> None:
        command = [
            "python3", str(self.settings.converter), str(package), str(output),
            "--prefix", prefix,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "analysis conversion failed")

    @staticmethod
    def find_package(job_dir: Path, prefix: str) -> Path:
        candidates = [job_dir, *[path.parent for path in job_dir.rglob(f"{prefix}_stage_table.csv")]]
        for path in candidates:
            if all((path / f"{prefix}{suffix}").exists() for suffix in CSV_SUFFIXES):
                return path
        raise RuntimeError("Codex run did not produce a complete six-table package")

    @staticmethod
    def detect_prefix(package: Path, requested: str) -> str:
        if (package / f"{requested}_stage_table.csv").exists():
            return requested
        candidates = [
            path.name.removesuffix("_stage_table.csv")
            for path in package.glob("*_stage_table.csv")
        ]
        complete = [
            prefix for prefix in candidates
            if all((package / f"{prefix}{suffix}").exists() for suffix in CSV_SUFFIXES)
        ]
        if len(complete) == 1:
            return complete[0]
        raise RuntimeError(
            f"cannot resolve table prefix; requested={requested!r}, candidates={complete}"
        )

    def cancel(self, job_id: str) -> bool:
        with self.lock:
            process = self.processes.get(job_id)
        if process and process.poll() is None:
            process.terminate()
            return True
        return False
