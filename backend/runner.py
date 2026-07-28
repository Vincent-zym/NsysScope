from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import threading
import tomllib
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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

    def submit_conversion_retry(self, job_id: str) -> None:
        self.pool.submit(self.retry_conversion, job_id)

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
                self.run_agent(job_id, job_dir, request, paths, sqlite_path)
                self.state(job_id, job_dir, "converting", 88, "正在构建前端 analysis.json")
                package = self.find_package(job_dir, request.prefix)
                self.convert(package, job_dir / "analysis.json", request.prefix)

            self.state(job_id, job_dir, "validating", 95, "正在校验前端数据契约")
            self.validate_analysis(job_dir / "analysis.json")
            self.state(job_id, job_dir, "succeeded", 100, "分析完成")
        except Exception as exc:
            self.log(job_dir, traceback.format_exc())
            if self.store.get(job_id)["status"] != "cancelled":
                self.store.update(
                    job_id, status="failed", progress=100,
                    message="分析失败", error=str(exc),
                )

    def retry_conversion(self, job_id: str) -> None:
        job = self.store.get(job_id)
        request = JobCreate.model_validate(job["request"])
        job_dir = Path(job["output_dir"])
        try:
            self.state(job_id, job_dir, "converting", 88, "正在重试构建前端 analysis.json")
            package = self.find_package(job_dir, request.prefix)
            prefix = self.detect_prefix(package, request.prefix)
            self.convert(package, job_dir / "analysis.json", prefix)
            self.state(job_id, job_dir, "validating", 95, "正在校验前端数据契约")
            self.validate_analysis(job_dir / "analysis.json")
            self.state(job_id, job_dir, "succeeded", 100, "分析完成")
        except Exception as exc:
            self.log(job_dir, traceback.format_exc())
            self.store.update(
                job_id, status="failed", progress=100,
                message="转换重试失败", error=str(exc),
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

    def run_agent(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
    ) -> None:
        if request.agent_provider == "comate":
            self.run_comate(job_id, job_dir, request, paths, sqlite_path)
        else:
            self.run_codex(job_id, job_dir, request, paths, sqlite_path)

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
            self.settings.codex_bin, "--ask-for-approval", "never",
        ]
        if request.agent_model:
            command.extend(["--model", request.agent_model])
        command.extend([
            "exec",
            "--ephemeral", "--json", "--sandbox", "workspace-write",
            "--skip-git-repo-check", "-C", str(job_dir),
        ])
        for directory in sorted({
            str(path if path.is_dir() else path.parent)
            for path in paths.values() if path is not None
        }):
            command.extend(["--add-dir", directory])
        command.extend(["--output-last-message", str(output_message), "-"])
        self.run_process(job_id, job_dir, command, stdin=prompt)

    def run_comate(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
    ) -> None:
        status = self._comate_status()
        if not status["ready"]:
            raise RuntimeError(status["message"])
        self.state(job_id, job_dir, "analyzing", 30, "Comate Skill Agent 正在分析模型与时间线")
        prompt = self.build_prompt(job_dir, request, paths, sqlite_path)
        (job_dir / "prompt.md").write_text(prompt)
        self.stage_comate_skill(job_dir)
        command = [
            self.settings.comate_bin, "run",
            "--query", prompt,
            "--cwd", str(job_dir),
            "--mode", "Agent",
            "--activate-skill", "sglang-nsys-static-analysis",
            "--display", "event-stream",
            "--background-timeout", str(self.settings.comate_timeout_seconds),
            "--disable-hooks",
        ]
        if self.settings.comate_username:
            command.extend(["--username", self.settings.comate_username])
        selected_model = request.agent_model or self.settings.comate_model
        if selected_model:
            command.extend(["--model", selected_model])
        self.run_process(
            job_id, job_dir, command, redacted_values={prompt},
            environment=self.comate_environment(),
        )

    def stage_comate_skill(self, job_dir: Path) -> Path:
        source = self.settings.skill_dir
        if not (source / "SKILL.md").exists():
            raise RuntimeError(f"analysis skill is missing: {source / 'SKILL.md'}")
        target = job_dir / ".comate" / "skills" / "sglang-nsys-static-analysis"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            source,
            target,
            ignore=shutil.ignore_patterns(
                "result", "evals", "agents", "__pycache__", "*.pyc",
            ),
        )
        skill_md = target / "SKILL.md"
        skill_md.write_text(
            skill_md.read_text().replace(
                "Use when Codex receives", "Use when the analysis agent receives",
            ),
        )
        return target

    def build_prompt(
        self, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
    ) -> str:
        return f"""Use the activated `sglang-nsys-static-analysis` skill.

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
        redacted_values: set[str] | None = None,
        environment: dict[str, str] | None = None,
    ) -> None:
        hidden = redacted_values or set()
        displayed = ["<prompt>" if item in hidden else item for item in command]
        self.log(job_dir, f"$ {shlex.join(displayed)}")
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE if stdin is not None else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=job_dir,
            env=environment or {**os.environ, "NO_COLOR": "1"},
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

    def provider_status(self) -> dict[str, dict[str, object]]:
        return {
            "codex": self._codex_status(),
            "comate": self._comate_status(),
        }

    def provider_models(self, provider: str) -> dict[str, object]:
        if provider == "codex":
            return self._codex_models()
        if provider == "comate":
            return self._comate_models()
        raise RuntimeError(f"unknown Agent Provider: {provider}")

    def _codex_models(self) -> dict[str, object]:
        codex_home = Path(os.getenv("CODEX_HOME", Path.home() / ".codex"))
        configured_model = ""
        config_path = codex_home / "config.toml"
        try:
            config = tomllib.loads(config_path.read_text())
            configured_model = str(config.get("model") or "")
        except (OSError, tomllib.TOMLDecodeError):
            pass

        choices: list[dict[str, str]] = []
        cache_path = codex_home / "models_cache.json"
        try:
            payload = json.loads(cache_path.read_text())
            records = payload.get("models", []) if isinstance(payload, dict) else payload
            for record in records:
                if record.get("visibility") == "hide":
                    continue
                model_id = str(record.get("slug") or "")
                if not model_id:
                    continue
                choices.append({
                    "id": model_id,
                    "label": str(record.get("display_name") or model_id),
                })
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
        if configured_model and not any(item["id"] == configured_model for item in choices):
            choices.insert(0, {"id": configured_model, "label": configured_model})
        return {
            "provider": "codex",
            "default_model": configured_model,
            "models": choices,
        }

    def _comate_models(self) -> dict[str, object]:
        completed = subprocess.run(
            [self.settings.comate_bin, "model", "list", "--ids"],
            text=True, capture_output=True, timeout=15,
            env=self.comate_environment(),
        )
        if completed.returncode:
            detail = (completed.stdout + completed.stderr).strip()
            raise RuntimeError(detail or "无法读取 Comate 模型列表")
        choices: list[dict[str, str]] = []
        ansi = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
        for raw_line in completed.stdout.splitlines():
            line = ansi.sub("", raw_line).strip()
            match = re.match(r"^[* ]*(.*?)\s+\(([^()]*)\)\s*$", line)
            if not match:
                continue
            label, model_id = match.groups()
            if model_id == "auto":
                continue
            choices.append({"id": model_id, "label": label.strip() or model_id})
        return {
            "provider": "comate",
            "default_model": self.settings.comate_model or "auto",
            "models": choices,
        }

    def comate_environment(self) -> dict[str, str]:
        environment = {
            **os.environ,
            "NO_COLOR": "1",
            "PLATFORM": self.settings.comate_platform,
        }
        if self.settings.comate_platform == "internal":
            for key in ("HTTPS_PROXY", "HTTP_PROXY", "https_proxy", "http_proxy", "ALL_PROXY", "all_proxy"):
                environment.pop(key, None)
        return environment

    def _codex_status(self) -> dict[str, object]:
        if not self.settings.codex_enabled:
            return {"enabled": False, "ready": False, "message": "Codex Provider 未启用"}
        executable = shutil.which(self.settings.codex_bin)
        if not executable:
            return {"enabled": True, "ready": False, "message": "找不到 Codex CLI"}
        try:
            completed = subprocess.run(
                [executable, "login", "status"],
                text=True, capture_output=True, timeout=5,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"enabled": True, "ready": False, "message": f"Codex 状态检查失败：{exc}"}
        output = (completed.stdout + completed.stderr).strip()
        return {
            "enabled": True,
            "ready": completed.returncode == 0,
            "message": output.splitlines()[-1] if output else "Codex CLI 已就绪",
        }

    def _comate_status(self) -> dict[str, object]:
        if not self.settings.comate_enabled:
            return {"enabled": False, "ready": False, "message": "Comate Provider 未启用"}
        executable = self.settings.comate_bin
        if (
            not executable
            or not Path(executable).is_file()
            or not os.access(executable, os.X_OK)
        ):
            return {"enabled": True, "ready": False, "message": "找不到 Comate Zulu CLI"}
        command = [executable, "status"]
        if self.settings.comate_username:
            command.extend(["--username", self.settings.comate_username])
        try:
            completed = subprocess.run(
                command, text=True, capture_output=True, timeout=5,
                env=self.comate_environment(),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {"enabled": True, "ready": False, "message": f"Comate 状态检查失败：{exc}"}
        output = (completed.stdout + completed.stderr).strip()
        not_logged_in = "未登录" in output or "not logged" in output.lower()
        ready = completed.returncode == 0 and not not_logged_in
        if ready:
            message = "Comate Zulu CLI 已登录"
        elif not_logged_in:
            message = "Comate Zulu CLI 未登录，请先执行 ./nsysscope login comate"
        else:
            detail = output.splitlines()[-1] if output else f"退出码 {completed.returncode}"
            message = f"Comate 状态检查失败：{detail}"
        return {
            "enabled": True,
            "ready": ready,
            "message": message,
        }

    def convert(self, package: Path, output: Path, prefix: str) -> None:
        command = [
            "python3", str(self.settings.converter), str(package), str(output),
            "--prefix", prefix,
        ]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode:
            raise RuntimeError(completed.stderr.strip() or "analysis conversion failed")

    @staticmethod
    def validate_analysis(path: Path) -> None:
        payload = json.loads(path.read_text())
        if payload.get("schemaVersion") != "1.0" or not payload.get("operators"):
            raise RuntimeError("analysis.json schema or operators are invalid")

    @staticmethod
    def find_package(job_dir: Path, prefix: str) -> Path:
        candidates = [job_dir, *[path.parent for path in job_dir.rglob(f"{prefix}_stage_table.csv")]]
        for path in candidates:
            if all((path / f"{prefix}{suffix}").exists() for suffix in CSV_SUFFIXES):
                return path
        raise RuntimeError("Agent run did not produce a complete six-table package")

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
