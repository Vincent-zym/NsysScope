from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import tempfile
import threading
import tomllib
import traceback
import zipfile
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
        self.log_lock = threading.Lock()

    def _run_tracked(
        self, job_id: str | None, command: list[str], *,
        env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        """Run a short deterministic subprocess (converter/validator) with a
        bounded timeout, registered under `job_id` so cancel() can kill it.

        Unlike run_process (used for the long-lived Agent call), this blocks
        until completion or timeout and returns a CompletedProcess, matching
        the call sites that previously used bare subprocess.run.
        """
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, env=env, start_new_session=True,
        )
        if job_id is not None:
            with self.lock:
                self.processes[job_id] = process
        try:
            stdout, _ = process.communicate(
                timeout=self.settings.subprocess_timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            self._kill_process_group(process)
            stdout, _ = process.communicate()
            raise RuntimeError(
                f"process timed out after {self.settings.subprocess_timeout_seconds}s: "
                f"{command[0]}"
            ) from None
        finally:
            if job_id is not None:
                with self.lock:
                    if self.processes.get(job_id) is process:
                        self.processes.pop(job_id, None)
        return subprocess.CompletedProcess(command, process.returncode, stdout, "")

    @staticmethod
    def _kill_process_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, 15)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, 9)
            except ProcessLookupError:
                pass

    def submit(self, job_id: str) -> None:
        self.pool.submit(self.run, job_id)

    def submit_conversion_retry(self, job_id: str) -> None:
        self.pool.submit(self.retry_conversion, job_id)

    @staticmethod
    def wipe_job_outputs(job_dir: Path) -> None:
        """Remove everything a cancelled job may have produced, keeping only
        the bounded job log so the user can see why/when it was cancelled.
        """
        preserve = {"logs"}
        for entry in job_dir.iterdir():
            if entry.name in preserve:
                continue
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)

    @staticmethod
    def job_log_path(job_dir: Path) -> Path:
        structured = job_dir / "logs" / "job.log"
        legacy = job_dir / "job.log"
        return legacy if legacy.exists() and not structured.exists() else structured

    def log(self, job_dir: Path, message: str) -> None:
        log_path = self.job_log_path(job_dir)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        text = message.rstrip() + "\n"
        payload = text.encode("utf-8", errors="replace")
        line_limit = self.settings.job_log_line_max_bytes
        if len(payload) > line_limit:
            marker = f"\n[日志单条内容已截断，原始大小 {len(payload)} bytes]\n".encode()
            prefix_size = max(0, line_limit - len(marker))
            payload = payload[:prefix_size] + marker
        with self.log_lock:
            current_size = log_path.stat().st_size if log_path.exists() else 0
            remaining = self.settings.job_log_max_bytes - current_size
            if remaining <= 0:
                log_path.touch()
                return
            if len(payload) >= remaining:
                marker = "\n[日志已达到大小上限，后续仅更新时间戳]\n".encode("utf-8")
                prefix_size = max(0, remaining - len(marker))
                payload = payload[:prefix_size] + marker[:remaining - prefix_size]
            with log_path.open("ab") as handle:
                handle.write(payload)

    def state(self, job_id: str, job_dir: Path, status: str, progress: int, message: str) -> None:
        self.store.update(job_id, status=status, progress=progress, message=message)
        self.log(job_dir, f"[{progress:03d}%] {message}")

    def is_cancelled(self, job_id: str) -> bool:
        return self.store.get(job_id)["status"] == "cancelled"

    def run(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job["status"] == "cancelled":
            return
        request = JobCreate.model_validate(job["request"])
        job_dir = Path(job["output_dir"])
        try:
            if request.mode == "existing_package":
                package = self.settings.resolve_allowed(
                    request.existing_package_path or "", kind="existing package",
                )
                self.state(job_id, job_dir, "converting", 70, "正在转换已有六表分析包")
                if package.is_file():
                    self.import_zip_package(package, job_dir, request, job_id=job_id)
                else:
                    csv_package = self.csv_package_dir(package)
                    analysis_path = package / "analysis.json"
                    if analysis_path.exists():
                        try:
                            prefix = self.detect_prefix(csv_package, request.prefix)
                        except RuntimeError:
                            prefix = None
                        if prefix:
                            self.validate_package(
                                csv_package, prefix, analysis_path=analysis_path,
                                job_id=job_id,
                            )
                            self.ensure_xlsx(csv_package, package / "xlsx", job_id=job_id)
                    else:
                        prefix = self.detect_prefix(csv_package, request.prefix)
                        self.validate_package(csv_package, prefix, job_id=job_id)
                        self.convert(
                            csv_package, package / "analysis.json", prefix, request,
                            job_id=job_id,
                        )
                        self.validate_package(
                            csv_package, prefix, analysis_path=package / "analysis.json",
                            job_id=job_id,
                        )
                        self.ensure_xlsx(csv_package, package / "xlsx", job_id=job_id)
            else:
                if self.is_cancelled(job_id):
                    return
                paths = self.resolve_inputs(request)
                context = {key: str(value) if value else None for key, value in paths.items()}
                context.update(request.model_dump())
                metadata_dir = job_dir / "metadata"
                metadata_dir.mkdir(exist_ok=True)
                (metadata_dir / "skill.json").write_text(
                    json.dumps(self.skill_provenance(), ensure_ascii=False, indent=2) + "\n",
                )
                (metadata_dir / "context.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                )
                if self.is_cancelled(job_id):
                    return
                sqlite_path = self.export_nsys(job_id, job_dir, paths["report"])
                context["sqlite_path"] = str(sqlite_path)
                (metadata_dir / "context.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                )
                if self.is_cancelled(job_id):
                    return
                self.run_agent(job_id, job_dir, request, paths, sqlite_path)
                if self.is_cancelled(job_id):
                    return
                self.state(job_id, job_dir, "converting", 88, "正在构建前端 analysis.json")
                package = self.find_package(job_dir, request.prefix)
                self.validate_package(package, request.prefix, job_id=job_id)
                self.convert(
                    package, job_dir / "analysis.json", request.prefix, request,
                    job_id=job_id,
                )
                self.validate_package(
                    package, request.prefix, analysis_path=job_dir / "analysis.json",
                    job_id=job_id,
                )
                trace_path = self.organize_result_package(
                    job_dir, package, request.prefix, sqlite_path, job_id=job_id,
                )
                context["sqlite_path"] = str(trace_path)
                (metadata_dir / "context.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                )

            if self.is_cancelled(job_id):
                return
            self.state(job_id, job_dir, "validating", 95, "正在校验前端数据契约")
            self.validate_analysis(job_dir / "analysis.json")
            # Never let a terminal write clobber a cancellation that raced in
            # after the last cancellation check above.
            if not self.is_cancelled(job_id):
                self.state(job_id, job_dir, "succeeded", 100, "分析完成")
        except Exception as exc:
            if self.is_cancelled(job_id):
                self.log(job_dir, "[cancelled] Agent 进程已终止")
            else:
                self.log(job_dir, traceback.format_exc())
                self.store.update(
                    job_id, status="failed", progress=100,
                    message="分析失败", error=str(exc),
                )
        finally:
            staged_skill = job_dir / ".comate"
            if staged_skill.exists():
                shutil.rmtree(staged_skill)

    def retry_conversion(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job["status"] == "cancelled":
            return
        request = JobCreate.model_validate(job["request"])
        job_dir = Path(job["output_dir"])
        try:
            self.state(job_id, job_dir, "converting", 88, "正在重试构建前端 analysis.json")
            package = self.find_package(job_dir, request.prefix)
            prefix = self.detect_prefix(package, request.prefix)
            self.convert(package, job_dir / "analysis.json", prefix, request, job_id=job_id)
            if request.mode == "existing_package" or package == job_dir / "csv":
                self.ensure_xlsx(package, job_dir / "xlsx", job_id=job_id)
            else:
                context_path = job_dir / "metadata" / "context.json"
                context = json.loads(context_path.read_text()) if context_path.exists() else {}
                candidates = [
                    Path(context["sqlite_path"]) if context.get("sqlite_path") else None,
                    *job_dir.glob("trace/*.sqlite"),
                    *job_dir.glob("*.sqlite"),
                ]
                sqlite_path = next(
                    (path for path in candidates if path is not None and path.exists()),
                    None,
                )
                if sqlite_path is None:
                    raise RuntimeError("conversion retry cannot locate the SQLite trace")
                trace_path = self.organize_result_package(
                    job_dir, package, prefix, sqlite_path, job_id=job_id,
                )
                context["sqlite_path"] = str(trace_path)
                context_path.write_text(
                    json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                )
            if self.is_cancelled(job_id):
                return
            self.state(job_id, job_dir, "validating", 95, "正在校验前端数据契约")
            self.validate_analysis(job_dir / "analysis.json")
            if not self.is_cancelled(job_id):
                self.state(job_id, job_dir, "succeeded", 100, "分析完成")
        except Exception as exc:
            if self.is_cancelled(job_id):
                self.log(job_dir, "[cancelled] 转换重试已终止")
            else:
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

    def skill_provenance(self) -> dict[str, str]:
        root = self.settings.skill_dir.resolve()
        project = Path(__file__).resolve().parents[1]
        bundled = project / "bundled" / "skills"
        if root.is_relative_to(bundled):
            source = "bundled"
        elif ".codex/skills" in root.as_posix():
            source = "codex"
        else:
            source = "external"
        digest = hashlib.sha256()
        ignored = {"result", "evals", "__pycache__"}
        for path in sorted(root.rglob("*")):
            if (
                not path.is_file()
                or any(part in ignored for part in path.relative_to(root).parts)
                or path.suffix in {".pyc", ".pyo"}
            ):
                continue
            digest.update(path.relative_to(root).as_posix().encode())
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return {
            "name": "sglang-nsys-static-analysis",
            "source": source,
            "path": str(root),
            "sha256": digest.hexdigest(),
        }

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
        prompt_path = job_dir / "metadata" / "prompt.md"
        prompt_path.parent.mkdir(exist_ok=True)
        prompt_path.write_text(prompt)
        output_message = job_dir / "metadata" / "agent-final.txt"
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
            for path in (*paths.values(), self.settings.skill_dir)
            if path is not None
        }):
            command.extend(["--add-dir", directory])
        command.extend(["--output-last-message", str(output_message), "-"])
        self.run_process(
            job_id, job_dir, command, stdin=prompt,
            heartbeat_seconds=self.settings.agent_heartbeat_seconds,
            heartbeat_message="Codex Agent 仍在运行",
        )

    def run_comate(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
    ) -> None:
        status = self._comate_status()
        if not status["ready"]:
            raise RuntimeError(status["message"])
        self.state(job_id, job_dir, "analyzing", 30, "Comate Skill Agent 正在分析模型与时间线")
        prompt = self.build_prompt(job_dir, request, paths, sqlite_path)
        metadata_dir = job_dir / "metadata"
        metadata_dir.mkdir(exist_ok=True)
        (metadata_dir / "prompt.md").write_text(prompt)
        self.stage_comate_skill(job_dir)
        command = [
            self.settings.comate_bin, "run",
            "--query", prompt,
            "--cwd", str(job_dir),
            "--mode", "Agent",
            "--activate-skill", "sglang-nsys-static-analysis",
            "--display", "task-json",
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
            output_formatter=self.format_comate_output,
            heartbeat_seconds=self.settings.agent_heartbeat_seconds,
            heartbeat_message="Comate Agent 仍在运行",
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
        return f"""Use the `sglang-nsys-static-analysis` skill at:
{self.settings.skill_dir}

Read that SKILL.md completely before analysis. This exact path is the task's
selected Skill version and overrides any older installed copy.

<user_acceptance_criteria>
{request.notes.strip() or "No additional scope constraint was supplied."}
</user_acceptance_criteria>

The acceptance criteria above are binding task requirements, not optional notes.
They have priority when selecting the analyzed layer, branch, and repeating-unit boundary.
If the user requests a specific layer subtype or branch, select exactly one complete layer
matching that subtype; do not silently widen it to a multi-layer architectural period.
In particular, "GLM5.2 non-shared Indexer" means one complete GLM5.2 layer containing the
non-shared Indexer, not the four-layer full/shared Indexer cycle. If the requested scope
cannot be proven from the supplied trace and materials, fail validation and explain why
instead of substituting another scope.

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

Write all artifacts only under:
{job_dir}

Requirements:
1. Run `scripts/audit_runtime_evidence.py` first. Captured server args/environment
   override launch intent and source defaults; record every conflict.
2. Read model evidence and write `{request.prefix}_architecture_taxonomy.json`
   before mapping kernels. Validate it with
   `scripts/validate_architecture_taxonomy.py`. It must define every structural
   position/variant, ordered functional modules, discriminators, shared paths
   and indivisible fusion groups from this model's evidence. Keep `module` as
   fine-grained attribution, but default each variant's `功能模块` rollup to 5–8
   architecture stages. Merge implementation substeps into their enclosing
   input/core/output/routing/expert/merge phase. More than eight stages requires
   a current-model `granularity_exception`; never copy `module` one-to-one into
   `功能模块`.
3. Select and prove the exact complete unit required by `user_acceptance_criteria`.
   Without an explicit subtype request, include every distinct layer variant in
   the smallest structural cycle; never present one convenient subtype as the
   model's generic single layer.
4. Annotate every selected kernel with unit_position, unit_id and unit_variant.
   Aggregate stages by position × id × variant × functional module, never by
   functional-module label alone. Generate the normalized six CSV tables with
   prefix `{request.prefix}`.
5. End every CSV with its required total row. In the operator overview, copy
   the origin `module` column immediately before `算子名称`. Accumulated
   operator/category/stage totals may exceed wall time or 100% under overlap.
6. Write `{request.prefix}_analysis_manifest.json`, semantic map, stable-statistics sidecar,
   and `validation_report.json`.
7. Compute MFU for every eligible GEMM when shape and a bundled verified hardware
   profile exist. Record logical/physical shape, compute dtype, dense per-GPU
   peak and source; "new model" is not a reason to leave MFU blank.
8. Never infer CPU delay from zero-kernel GPU idle. Require CUDA Runtime launch
   timestamp evidence or label the interval GPU idle/queue/dependency gap.
9. Run `scripts/validate_analysis_package.py` with the taxonomy and finish only when every required
   invariant, including the requested scope, passes. The manifest boundary
   evidence must explicitly show how the selected unit satisfies the acceptance
   criteria.
10. Never edit input reports, config, launch files, design notes, or model source.
"""

    def run_process(
        self, job_id: str, job_dir: Path, command: list[str], stdin: str | None = None,
        redacted_values: set[str] | None = None,
        environment: dict[str, str] | None = None,
        output_formatter: Callable[[str], str | None] | None = None,
        heartbeat_seconds: int = 0,
        heartbeat_message: str = "Agent 仍在运行",
    ) -> None:
        hidden = redacted_values or set()
        displayed = ["<prompt>" if item in hidden else item for item in command]
        self.log(job_dir, f"$ {shlex.join(displayed)}")
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE if stdin is not None else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, cwd=job_dir,
            env=environment or {**os.environ, "NO_COLOR": "1"},
            start_new_session=True,
        )
        with self.lock:
            self.processes[job_id] = process
        heartbeat_stop = threading.Event()
        heartbeat_thread: threading.Thread | None = None
        if heartbeat_seconds > 0:
            def emit_heartbeat() -> None:
                while not heartbeat_stop.wait(heartbeat_seconds):
                    if process.poll() is not None:
                        return
                    self.log(job_dir, f"[heartbeat] {heartbeat_message}")

            heartbeat_thread = threading.Thread(
                target=emit_heartbeat,
                name=f"nsysscope-heartbeat-{job_id}",
                daemon=True,
            )
            heartbeat_thread.start()
        try:
            assert process.stdout is not None
            if stdin is not None and process.stdin is not None:
                process.stdin.write(stdin)
                process.stdin.close()
            for line in process.stdout:
                if line.strip():
                    rendered = output_formatter(line) if output_formatter else line
                    if rendered:
                        self.log(job_dir, rendered)
            code = process.wait()
            if code:
                raise RuntimeError(f"process exited with code {code}: {command[0]}")
        finally:
            heartbeat_stop.set()
            if heartbeat_thread is not None:
                heartbeat_thread.join(timeout=1)
            with self.lock:
                self.processes.pop(job_id, None)

    @staticmethod
    def format_comate_output(line: str) -> str:
        raw_size = len(line.encode("utf-8", errors="replace"))
        if raw_size > 256 * 1024:
            return f"[Comate] 收到任务结果（{raw_size} bytes，详细会话未写入日志）"
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            compact = " ".join(line.split())
            return f"[Comate] {compact[:1000]}"
        event_type = str(payload.get("type") or "task-json")
        error = payload.get("error") or payload.get("failure_message")
        if error:
            return f"[Comate:{event_type}] 错误：{str(error)[:1000]}"
        status = payload.get("status")
        if isinstance(payload.get("task"), dict):
            status = status or payload["task"].get("status")
        suffix = f"，状态：{status}" if status else ""
        return f"[Comate] 收到 {event_type} 结果（{raw_size} bytes{suffix}）"

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

        payload: object | None = None
        executable = shutil.which(self.settings.codex_bin)
        if executable:
            try:
                completed = subprocess.run(
                    [executable, "debug", "models"],
                    text=True, capture_output=True, timeout=15,
                )
                if completed.returncode == 0:
                    payload = json.loads(completed.stdout)
            except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError):
                pass

        if payload is None:
            cache_path = codex_home / "models_cache.json"
            try:
                payload = json.loads(cache_path.read_text())
            except (OSError, json.JSONDecodeError):
                payload = {}

        records = payload.get("models", []) if isinstance(payload, dict) else payload
        choices: list[dict[str, str]] = []
        seen: set[str] = set()
        for record in records if isinstance(records, list) else []:
            if not isinstance(record, dict) or record.get("visibility") == "hide":
                continue
            model_id = str(record.get("slug") or record.get("id") or "")
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            choices.append({
                "id": model_id,
                "label": str(record.get("display_name") or record.get("name") or model_id),
            })
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
            match = re.match(
                r"^[*•·\s-]*(.*?)\s+[(（]([^()（）]+)[)）]\s*$",
                line,
            )
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

    def convert(
        self, package: Path, output: Path, prefix: str,
        request: JobCreate | None = None, job_id: str | None = None,
    ) -> None:
        command = [
            "python3", str(self.settings.converter), str(package), str(output),
            "--prefix", prefix,
        ]
        if request is not None:
            command.extend([
                "--model", request.model_name,
                "--stage", request.stage,
                "--hardware", request.hardware,
            ])
        completed = self._run_tracked(job_id, command)
        if completed.returncode:
            raise RuntimeError(completed.stdout.strip() or "analysis conversion failed")

    def ensure_xlsx(self, csv_dir: Path, xlsx_dir: Path, job_id: str | None = None) -> None:
        command = [
            "python3", str(self.settings.xlsx_converter),
            str(csv_dir), str(xlsx_dir),
        ]
        completed = self._run_tracked(job_id, command)
        if completed.returncode:
            raise RuntimeError(completed.stdout.strip() or "CSV to XLSX conversion failed")

    def import_zip_package(
        self, archive_path: Path, result_dir: Path, request: JobCreate,
        job_id: str | None = None,
    ) -> None:
        if archive_path.suffix.lower() != ".zip":
            raise RuntimeError("existing package file must be a .zip archive")
        with tempfile.TemporaryDirectory(
            prefix="import-", dir=self.settings.data_dir,
        ) as temporary:
            extracted = Path(temporary)
            with zipfile.ZipFile(archive_path) as archive:
                total_size = sum(item.file_size for item in archive.infolist())
                if total_size > 100 * 1024 * 1024 * 1024:
                    raise RuntimeError("ZIP package expands beyond the 100 GiB safety limit")
                for item in archive.infolist():
                    if item.is_dir():
                        continue
                    mode = (item.external_attr >> 16) & 0xFFFF
                    if stat.S_ISLNK(mode):
                        raise RuntimeError(f"unsafe ZIP entry (symlink): {item.filename}")
                    target = (extracted / item.filename).resolve()
                    if not target.is_relative_to(extracted.resolve()):
                        raise RuntimeError(f"unsafe ZIP path: {item.filename}")
                archive.extractall(extracted)
            source = self.find_import_root(extracted, request.prefix)
            csv_source = self.csv_package_dir(source)
            prefix = self.detect_prefix(csv_source, request.prefix)
            csv_dir = result_dir / "csv"
            xlsx_dir = result_dir / "xlsx"
            metadata_dir = result_dir / "metadata"
            trace_dir = result_dir / "trace"
            for directory in (csv_dir, xlsx_dir, metadata_dir, trace_dir):
                directory.mkdir(exist_ok=True)
            for suffix in CSV_SUFFIXES:
                shutil.copy2(csv_source / f"{prefix}{suffix}", csv_dir)
            source_analysis = source / "analysis.json"
            if source_analysis.exists():
                shutil.copy2(source_analysis, result_dir / "analysis.json")
            else:
                self.convert(
                    csv_source, result_dir / "analysis.json", prefix, request,
                    job_id=job_id,
                )
            self.ensure_xlsx(csv_dir, xlsx_dir, job_id=job_id)
            for sidecar in source.rglob("*.json"):
                if sidecar.name in {"analysis.json", "nsysscope-package.json"}:
                    continue
                shutil.copy2(sidecar, metadata_dir / sidecar.name)
            traces = [*source.rglob("*.sqlite")]
            for trace in traces:
                shutil.copy2(trace, trace_dir / trace.name)
            package_manifest = {
                "schemaVersion": "1.0",
                "kind": "nsysscope-analysis-package",
                "analysis": "analysis.json",
                "csvDirectory": "csv",
                "xlsxDirectory": "xlsx",
                "trace": f"trace/{traces[0].name}" if traces else None,
                "log": "logs/job.log",
                "metadataDirectory": "metadata",
                "prefix": prefix,
                "tables": [f"{prefix}{suffix}" for suffix in CSV_SUFFIXES],
                "importedFrom": str(archive_path),
            }
            (result_dir / "nsysscope-package.json").write_text(
                json.dumps(package_manifest, ensure_ascii=False, indent=2) + "\n",
            )

    @classmethod
    def find_import_root(cls, extracted: Path, requested: str) -> Path:
        candidates = [extracted]
        for path in extracted.rglob("*_stage_table.csv"):
            table_dir = path.parent
            if table_dir.name == "csv":
                candidates.append(table_dir.parent)
            candidates.append(table_dir)
        for candidate in dict.fromkeys(candidates):
            csv_dir = cls.csv_package_dir(candidate)
            try:
                cls.detect_prefix(csv_dir, requested)
            except RuntimeError:
                continue
            return candidate
        raise RuntimeError("ZIP package does not contain one complete six-table directory")

    def organize_result_package(
        self, result_dir: Path, package_dir: Path, prefix: str, sqlite_path: Path,
        job_id: str | None = None,
    ) -> Path:
        csv_dir = result_dir / "csv"
        xlsx_dir = result_dir / "xlsx"
        trace_dir = result_dir / "trace"
        metadata_dir = result_dir / "metadata"
        for directory in (csv_dir, xlsx_dir, trace_dir, metadata_dir):
            directory.mkdir(exist_ok=True)

        csv_files = []
        for suffix in CSV_SUFFIXES:
            source = package_dir / f"{prefix}{suffix}"
            target = csv_dir / source.name
            shutil.move(str(source), target)
            csv_files.append(target.name)
        for source_dir in dict.fromkeys((package_dir, result_dir)):
            for extra_csv in source_dir.glob("*.csv"):
                shutil.move(str(extra_csv), metadata_dir / extra_csv.name)

        self.ensure_xlsx(csv_dir, xlsx_dir, job_id=job_id)
        if sqlite_path.resolve() != trace_path.resolve():
            if sqlite_path.is_relative_to(result_dir):
                shutil.move(str(sqlite_path), trace_path)
            else:
                shutil.copy2(sqlite_path, trace_path)

        for source_dir in dict.fromkeys((package_dir, result_dir)):
            for sidecar in source_dir.glob("*.json"):
                if sidecar.name in {"analysis.json", "nsysscope-package.json"}:
                    continue
                shutil.move(str(sidecar), metadata_dir / sidecar.name)
        staged_skill = result_dir / ".comate"
        if staged_skill.exists():
            shutil.rmtree(staged_skill)

        manifest = {
            "schemaVersion": "1.0",
            "kind": "nsysscope-analysis-package",
            "analysis": "analysis.json",
            "csvDirectory": "csv",
            "xlsxDirectory": "xlsx",
            "trace": f"trace/{trace_path.name}",
            "log": "logs/job.log",
            "metadataDirectory": "metadata",
            "prefix": prefix,
            "tables": csv_files,
        }
        (result_dir / "nsysscope-package.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        )
        return trace_path

    @staticmethod
    def csv_package_dir(package: Path) -> Path:
        csv_dir = package / "csv"
        return csv_dir if csv_dir.is_dir() else package

    @staticmethod
    def validate_analysis(path: Path) -> None:
        payload = json.loads(path.read_text())
        operators = payload.get("operators")
        if payload.get("schemaVersion") != "1.0" or not isinstance(operators, list) or not operators:
            raise RuntimeError("analysis.json schema or operators are invalid")
        categories = {"core", "communication", "auxiliary"}
        if any(operator.get("category") not in categories for operator in operators):
            raise RuntimeError("analysis.json contains an invalid operator category")
        summary = payload.get("summary") or {}
        if summary.get("operatorCount") != len(operators):
            raise RuntimeError("analysis.json operatorCount disagrees with operators")
        if not summary.get("devices") or int(summary.get("stableSamples", 0)) < 1:
            raise RuntimeError("analysis.json stable sample/device scope is missing")
        if summary.get("heterogeneous"):
            variants = set(summary.get("distinctUnitVariants") or [])
            operator_variants = {
                operator.get("unitVariant")
                for operator in operators
                if operator.get("unitVariant")
            }
            if len(variants) < 2 or operator_variants != variants:
                raise RuntimeError("analysis.json loses heterogeneous unit variants")
            if any(
                operator.get("unitPosition") is None or not operator.get("unitId")
                for operator in operators
            ):
                raise RuntimeError("heterogeneous analysis has unscoped operators")
            if summary.get("durationLabel") in {"单层耗时", "平均单层耗时"}:
                raise RuntimeError("heterogeneous cycle is mislabeled as single-layer duration")
            stages = payload.get("stages") or []
            stage_variants = {
                stage.get("unitVariant")
                for stage in stages
                if stage.get("unitVariant")
            }
            pattern_rollup = bool(stages) and all(
                stage.get("unitId") == "__pattern_total__"
                and stage.get("unitPosition") is None
                and not stage.get("unitVariant")
                for stage in stages
            )
            if (not pattern_rollup and (
                stage_variants != variants or any(
                    stage.get("unitPosition") is None or not stage.get("unitId")
                    for stage in stages
                )
            )):
                raise RuntimeError("heterogeneous stage view loses structural-unit identity")
            units = payload.get("units") or []
            if len(units) < 2 or {
                unit.get("variant") for unit in units if unit.get("variant")
            } != variants:
                raise RuntimeError("heterogeneous analysis needs an explicit structural-unit index")
            unit_positions = [unit.get("position") for unit in units]
            if unit_positions != list(range(1, len(units) + 1)):
                raise RuntimeError("structural-unit positions must be contiguous and ordered")

    def validate_package(
        self, package: Path, prefix: str, *, analysis_path: Path | None = None,
        job_id: str | None = None,
    ) -> None:
        validator = self.settings.skill_dir / "scripts" / "validate_analysis_package.py"
        if not validator.is_file():
            raise RuntimeError(f"analysis Skill is missing package validator: {validator}")
        command = [
            shutil.which("python3") or "python3",
            str(validator),
            str(package),
            "--prefix", prefix,
        ]
        if analysis_path is not None:
            command.extend(["--analysis-json", str(analysis_path)])
        completed = self._run_tracked(job_id, command)
        if completed.returncode:
            raise RuntimeError(f"analysis package validation failed: {completed.stdout.strip()}")

    def publish_to_popo(self, job_id: str, job_dir: Path, username: str) -> str:
        analysis_path = job_dir / "analysis.json"
        if not analysis_path.is_file():
            raise RuntimeError("analysis.json is missing; cannot publish")
        return self._publish_analysis_bytes(
            analysis_path.read_bytes(), slug=f"nsysscope-{job_id}", username=username,
        )

    def publish_analysis_payload(self, analysis: dict, username: str) -> str:
        payload_bytes = json.dumps(analysis, ensure_ascii=False).encode("utf-8")
        slug = f"nsysscope-{secrets.token_hex(8)}"
        return self._publish_analysis_bytes(payload_bytes, slug=slug, username=username)

    def _publish_analysis_bytes(
        self, analysis_bytes: bytes, *, slug: str, username: str,
    ) -> str:
        static_dir = Path(__file__).resolve().parent / "static"
        index_html = static_dir / "index.html"
        assets_dir = static_dir / "assets"
        if not index_html.is_file() or not assets_dir.is_dir():
            raise RuntimeError("dashboard static assets are missing; cannot publish")
        upload_script = self.settings.popo_upload_script
        if not upload_script.is_file():
            raise RuntimeError("popo upload script is not available")
        if not username:
            raise RuntimeError("no popo username was provided")

        site_dir = Path(tempfile.mkdtemp(prefix="nsysscope-popo-"))
        try:
            shutil.copy2(index_html, site_dir / "index.html")
            shutil.copytree(assets_dir, site_dir / "assets")
            title = f"NsysScope 分析结果 {slug}"

            # The outbound network path drops HTTPS requests whose body
            # exceeds roughly 300KB (verified: <=280KB succeeds, >=290KB
            # fails with a TLS-layer SSLEOFError on every retry). The shell
            # and JS assets alone are ~250KB; adding the analysis JSON often
            # pushes the single-request body past that ceiling. Split into
            # two requests instead: publish the shell first, then re-deploy
            # onto the same slug once the JSON is added, so no single
            # request body needs to exceed the ceiling.
            shell_command = [
                shutil.which("python3") or "python3",
                str(upload_script),
                "--username", username,
                "--title", title,
                "--slug", slug,
                "--visibility", "internal",
                "--base", str(site_dir),
                "--entry", "index.html",
                "--project-dir", str(site_dir),
            ]
            self._run_popo_upload(shell_command)

            (site_dir / "demo-analysis.json").write_bytes(analysis_bytes)
            full_command = [
                shutil.which("python3") or "python3",
                str(upload_script),
                "--username", username,
                "--title", title,
                "--slug", slug,
                "--previous-slug", slug,
                "--base", str(site_dir),
                "--entry", "index.html",
                "--project-dir", str(site_dir),
            ]
            payload = self._run_popo_upload(full_command)
            published_slug = payload.get("data", {}).get("slug", slug)
            return f"https://{published_slug}.popo.baidu-int.com"
        finally:
            shutil.rmtree(site_dir, ignore_errors=True)

    @staticmethod
    def _run_popo_upload(command: list[str]) -> dict:
        # The outbound HTTP(S)_PROXY inherited from the parent process drops
        # POST bodies once they exceed roughly 300KB (verified: proxied
        # requests fail with a TLS-layer SSLEOFError above that size, while
        # direct connections to api.popo.baidu-int.com succeed at any size
        # tested). Popo's upload endpoint is an internal host that does not
        # need the proxy, so strip proxy vars for this subprocess only.
        env = {
            key: value for key, value in os.environ.items()
            if key.lower() not in {
                "http_proxy", "https_proxy", "all_proxy",
                "http_proxy_url", "no_proxy",
            }
        }
        completed = subprocess.run(command, text=True, capture_output=True, env=env)
        if completed.returncode:
            detail = (completed.stdout + completed.stderr).strip()
            raise RuntimeError(f"popo publish failed: {detail}")
        try:
            payload = json.loads(completed.stdout.strip().splitlines()[-1])
        except (json.JSONDecodeError, IndexError) as exc:
            raise RuntimeError(
                f"popo publish returned an unexpected response: {completed.stdout.strip()}"
            ) from exc
        if not payload.get("success"):
            raise RuntimeError(f"popo publish failed: {payload}")
        return payload

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
            self._kill_process_group(process)
            return True
        return False
