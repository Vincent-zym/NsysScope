from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import tarfile
import threading
import time
import traceback
import urllib.request
import zipfile

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 ships no tomllib
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        # Only used to read the default model out of Codex's config.toml, so a
        # missing parser costs the configured default and nothing else.
        tomllib = None  # type: ignore[assignment]
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from .config import Settings
from .models import JobCreate
from .store import JobStore


AGENT_CSV_SUFFIXES = (
    "_operator_origin_table.csv",
    "_opreator_table.csv",
    "_core_compute_table.csv",
    "_auxiliary_operator_table.csv",
    "_op_classification_table.csv",
    "_stage_table.csv",
)

# The seventh table carries the forward-step breakdown -- the only place the package
# says how the measured unit relates to a whole forward step -- so a package without it
# is incomplete. It is generated from the trace after the agent's six tables exist,
# which is why package detection keys off AGENT_CSV_SUFFIXES while completeness checks
# use CSV_SUFFIXES.
FORWARD_PIPELINE_SUFFIX = "_forward_pipeline_table.csv"
CSV_SUFFIXES = AGENT_CSV_SUFFIXES + (FORWARD_PIPELINE_SUFFIX,)

# CPU time an agent must burn between two heartbeats to count as working. A dropped
# model request leaves the process parked on a socket at roughly one clock tick per
# 20 seconds, while a long reasoning turn keeps at least one core busy. Expressed as a
# floor; the effective threshold scales with the heartbeat interval, see
# CPU_PROGRESS_FRACTION.
CPU_PROGRESS_SECONDS = 1.0

# The Comate Zulu CLI is a modern-JS bundle: on Node <= 12 it dies while loading and
# still exits 0, so its version is a hard precondition rather than a warning.
MIN_NODE_MAJOR = 18

# Fraction of a heartbeat interval of CPU time that counts as "the agent is thinking".
# A flat 1 second was too generous: an idle Node event loop, and especially the burst
# of timers it catches up on after the host resumes from suspend, clears 1 second
# easily and kept resetting the stall timer on an agent that had already finished.
CPU_PROGRESS_FRACTION = 0.1

# How long the agent's conversation file must sit untouched before complete outputs
# are taken as "the run is over, only the CLI is still parked". Measured runs took
# up to 14 minutes to write the mandatory sidecars *after* the six tables appeared,
# and a busy agent's longest observed silence was about 3 minutes, so 30 minutes
# keeps a wide margin on both sides. Cutting in too early would kill the agent
# mid-validation and surface as a confusing package error.
EARLY_FINISH_IDLE_SECONDS = 1800

# Lines of a child process's raw stdout kept for crash-signature matching.
RAW_OUTPUT_TAIL_LINES = 400


def elapsed_seconds() -> float:
    """A clock for measuring durations that keeps counting while the host sleeps.

    `time.monotonic()` excludes suspend time on Linux, which made the stall timer
    under-count badly: one job's heartbeat showed its agent session ageing 60
    minutes (wall clock) while `idle` advanced only 9, so a 30-minute stall
    timeout never fired. CLOCK_BOOTTIME includes suspend and, unlike
    `time.time()`, cannot jump backwards when the clock is stepped.
    """
    boottime = getattr(time, "CLOCK_BOOTTIME", None)
    if boottime is not None:
        try:
            return time.clock_gettime(boottime)
        except OSError:
            pass
    return time.monotonic()

# `nsys export` refuses a report produced by a newer Nsight Systems than itself, and a
# .nsys-rep's first line carries the exact build that wrote it, e.g.
# "NVIDIA Tegra Profiler Report 2026@4@1@191-264138605071v0." -- which is the same
# build string the public devtools apt repo uses for its CLI-only packages. That makes
# "which nsys can read this file" a lookup, not a guess.
REPORT_VERSION_RE = re.compile(rb"Report (\d+)@(\d+)@(\d+)@(\d+)-(\S+?)\.")
NSYS_VERSION_RE = re.compile(r"version (\d+)\.(\d+)\.(\d+)\.(\d+)")
# The devtools repo is laid out per distro and per architecture. Only x86_64 has ever
# been exercised here, but hardcoding amd64 made an aarch64 host download a package it
# cannot execute, so map the machine name and let an unknown one fail on the index
# fetch (which falls back to the local nsys) rather than on a wrong binary.
NSYS_REPO_ARCH = {
    "x86_64": "amd64", "amd64": "amd64",
    "aarch64": "arm64", "arm64": "arm64",
}.get(platform.machine().lower(), platform.machine().lower())
NSYS_REPO_URLS = tuple(
    url for url in (
        # Overridable so an air-gapped site can point at an internal mirror of the same
        # layout (a Packages index plus the .deb files beside it).
        os.getenv("NSYSSCOPE_NSYS_REPO"),
        # NVIDIA's China CDN first: it serves a byte-identical index and measured
        # faster from here (0.47 vs 0.33 MB/s on a 64 MB sample). The domestic
        # university/Aliyun mirrors are not an option -- they only ever carried the
        # CUDA repo, and those paths now 404.
        f"https://developer.download.nvidia.cn/devtools/repos/ubuntu2004/{NSYS_REPO_ARCH}",
        f"https://developer.download.nvidia.com/devtools/repos/ubuntu2004/{NSYS_REPO_ARCH}",
    ) if url
)
# Same candidates, same order as the launcher's proxy_candidates(): caller's own
# setting first, then the environment's, then the two office proxies. Only downloads
# use them; agent CLIs must run with no proxy variables at all.
DOWNLOAD_PROXY_CANDIDATES = (
    "http://agent.baidu.com:8891",
    "http://10.162.37.16:8128",
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
        # Ellipsis = "not probed yet"; None is a valid result meaning "direct".
        self._download_proxy: object = ...
        self.nsys_repo_url = NSYS_REPO_URLS[0]

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

    @staticmethod
    def detect_popo_accounts() -> list[str]:
        """List local usernames that have a cached popo ugate token, so the
        frontend can offer them instead of asking the user to type one from
        scratch. Reads only filenames, never token contents.
        """
        uuap_dir = Path.home() / ".config" / "uuap"
        if not uuap_dir.is_dir():
            return []
        prefix = ".eac_ugate_token_"
        return sorted(
            entry.name.removeprefix(prefix)
            for entry in uuap_dir.iterdir()
            if entry.is_file() and entry.name.startswith(prefix)
        )

    @staticmethod
    def has_popo_token(username: str) -> bool:
        """Whether a cached ugate token already exists for this username."""
        uuap_dir = Path.home() / ".config" / "uuap"
        return (uuap_dir / f".eac_ugate_token_{username}").is_file()

    @staticmethod
    def save_popo_token(username: str, token: str) -> None:
        """Cache a ugate token supplied through the web UI.

        The bundled popo upload script (and the get-ugate-token CLI helper it
        shares a cache format with) only knows how to prompt for this token in
        a conversational agent session -- a standalone browser user has no such
        session to answer in. Accepting the token straight from the publish
        dialog and writing it in the same cache format lets the upload script
        find it without ever needing that prompt.
        """
        token = token.strip()
        if not token:
            raise RuntimeError("token 不能为空")
        uuap_dir = Path.home() / ".config" / "uuap"
        uuap_dir.mkdir(parents=True, exist_ok=True)
        cache_file = uuap_dir / f".eac_ugate_token_{username}"
        cache_file.write_text(
            json.dumps({"token": token, "permanent": True}), encoding="utf-8",
        )

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

    def log_tail(self, job_dir: Path, limit: int = 64 * 1024) -> str:
        """Tail of the job log, for post-run inspection of an agent's own output."""
        path = self.job_log_path(job_dir)
        try:
            with path.open("rb") as handle:
                handle.seek(0, os.SEEK_END)
                handle.seek(max(0, handle.tell() - limit))
                return handle.read().decode("utf-8", "replace")
        except OSError:
            return ""

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

    def log_quietly(self, job_dir: Path, message: str) -> None:
        """Append to the job log, swallowing filesystem errors.

        For use on failure paths, where the log write is the least important thing
        left to do: a full or read-only disk must not replace the real error with an
        OSError, nor abort the caller before it has recorded the job's status.
        """
        try:
            self.log(job_dir, message)
        except OSError:
            pass

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
                self.state(job_id, job_dir, "converting", 70, "正在转换已有七表分析包")
                if package.is_file():
                    self.import_zip_package(package, job_dir, request, job_id=job_id)
                    self.ensure_final_report(
                        job_id, job_dir, job_dir,
                        self.detect_prefix(job_dir / "csv", request.prefix),
                    )
                else:
                    csv_package = self.csv_package_dir(package)
                    analysis_path = package / "analysis.json"
                    if analysis_path.exists():
                        try:
                            prefix = self.detect_prefix(csv_package, request.prefix)
                        except RuntimeError:
                            prefix = None
                        if prefix:
                            self.ensure_forward_pipeline(
                                job_id, job_dir, csv_package, prefix,
                                self.package_trace(package),
                                metadata_root=package / "metadata",
                            )
                            self.validate_package(
                                csv_package, prefix, analysis_path=analysis_path,
                                job_id=job_id, log_to=job_dir,
                            )
                            self.ensure_final_report(job_id, job_dir, package, prefix)
                            self.organize_result_package(
                                package, csv_package, prefix,
                                self.package_trace(package), job_id=job_id,
                            )
                    else:
                        prefix = self.detect_prefix(csv_package, request.prefix)
                        self.ensure_forward_pipeline(
                            job_id, job_dir, csv_package, prefix,
                            self.package_trace(package),
                            metadata_root=package / "metadata",
                        )
                        self.validate_package(
                            csv_package, prefix, job_id=job_id, log_to=job_dir,
                        )
                        self.convert(
                            csv_package, package / "analysis.json", prefix, request,
                            job_id=job_id,
                        )
                        self.validate_package(
                            csv_package, prefix, analysis_path=package / "analysis.json",
                            job_id=job_id, log_to=job_dir,
                        )
                        self.ensure_final_report(job_id, job_dir, package, prefix)
                        self.organize_result_package(
                            package, csv_package, prefix,
                            self.package_trace(package), job_id=job_id,
                        )
            else:
                if self.is_cancelled(job_id):
                    return
                paths = self.resolve_inputs(request, job_dir)
                context = {key: str(value) if value else None for key, value in paths.items()}
                context.update(request.model_dump())
                metadata_dir = job_dir / "metadata"
                metadata_dir.mkdir(exist_ok=True)
                provenance = self.skill_provenance()
                (metadata_dir / "skill.json").write_text(
                    json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
                )
                self.log_skill_source(job_dir, provenance)
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
                dispatch_cache = self.try_build_dispatch_cache(job_id, job_dir, request, paths)
                if dispatch_cache is not None:
                    context["dispatch_cache_path"] = str(dispatch_cache)
                    (metadata_dir / "context.json").write_text(
                        json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                    )
                if self.is_cancelled(job_id):
                    return
                self.run_agent(job_id, job_dir, request, paths, sqlite_path, dispatch_cache)
                if self.is_cancelled(job_id):
                    return
                self.state(job_id, job_dir, "converting", 88, "正在构建前端 analysis.json")
                package = self.find_package(job_dir, request.prefix)
                # The agent sometimes names its tables after the model it found in the
                # trace rather than the requested prefix. That is not a failure, so
                # follow the tables that exist instead of failing on their names.
                prefix = self.detect_prefix(package, request.prefix)
                if prefix != request.prefix:
                    self.log(
                        job_dir,
                        f"[prefix] Agent 实际使用的表名前缀为 {prefix}"
                        f"（请求为 {request.prefix}），后续按实际前缀继续",
                    )
                self.ensure_forward_pipeline(
                    job_id, job_dir, package, prefix, sqlite_path,
                )
                self.validate_package(package, prefix, job_id=job_id, log_to=job_dir)
                self.convert(
                    package, job_dir / "analysis.json", prefix, request,
                    job_id=job_id,
                )
                self.validate_package(
                    package, prefix, analysis_path=job_dir / "analysis.json",
                    job_id=job_id, log_to=job_dir,
                )
                trace_path = self.organize_result_package(
                    job_dir, package, prefix, sqlite_path, job_id=job_id,
                )
                context["sqlite_path"] = str(trace_path)
                (metadata_dir / "context.json").write_text(
                    json.dumps(context, ensure_ascii=False, indent=2) + "\n",
                )
                self.ensure_final_report(job_id, job_dir, job_dir, prefix)

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
                # Record the terminal status first: logging touches the disk, and a
                # full disk is exactly the kind of failure that lands here, so a
                # raise from log() must not leave the job stuck in "analyzing".
                self.store.update(
                    job_id, status="failed", progress=100,
                    message="分析失败", error=str(exc),
                )
                self.log_quietly(job_dir, traceback.format_exc())
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
            # The main path validates the converted package here; the retry used to
            # skip it, so a retry could publish an analysis.json the first run would
            # have rejected.
            self.validate_package(
                package, prefix, analysis_path=job_dir / "analysis.json",
                job_id=job_id, log_to=job_dir,
            )
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
                self.store.update(
                    job_id, status="failed", progress=100,
                    message="转换重试失败", error=str(exc),
                )
                self.log_quietly(job_dir, traceback.format_exc())

    # Inputs whose only job is to make the analysis faster, not to feed it
    # evidence. A typo in one of these must not cost the caller a whole run, so
    # it degrades to "not supplied". Every other input stays fatal on purpose:
    # analysing without the source tree or the config succeeds and produces a
    # worse answer, which is harder to notice than a failed job.
    OPTIONAL_INPUTS = ("torch_trace",)

    def resolve_inputs(
        self, request: JobCreate, job_dir: Path | None = None,
    ) -> dict[str, Path | None]:
        values = {
            "report": request.report_path,
            "config": request.config_path,
            "launch": request.launch_path,
            "source": request.source_path,
            "design": request.design_path,
            "torch_trace": request.torch_trace_path,
        }
        resolved: dict[str, Path | None] = {}
        for key, value in values.items():
            if not value:
                resolved[key] = None
                continue
            try:
                resolved[key] = self.settings.resolve_allowed(value, kind=key)
            except ValueError:
                if key not in self.OPTIONAL_INPUTS:
                    raise
                resolved[key] = None
                if job_dir is not None:
                    self.log(
                        job_dir,
                        f"[inputs] 忽略无法解析的可选输入 {key}={value}："
                        "本次分析按未提供该输入继续",
                    )
        return resolved

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
        nsys_bin = self.resolve_nsys(job_id, job_dir, report)
        self.run_process(
            job_id, job_dir,
            [nsys_bin, "export", "--type", "sqlite", "--output", str(sqlite_path), str(report)],
        )
        if not sqlite_path.exists():
            raise RuntimeError("nsys export completed without producing SQLite")
        return sqlite_path

    def resolve_nsys(self, job_id: str, job_dir: Path, report: Path) -> str:
        """The nsys that can export this report -- the installed one, or a fetched CLI.

        Reports carry the build that wrote them and `nsys export` only reads its own
        version or older, so a machine with an older nsys than the capture cannot
        export at all. Rather than failing, fetch the CLI-only package for that exact
        build into our own cache (~200 MB, no root, nothing installed system-wide) and
        use it just for this step. Any failure here falls back to the configured nsys,
        whose own error message is the honest one to show.
        """
        needed = self.report_tool_version(report)
        if needed is None:
            return self.settings.nsys_bin
        current = self.nsys_version(self.settings.nsys_bin)
        if current is not None and current >= needed[0]:
            return self.settings.nsys_bin
        label = ".".join(str(part) for part in needed[0])
        have = ".".join(str(part) for part in current) if current else "未检测到"
        self.log(
            job_dir,
            f"[nsys] 报告由 Nsight Systems {label} 生成，本机 nsys 为 {have}，"
            "尝试获取可解析该版本的 CLI",
        )
        try:
            fetched = self.fetch_nsys_cli(job_id, job_dir, needed[0], needed[1])
        except Exception as exc:
            self.log(job_dir, f"[nsys] 获取 Nsight Systems CLI 失败，仍用本机 nsys：{exc}")
            return self.settings.nsys_bin
        # Unpacking says nothing about whether the binary runs here: a wrong
        # architecture, or a missing shared library, only shows up on execution. Ask
        # it for its version, and fall back to the local nsys if it cannot answer --
        # otherwise the export fails with an exec error instead of the honest
        # "your nsys is too old".
        if self.nsys_version(str(fetched)) is None:
            self.log(
                job_dir,
                f"[nsys] 获取到的 {fetched} 无法在本机执行"
                f"（架构 {platform.machine()}），仍用本机 nsys",
            )
            return self.settings.nsys_bin
        self.log(job_dir, f"[nsys] 使用 {fetched}")
        return str(fetched)

    @staticmethod
    def report_tool_version(report: Path) -> tuple[tuple[int, ...], str] | None:
        """(version tuple, full build string) of the tool that wrote a .nsys-rep."""
        try:
            with report.open("rb") as handle:
                head = handle.read(256)
        except OSError:
            return None
        match = REPORT_VERSION_RE.search(head)
        if not match:
            return None
        numbers = tuple(int(match.group(index)) for index in range(1, 5))
        build = f"{'.'.join(str(n) for n in numbers)}-{match.group(5).decode()}"
        return numbers, build

    @staticmethod
    def nsys_version(binary: str) -> tuple[int, ...] | None:
        try:
            completed = subprocess.run(
                [binary, "--version"], text=True, capture_output=True, timeout=30,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        match = NSYS_VERSION_RE.search(completed.stdout + completed.stderr)
        if not match:
            return None
        return tuple(int(match.group(index)) for index in range(1, 5))

    @staticmethod
    def nsys_cache_dir() -> Path:
        """Where fetched CLIs live -- same cache root the launcher uses for node."""
        base = Path(os.getenv("XDG_CACHE_HOME", Path.home() / ".cache"))
        return base / "nsysscope" / "nsys"

    @classmethod
    def cached_nsys(cls, build: str) -> Path | None:
        root = cls.nsys_cache_dir() / build
        for pattern in ("**/target-linux-x64/nsys", "**/bin/nsys", "**/nsys"):
            for candidate in sorted(root.glob(pattern)):
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
        return None

    def fetch_nsys_cli(
        self, job_id: str, job_dir: Path, version: tuple[int, ...], build: str,
    ) -> Path:
        """Download and unpack the CLI-only Nsight Systems package for one build."""
        cached = self.cached_nsys(build)
        if cached is not None:
            self.log(job_dir, f"[nsys] 命中缓存 {cached}")
            return cached
        label = ".".join(str(part) for part in version)
        filename = self.nsys_package_filename(version, build)
        target = self.nsys_cache_dir() / build
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="nsys-cli-") as temporary:
            archive = Path(temporary) / filename
            url = f"{self.nsys_repo_url}/{filename}"
            self.log(job_dir, f"[nsys] 下载 {url}")
            self.download(
                url, archive, log_to=job_dir,
                on_progress=self.nsys_progress_reporter(job_id, job_dir, label),
            )
            if not self.is_cancelled(job_id):
                self.state(
                    job_id, job_dir, "exporting", 16,
                    f"正在解包 Nsight Systems {label} CLI"
                    f"（{archive.stat().st_size // 1048576}MB）",
                )
            # dpkg-deb keeps this to one call; ar + tar is the fallback for hosts
            # without dpkg (the package is a plain ar archive either way).
            if shutil.which("dpkg-deb"):
                extract = [shutil.which("dpkg-deb"), "-x", str(archive), str(target)]
                completed = subprocess.run(extract, text=True, capture_output=True)
                if completed.returncode != 0:
                    raise RuntimeError(f"dpkg-deb 解包失败：{completed.stderr.strip()[:500]}")
            else:
                self.extract_deb_without_dpkg(archive, target)
        binary = self.cached_nsys(build)
        if binary is None:
            raise RuntimeError(f"{filename} 解包后没有找到 nsys 可执行文件")
        return binary

    def nsys_progress_reporter(
        self, job_id: str, job_dir: Path, label: str,
    ) -> Callable[[int, int | None, float], None]:
        """Turn chunk progress into a status line with a bar, a rate and an ETA.

        This is the one step that can take ten minutes with nothing else happening, so
        it reports like a foreground task: the UI's status message carries the bar and
        the job log keeps one line per chunk as the durable record.
        """
        def report(written: int, total: int | None, rate: float) -> None:
            # A fetch runs for minutes; a cancel that lands mid-download must not be
            # overwritten by the next progress tick, which would put the job back into
            # "exporting" and make the UI look stuck instead of cancelled.
            if self.is_cancelled(job_id):
                return
            done = written / 1048576
            speed = rate / 1048576
            if total:
                share = written / total
                filled = int(share * 20)
                bar = "=" * filled + " " * (20 - filled)
                remaining = (total - written) / rate if rate > 0 else 0
                eta = (
                    f"{remaining / 60:.0f} 分钟" if remaining >= 60
                    else f"{remaining:.0f} 秒"
                )
                detail = (
                    f"[{bar}] {share * 100:.0f}%  {done:.0f}/{total / 1048576:.0f}MB  "
                    f"{speed:.2f}MB/s  剩余约 {eta}"
                )
                # 10 -> 16% spans the fetch, keeping the export itself above it.
                progress = 10 + int(share * 6)
            else:
                detail = f"{done:.0f}MB  {speed:.2f}MB/s"
                progress = 10
            self.state(
                job_id, job_dir, "exporting", progress,
                f"正在获取 Nsight Systems {label} CLI {detail}",
            )
        return report

    @staticmethod
    def extract_deb_without_dpkg(archive: Path, target: Path) -> None:
        work = archive.parent
        for command in (
            [shutil.which("ar") or "ar", "x", str(archive)],
        ):
            completed = subprocess.run(command, text=True, capture_output=True, cwd=work)
            if completed.returncode != 0:
                raise RuntimeError(f"ar 解包失败：{completed.stderr.strip()[:500]}")
        data = next((path for path in work.glob("data.tar*")), None)
        if data is None:
            raise RuntimeError("deb 包内没有 data.tar*")
        with tarfile.open(data) as handle:
            handle.extractall(target)

    def nsys_package_filename(self, version: tuple[int, ...], build: str) -> str:
        """Look the exact build up in the repo's Packages index.

        Matching on the full build string rather than composing a filename: the
        filename embeds an internal build id (…-3860507.deb) that cannot be derived
        from the version, and only the index knows it.
        """
        index = ""
        for base in NSYS_REPO_URLS:
            try:
                index = self.download_text(f"{base}/Packages")
            except Exception:
                continue
            # Remember which mirror answered; the package comes from the same one.
            self.nsys_repo_url = base
            break
        if not index:
            raise RuntimeError(
                "无法读取 Nsight Systems 仓库索引（可用 NSYSSCOPE_NSYS_REPO 指向内网镜像）"
            )
        entries: dict[str, str] = {}
        for block in index.split("\n\n"):
            # `nsight-systems-cli-*` only: the same index also carries the full
            # package (GUI plus target binaries, several GB) and export needs none
            # of it.
            if "Package: nsight-systems-cli" not in block:
                continue
            fields = dict(re.findall(r"^(\w+): (.+)$", block, re.M))
            recorded = fields.get("Version", "")
            filename = fields.get("Filename", "")
            if recorded and filename:
                entries[recorded] = filename.rsplit("/", 1)[-1]
        if build in entries:
            return entries[build]
        # No exact build: any newer CLI can read this report, so take the lowest one
        # that is new enough rather than failing.
        newer = sorted(
            (parsed, name)
            for recorded, name in entries.items()
            if (parsed := tuple(
                int(part) for part in recorded.split("-", 1)[0].split(".")
                if part.isdigit()
            )) >= version
        )
        if not newer:
            raise RuntimeError(f"仓库索引里没有 {build} 或更新的 CLI 包")
        return newer[0][1]

    def download_candidates(self) -> list[str | None]:
        """Ways out to the public internet, best first, same order as the launcher's
        proxy_candidates(): direct, the caller's own setting, the environment's, then
        the two office proxies. A proxy that already carried a transfer in this
        process is tried first.
        """
        ordered: list[str | None] = [
            None,
            os.getenv("NSYSSCOPE_DOWNLOAD_PROXY"),
            os.getenv("https_proxy"),
            os.getenv("HTTPS_PROXY"),
            *DOWNLOAD_PROXY_CANDIDATES,
        ]
        if self._download_proxy is not ...:
            ordered.insert(0, self._download_proxy)  # type: ignore[arg-type]
        seen: list[str | None] = []
        for candidate in ordered:
            if candidate == "" or candidate in seen:
                continue
            seen.append(candidate)
        return seen

    @staticmethod
    def _opener(proxy: str | None) -> urllib.request.OpenerDirector:
        return urllib.request.build_opener(
            urllib.request.ProxyHandler(
                {} if proxy is None else {"http": proxy, "https": proxy}
            )
        )

    def fastest_download_path(
        self, url: str, probe_bytes: int = 4 * 1024 * 1024, timeout: int = 30,
        log_to: Path | None = None,
    ) -> str | None:
        """Time a small ranged read on every candidate and keep the quickest.

        Order alone picks badly here: the first *working* path was one of the office
        proxies at ~0.24 MB/s while the other does ~0.95 MB/s, and settling on the
        former turned a 4-minute download into 13. One 4 MB probe per candidate costs
        seconds and is paid once per process.
        """
        best: tuple[float, str | None] | None = None
        for candidate in self.download_candidates():
            request = urllib.request.Request(
                url, headers={"Range": f"bytes=0-{probe_bytes - 1}"},
            )
            started = time.monotonic()
            try:
                with self._opener(candidate).open(request, timeout=timeout) as response:
                    read = len(response.read())
            except Exception:
                continue
            rate = read / max(time.monotonic() - started, 1e-6)
            if best is None or rate > best[0]:
                best = (rate, candidate)
        if best is None:
            return None
        self._download_proxy = best[1]
        if log_to is not None:
            self.log(
                log_to,
                f"[download] 选用通路 {best[1] or '直连'}（实测 {best[0] / 1048576:.2f} MB/s）",
            )
        return best[1]

    def download_text(self, url: str) -> str:
        errors = []
        for proxy in self.download_candidates():
            try:
                with self._opener(proxy).open(url, timeout=20) as response:
                    payload = response.read().decode("utf-8", "replace")
            except Exception as exc:
                errors.append(f"{proxy or '直连'}: {exc}")
                continue
            self._download_proxy = proxy
            return payload
        raise RuntimeError(
            "无法访问下载源（可设置 NSYSSCOPE_DOWNLOAD_PROXY，"
            f"或用 NSYSSCOPE_NSYS_REPO 指向内网镜像）：{'; '.join(errors)}"
        )

    def download(
        self, url: str, destination: Path, stall_seconds: int = 30,
        chunk_bytes: int = 16 * 1024 * 1024, log_to: Path | None = None,
        on_progress: Callable[[int, int | None, float], None] | None = None,
    ) -> None:
        """Fetch a large file in ranged chunks, switching proxies on failure.

        Measured on the office proxies, two things make a single long-lived GET the
        wrong shape for a 200 MB package:

        - sustained throughput is a fraction of what a short read suggests (1.7 MB/s
          on the first 4 MB, ~0.35 MB/s averaged over the whole package), and
          client-side parallelism does not lift it -- 1, 4 and 8 concurrent ranges all
          measured ~0.95 MB/s, so the cap is the proxy's, not the connection's;
        - one proxy buffers a whole body before it forwards anything, so a plain GET
          looks dead for minutes while a ranged GET answers in seconds.

        So: sequential `Range` requests, each a fresh connection, appended to the file.
        A failed chunk is retried on the next proxy and resumes from what is already
        on disk rather than starting over. Reachability is never probed separately --
        the transfer itself decides which proxy is usable.
        """
        errors: list[str] = []
        written = 0
        total: int | None = None
        # Pick by measured throughput before committing to a 200 MB transfer; chunks
        # still fail over on their own if the chosen path dies mid-download.
        self.fastest_download_path(url, log_to=log_to)
        started = time.monotonic()
        with destination.open("wb") as sink:
            while total is None or written < total:
                chunk, total, complete = self.download_range(
                    url, written, chunk_bytes, stall_seconds, errors, log_to,
                )
                sink.write(chunk)
                written += len(chunk)
                if on_progress is not None:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    on_progress(written, total, written / elapsed)
                if complete or not chunk:
                    return
        return

    def download_range(
        self, url: str, offset: int, length: int, stall_seconds: int,
        errors: list[str], log_to: Path | None,
    ) -> tuple[bytes, int | None, bool]:
        """One chunk, tried across every candidate path.

        Returns the bytes, the file's total size when the server reported it, and
        whether the response was the whole file (a server that ignores `Range`).
        """
        for proxy in self.download_candidates():
            try:
                request = urllib.request.Request(
                    url, headers={"Range": f"bytes={offset}-{offset + length - 1}"},
                )
                with self._opener(proxy).open(request, timeout=stall_seconds) as response:
                    payload = response.read()
                    content_range = response.headers.get("Content-Range", "")
                    status = response.status
            except Exception as exc:
                errors.append(f"{proxy or '直连'}: {exc}")
                if log_to is not None:
                    self.log(
                        log_to,
                        f"[download] {errors[-1]}（偏移 {offset // 1048576}MB），改用下一个通路",
                    )
                continue
            self._download_proxy = proxy
            total = None
            if "/" in content_range:
                tail = content_range.rsplit("/", 1)[1]
                total = int(tail) if tail.isdigit() else None
            # 200 instead of 206: the server ignored Range and sent everything.
            return payload, total, status == 200 and offset == 0
        raise RuntimeError(f"下载失败：{'; '.join(errors[-4:])}")

    def build_dispatch_cache(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None],
    ) -> Path | None:
        """Pre-resolve kernel -> Python dispatch sites from a torch profiler trace.

        Optional value-add pass: it turns the trace's Python stacks into a
        per-kernel-symbol lookup table so the analysis Skill does not have to
        search the source tree once per kernel. Any failure here is logged and
        swallowed -- the analysis must still run exactly as it did before, just
        without the shortcut.
        """
        trace = paths.get("torch_trace")
        if trace is None:
            return None
        scripts = self.settings.call_tree_skill_dir / "scripts"
        call_tree = scripts / "layer_call_tree.py"
        if not call_tree.is_file():
            self.log(
                job_dir,
                "[dispatch] 未找到 reconstruct-profiler-call-tree Skill，跳过调用栈预解析："
                f"{call_tree}",
            )
            return None
        self.state(job_id, job_dir, "analyzing", 24, "正在从 torch profiler trace 预解析 kernel 调用栈")
        out_dir = job_dir / "dispatch_sites"
        command = [
            sys.executable, str(call_tree),
            "--trace", str(trace),
            # A negative index asks the script to pick a steady-state pass.
            "--fwd-pass", "-1",
            "--output-dir", str(out_dir),
        ]
        if paths.get("config"):
            command += ["--config", str(paths["config"])]
        self.run_process(job_id, job_dir, command)
        cache = out_dir / "mappings" / "dispatch_site_cache.json"
        if not cache.is_file():
            raise RuntimeError(f"call-tree pass produced no dispatch cache at {cache}")
        source = paths.get("source")
        if source is not None:
            snippets = scripts / "resolve_dispatch_snippets.py"
            if snippets.is_file():
                self.run_process(job_id, job_dir, [
                    sys.executable, str(snippets),
                    "--cache", str(cache),
                    "--source-root", str(source if source.is_dir() else source.parent),
                ])
                enriched = cache.with_name("dispatch_site_cache_with_snippets.json")
                if enriched.is_file():
                    cache = enriched
        self.log(job_dir, f"[dispatch] 调用栈查表已生成：{cache}")
        return cache

    def try_build_dispatch_cache(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None],
    ) -> Path | None:
        try:
            return self.build_dispatch_cache(job_id, job_dir, request, paths)
        except Exception:
            self.log(
                job_dir,
                "[dispatch] 调用栈预解析失败，本次分析回退为由 Agent 自行检索源码：\n"
                + traceback.format_exc(),
            )
            return None

    def run_agent(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
        dispatch_cache: Path | None = None,
    ) -> None:
        if request.agent_provider == "comate":
            self.run_comate(job_id, job_dir, request, paths, sqlite_path, dispatch_cache)
        else:
            self.run_codex(job_id, job_dir, request, paths, sqlite_path, dispatch_cache)

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

    def log_skill_source(self, job_dir: Path, provenance: dict[str, str]) -> None:
        """Say which copy of the skill is about to run, and warn if it is not ours.

        skill_manager resolves ~/.codex/skills before bundled/, so a copy left
        behind by an older install silently shadows the repository's skill and
        the job produces the old behaviour with no sign of it. metadata/skill.json
        has always recorded this; the job log is where someone actually looks.
        """
        source = provenance.get("source", "?")
        line = (
            f"[skill] 使用 {provenance.get('name')} ({source})："
            f"{provenance.get('path')} sha256={provenance.get('sha256', '')[:12]}"
        )
        if source != "bundled":
            line += (
                "\n[skill] 注意：这不是仓库自带的 skill，行为可能与当前代码不一致。"
                "要强制使用自带版本，请 unset NSYSSCOPE_SKILL_DIR、"
                "执行 nsysscope skill reset，或移走 ~/.codex/skills 下的同名目录。"
            )
        self.log(job_dir, line)

    def run_codex(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
        dispatch_cache: Path | None = None,
    ) -> None:
        if not self.settings.codex_enabled:
            raise RuntimeError(
                "Codex analyzer is disabled. Set NSYSSCOPE_CODEX_ENABLED=true on an isolated runner."
            )
        self.state(job_id, job_dir, "analyzing", 30, "Codex Skill Agent 正在分析模型与时间线")
        prompt = self.build_prompt(job_dir, request, paths, sqlite_path, dispatch_cache)
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
            stall_timeout_seconds=self.settings.agent_stall_timeout_seconds,
            heartbeat_message="Codex Agent 仍在运行",
        )

    def run_comate(
        self, job_id: str, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
        dispatch_cache: Path | None = None,
    ) -> None:
        status = self._comate_status()
        if not status["ready"]:
            raise RuntimeError(status["message"])
        self.state(job_id, job_dir, "analyzing", 30, "Comate Skill Agent 正在分析模型与时间线")
        prompt = self.build_prompt(job_dir, request, paths, sqlite_path, dispatch_cache)
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
        output = self.run_process(
            job_id, job_dir, command, redacted_values={prompt},
            environment=self.comate_environment(),
            output_formatter=self.format_comate_output,
            heartbeat_seconds=self.settings.agent_heartbeat_seconds,
            stall_timeout_seconds=self.settings.agent_stall_timeout_seconds,
            session_store=self.settings.comate_store_dir,
            heartbeat_message="Comate Agent 仍在运行",
        )
        # zulu can crash at parse time and still exit 0 (see node_runtime_problem),
        # so a zero return code is not evidence that the agent ran. Fail here with
        # the real cause instead of later, on the missing tables. Matched against the
        # CLI's own output rather than the job log, and only when nothing was
        # produced: an agent that wrote its tables plainly did not fail to start, and
        # its transcript can quote any string it likes.
        if not self.agent_outputs_complete(job_dir):
            problem = self.node_runtime_problem(output)
            if problem:
                raise RuntimeError(problem)

    def stage_comate_skill(self, job_dir: Path) -> Path:
        source = self.settings.skill_dir
        if not (source / "SKILL.md").exists():
            raise RuntimeError(f"analysis skill is missing: {source / 'SKILL.md'}")
        target = job_dir / ".comate" / "skills" / "sglang-nsys-static-analysis"
        target.parent.mkdir(parents=True, exist_ok=True)
        # Re-running into an existing job dir must replace the staged skill, not
        # fail or merge into it.
        if target.exists():
            shutil.rmtree(target)
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

    @staticmethod
    def dispatch_cache_prompt(dispatch_cache: Path | None) -> str:
        """Prompt section describing the pre-resolved dispatch-site cache.

        Empty when no torch profiler trace was supplied, so the prompt stays
        byte-identical to the previous version for jobs that do not use it.
        """
        if dispatch_cache is None:
            return ""
        # The same pre-pass also derived the layer segmentation the analysis has to
        # establish anyway. Listing those artifacts lets the agent verify a prior
        # instead of rediscovering the anchor from scratch; they come from a
        # *different* capture than the nsys trace, so they are priors, not facts.
        root = dispatch_cache.parent.parent
        priors = "".join(
            f"- {label}: {path}\n"
            for label, path in (
                ("layer segmentation prior (torch trace)", root / "final_report.md"),
                ("per-layer timing prior (torch trace)", root / "rankings" / "slowest_layers.csv"),
                ("kernel -> layer/submodule prior (torch trace)",
                 root / "mappings" / "kernel_to_layer.csv"),
            )
            if path.is_file()
        )
        prior_note = f"""
{priors}These priors report the anchor kernel, layer count and per-layer timings the
pre-pass derived from the torch profiler trace. Use them as a starting hypothesis
to confirm against the nsys trace -- they come from a separate capture, so the
anchor, layer count and repeating unit must still be re-established from the nsys
SQLite before you rely on them, and nsys stays the sole timing authority. Say so
explicitly if the two captures disagree.
""" if priors else ""
        return f"""- pre-resolved kernel dispatch sites: {dispatch_cache}
{prior_note}
A dispatch-site cache was built from this task's torch profiler trace. Consult it
per kernel symbol before searching the source tree, and record the cache as the
mapping evidence when you use it. Trust levels differ per entry:
- `dispatch_code_snippet` present: use it as the dispatching statement.
- only `dispatch_function_body` present: the launching statement was not
  identified; treat the body as the search window and pick the statement yourself.
- `line_drift: true`: the local checkout disagrees with the traced build for that
  function; the function name is authoritative, the line number is not.
Kernels absent from the cache still require the normal source-tree search. The
cache never overrides the trace as timing authority.
"""

    def build_prompt(
        self, job_dir: Path, request: JobCreate,
        paths: dict[str, Path | None], sqlite_path: Path,
        dispatch_cache: Path | None = None,
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
{self.dispatch_cache_prompt(dispatch_cache)}
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
   peak and source; "new model" is not a reason to leave MFU blank. A source
   commit that does not match the captured build only downgrades source-derived
   defaults and branch claims: keep the `file:line` call chain, the dispatch code
   snippet, the Chinese functional description and the GEMM shape/MFU/MBU, marked
   unverified where they depend on the source.
8. Never infer CPU delay from zero-kernel GPU idle. Require CUDA Runtime launch
   timestamp evidence or label the interval GPU idle/queue/dependency gap.
9. Run `scripts/validate_analysis_package.py` with the taxonomy and finish only when every required
   invariant, including the requested scope, passes. The manifest boundary
   evidence must explicitly show how the selected unit satisfies the acceptance
   criteria.
10. Never edit input reports, config, launch files, design notes, or model source.
11. Sample at least three complete steady-state occurrences of the unit, never the
   capture's first forward, and pick a device whose layer mix reproduces the
   declared unit. Align the window to the layer-start kernel so positions of one
   variant hold the same operators, and never attribute a pipeline handoff wait
   (`SendRecv`) to a layer.
12. Finish by writing `final_report.md` in {job_dir}: run
   `scripts/build_final_report.py {job_dir} --prefix {request.prefix}` for the
   tables, then replace every `<!-- TODO -->` marker with your own conclusions,
   optimisation paths and analysis reasoning. It is the deliverable a human reads
   instead of the tables, so keep it short, factual and number-backed.
"""

    @staticmethod
    def agent_outputs_complete(job_dir: Path) -> bool:
        """Whether the agent's six mandatory tables already exist somewhere below.

        Asking the question during the wait is what lets a finished-but-parked CLI be
        cut short instead of held to its own timeout. Any prefix counts, and the
        directory may be any depth: unlike `find_package`, which pins the requested
        prefix once the process has exited, this only needs to know that the work is
        done. Filesystem errors mean "cannot tell", i.e. keep waiting.
        """
        try:
            markers = list(job_dir.rglob("*_stage_table.csv"))
        except OSError:
            return False
        for marker in markers:
            prefix = marker.name.removesuffix("_stage_table.csv")
            try:
                if all(
                    (marker.parent / f"{prefix}{suffix}").exists()
                    for suffix in AGENT_CSV_SUFFIXES
                ):
                    return True
            except OSError:
                continue
        return False

    @staticmethod
    def newest_artifact(job_dir: Path) -> tuple[float, str | None]:
        """Newest file the agent produced under the job directory, and its mtime.

        The name is returned so the heartbeat can say *what* was produced last —
        that is the only progress signal available while the agent is silent.
        Only the job log itself and the one-shot skill snapshot are excluded: the
        heartbeat writes the log, so counting it would make a stalled agent look
        busy forever. Everything else counts, including the agent's own scratch
        dumps under `logs/`, which are often the only evidence of progress during a
        long segmentation pass.
        """
        newest = 0.0
        name: str | None = None
        # Both the structured and the legacy job log locations, see job_log_path.
        skipped = {job_dir / "logs" / "job.log", job_dir / "job.log"}
        for path in job_dir.rglob("*"):
            relative = path.relative_to(job_dir)
            if relative.parts and relative.parts[0] == ".comate":
                continue
            if path in skipped:
                continue
            try:
                if path.is_file():
                    mtime = path.stat().st_mtime
                    if mtime > newest:
                        newest, name = mtime, relative.as_posix()
            except OSError:
                continue
        return newest, name

    @staticmethod
    def process_group_cpu_seconds(pgid: int) -> float:
        """CPU seconds burnt by every process in the agent's process group.

        The agent is started with `start_new_session`, so its pid is the group id and
        the whole engine/child tree is covered. Reading /proc keeps this dependency
        free; an unreadable or vanished process simply contributes nothing.
        """
        ticks = os.sysconf("SC_CLK_TCK") or 100
        total = 0.0
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                stat_line = (entry / "stat").read_text()
            except OSError:
                continue
            head, _, tail = stat_line.rpartition(")")
            fields = tail.split()
            if len(fields) < 13:
                continue
            try:
                # tail starts at the state field, so pgrp is index 2 and utime/stime
                # are 11 and 12.
                if int(fields[2]) != pgid:
                    continue
                total += (int(fields[11]) + int(fields[12])) / ticks
            except ValueError:
                continue
        return total

    @staticmethod
    def find_agent_session(store_dir: Path, job_dir: Path, since: float) -> Path | None:
        """The Comate conversation file belonging to this job, if it exists yet.

        The engine writes one `chat_session_<uuid>` file per conversation and stores
        the `--cwd` we passed in its `workspaceDirectory` field, so the job dir
        identifies our own conversation without parsing engine logs. Only files
        touched since the process started are considered, and the path is confirmed
        by parsing the field rather than by a substring hit, so an unrelated
        conversation that merely mentions the job dir in its transcript cannot be
        mistaken for ours. Returns None when the store is missing, unreadable or the
        conversation has not been persisted yet.
        """
        needle = json.dumps(str(job_dir)).encode()
        # Matching the field name together with the value keeps the scan cheap: an
        # unrelated conversation that only quotes the path in its transcript fails the
        # test without paying for a JSON parse of a multi-megabyte file.
        keyed = [
            b'"workspaceDirectory":' + prefix + needle for prefix in (b"", b" ")
        ]
        newest: Path | None = None
        newest_mtime = 0.0
        try:
            candidates = sorted(store_dir.glob("chat_session_*"))
        except OSError:
            return None
        for path in candidates:
            try:
                mtime = path.stat().st_mtime
                if mtime < since or mtime <= newest_mtime:
                    continue
                raw = path.read_bytes()
                if not any(pattern in raw for pattern in keyed):
                    continue
                if json.loads(raw).get("workspaceDirectory") != str(job_dir):
                    continue
            except (OSError, ValueError):
                continue
            newest, newest_mtime = path, mtime
        return newest

    @staticmethod
    def session_activity(path: Path) -> tuple[float, int]:
        """Modification time and size of a conversation file, zeros if it is gone.

        Either value growing means the engine appended a message or a tool result,
        which is direct evidence that the agent is still working.
        """
        try:
            stat = path.stat()
        except OSError:
            return 0.0, 0
        return stat.st_mtime, stat.st_size

    def run_process(
        self, job_id: str, job_dir: Path, command: list[str], stdin: str | None = None,
        redacted_values: set[str] | None = None,
        environment: dict[str, str] | None = None,
        output_formatter: Callable[[str], str | None] | None = None,
        heartbeat_seconds: int = 0,
        heartbeat_message: str = "Agent 仍在运行",
        stall_timeout_seconds: int = 0,
        session_store: Path | None = None,
    ) -> str:
        """Run a child process, streaming its output into the job log.

        Returns the tail of the child's own raw stdout+stderr. Callers that need to
        diagnose *how* a process failed must use that instead of the job log: the
        log also holds the command line and every other step's output, so matching
        crash signatures against it produces false positives.
        """
        hidden = redacted_values or set()
        displayed = ["<prompt>" if item in hidden else item for item in command]
        self.log(job_dir, f"$ {shlex.join(displayed)}")
        # Conversation files are only created after launch, so remember when that was
        # (with a little slack for filesystem timestamp granularity).
        launched_at = time.time() - 5
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
        stalled = threading.Event()
        # Set when the heartbeat stops waiting because the agent's outputs are already
        # complete. The process is killed to get there, so its exit code must not be
        # reported as a failure.
        finished_early = threading.Event()
        # Raw child output, kept for callers that must recognise a crash signature.
        # Bounded because an agent turn can emit megabytes in a single line.
        captured: list[str] = []
        last_output = [elapsed_seconds()]
        heartbeat_thread: threading.Thread | None = None
        if heartbeat_seconds > 0:
            def heartbeat_loop() -> None:
                seen, artifact = self.newest_artifact(job_dir)
                cpu = self.process_group_cpu_seconds(process.pid)
                cpu_threshold = max(
                    CPU_PROGRESS_SECONDS, heartbeat_seconds * CPU_PROGRESS_FRACTION,
                )
                progress_at = elapsed_seconds()
                session: Path | None = None
                session_state = (0.0, 0)
                while not heartbeat_stop.wait(heartbeat_seconds):
                    if process.poll() is not None:
                        return
                    current, name = self.newest_artifact(job_dir)
                    if current > seen:
                        seen, artifact = current, name
                        progress_at = elapsed_seconds()
                    # An agent can spend a long turn reading and reasoning without
                    # writing anything, and `--display task-json` keeps stdout silent
                    # until the end, so burnt CPU is the third progress signal. Only a
                    # process that produces nothing AND burns no CPU is really stuck.
                    busy = self.process_group_cpu_seconds(process.pid)
                    if busy - cpu >= cpu_threshold:
                        cpu = busy
                        progress_at = elapsed_seconds()
                    # The strongest signal: the agent's own conversation file. It grows
                    # on every message and tool result, so while it advances the agent
                    # is demonstrably still working, whatever the job dir looks like.
                    if session_store is not None:
                        if session is None:
                            session = self.find_agent_session(
                                session_store, job_dir, launched_at,
                            )
                            if session is not None:
                                session_state = self.session_activity(session)
                                progress_at = elapsed_seconds()
                                self.log(
                                    job_dir,
                                    f"[heartbeat] 已定位 Agent 会话 {session.name}，"
                                    "后续以会话更新判断存活",
                                )
                        else:
                            state = self.session_activity(session)
                            if state > session_state:
                                session_state = state
                                progress_at = elapsed_seconds()
                    idle = elapsed_seconds() - max(progress_at, last_output[0])
                    # Age of the conversation file, in wall-clock terms. Once the file
                    # exists this is the authoritative liveness signal, so it gets to
                    # end the wait on its own -- the other signals can only extend it,
                    # and CPU noise from an idle event loop was enough to keep a
                    # finished agent parked until the CLI's own two-hour timeout.
                    session_idle = (
                        max(0.0, time.time() - session_state[0])
                        if session is not None and session_state[0] else None
                    )
                    produced = f"最近产出 {artifact}" if artifact else "尚未产出任何文件"
                    if session_store is None:
                        alive = ""
                    elif session is None:
                        alive = "，未找到 Agent 会话文件"
                    else:
                        alive = f"，会话 {(session_idle or 0.0) / 60:.0f} 分钟前更新"
                    self.log(
                        job_dir,
                        f"[heartbeat] {heartbeat_message}（{produced}，"
                        f"距上次进展 {idle / 60:.0f} 分钟，累计 CPU {busy / 60:.1f} 分钟"
                        f"{alive}）",
                    )
                    # The agent wrote everything the pipeline needs and then went quiet:
                    # the model turn is over and only the CLI process is still parked.
                    # Waiting for it buys nothing, so stop waiting and let the caller
                    # proceed with the package that is already on disk. Checked before
                    # the stall branch on purpose: with the outputs present this is a
                    # success, and it must win when both thresholds are crossed.
                    if (
                        stall_timeout_seconds
                        and session_idle is not None
                        and session_idle > min(
                            EARLY_FINISH_IDLE_SECONDS, stall_timeout_seconds,
                        )
                        and self.agent_outputs_complete(job_dir)
                    ):
                        self.log(
                            job_dir,
                            f"[heartbeat] Agent 产物已齐全且会话静默 "
                            f"{session_idle / 60:.0f} 分钟，视为已完成，停止等待进程退出",
                        )
                        finished_early.set()
                        self._kill_process_group(process)
                        return
                    stalled_for = max(idle, session_idle or 0.0)
                    if stall_timeout_seconds and stalled_for > stall_timeout_seconds:
                        stalled.set()
                        self.log(
                            job_dir,
                            f"[stalled] Agent {stalled_for / 60:.0f} 分钟没有输出、"
                            f"没有新产物、没有会话更新（{produced}），"
                            "判定为停滞并终止（模型请求丢失时进程会活着但无事可做）",
                        )
                        self._kill_process_group(process)
                        return

            def emit_heartbeat() -> None:
                # The loop reads /proc, the job dir and the agent's session store,
                # all of which can raise at any moment. An unhandled raise here would
                # kill the daemon thread silently and disable stall detection for the
                # rest of the run -- the exact failure this thread exists to catch --
                # so report it and let the process fall back to its own timeout.
                try:
                    heartbeat_loop()
                except Exception:
                    self.log_quietly(
                        job_dir,
                        "[heartbeat] 心跳线程异常退出，本次运行不再做停滞判定：\n"
                        + traceback.format_exc(),
                    )

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
                last_output[0] = elapsed_seconds()
                captured.append(line)
                if len(captured) > RAW_OUTPUT_TAIL_LINES:
                    del captured[:-RAW_OUTPUT_TAIL_LINES]
                if line.strip():
                    rendered = output_formatter(line) if output_formatter else line
                    if rendered:
                        self.log(job_dir, rendered)
            code = process.wait()
            if stalled.is_set():
                raise RuntimeError(
                    f"agent 停滞超过 {stall_timeout_seconds // 60} 分钟，已终止：{command[0]}"
                )
            # A process we killed ourselves reports its signal as a negative code;
            # that one is expected. Any other nonzero code is a real failure even
            # when the outputs looked complete, so it must still be raised.
            if code and not (finished_early.is_set() and code < 0):
                raise RuntimeError(f"process exited with code {code}: {command[0]}")
            return "".join(captured)
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
        if tomllib is not None:
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
        node_problem = self.node_version_problem()
        if node_problem:
            return {"enabled": True, "ready": False, "message": node_problem}
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
        node_problem = self.node_runtime_problem(output)
        if node_problem:
            return {"enabled": True, "ready": False, "message": node_problem}
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

    def node_version_problem(self) -> str | None:
        """Reject a Node runtime the Zulu CLI cannot run on, before any job starts.

        This is the primary gate: checked up front, independent of how a given Node
        version happens to fail. `node_runtime_problem` stays as a backstop for the
        case where node itself is new enough but zulu still dies at load time.

        An unreadable or absent `node --version` is not treated as a failure -- a CLI
        may ship its own runtime, and a genuinely missing node fails loudly on its
        own -- so only a version we can read and know is too old blocks a job.
        """
        try:
            probe = subprocess.run(
                ["node", "--version"], text=True, capture_output=True, timeout=5,
                env=self.comate_environment(),
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        version = (probe.stdout or probe.stderr).strip()
        match = re.match(r"v?(\d+)\.", version)
        if not match:
            return None
        major = int(match.group(1))
        if major >= MIN_NODE_MAJOR:
            return None
        return (
            f"node 版本过低：{version}，Comate Zulu CLI 需要 Node {MIN_NODE_MAJOR} 及以上"
            "（低版本会在加载时以 Error: Not supported 崩溃且仍返回退出码 0）。"
            "请升级 node，或把新版 node 放到 PATH 前面后重试"
        )

    def node_runtime_problem(self, output: str) -> str | None:
        """Explain a zulu crash caused by the machine's `node` being too old.

        The Zulu CLI is a modern-JS bundle. On Node <= 12 it dies at parse time with
        `Error: Not supported` plus `UnhandledPromiseRejectionWarning`, and -- because
        unhandled rejections were only warnings back then -- the process still exits
        0. Every downstream step therefore looks like "the agent ran and produced
        nothing", which surfaced as the misleading "did not produce the six agent
        tables". Name the real cause instead.

        `output` must be the CLI's own stdout+stderr, not the job log: the log holds
        the command line (whose path contains "zulu") and the agent's transcript, so
        both signatures below would match text the agent merely echoed.
        """
        crashed = "Error: Not supported" in output or (
            "UnhandledPromiseRejectionWarning" in output and "zulu" in output
        )
        if not crashed:
            return None
        version = ""
        try:
            probe = subprocess.run(
                ["node", "--version"], text=True, capture_output=True, timeout=5,
                env=self.comate_environment(),
            )
            version = (probe.stdout or probe.stderr).strip()
        except (OSError, subprocess.TimeoutExpired):
            version = "未找到 node"
        return (
            f"Comate Zulu CLI 启动即崩溃（Error: Not supported），当前 node 版本：{version}。"
            f"Zulu 需要 Node {MIN_NODE_MAJOR} 及以上，"
            "请升级 node 或把新版 node 放到 PATH 前面后重试"
        )

    def build_forward_pipeline(
        self, job_id: str, job_dir: Path, package: Path, prefix: str,
        sqlite_path: Path, metadata_root: Path | None = None,
    ) -> None:
        """Generate the optional forward-pipeline table, the package's seventh table.

        It is the only place the package relates the measured unit to a whole
        forward step, so it is worth having whenever the capture supports it -- but
        some captures genuinely cannot produce it (a single forward step, no usable
        step marker, a schema quirk the builder does not yet handle). Log and
        continue instead of failing the whole job: the other six tables remain a
        complete, valid package on their own, and the frontend already renders
        without this module when it is absent (see app/page.js's optional-chained
        forwardPipeline check).
        """
        script = self.settings.skill_dir / "scripts" / "build_forward_pipeline_table.py"
        if not script.exists():
            self.log(job_dir, f"[forward-pipeline] 跳过：缺少生成脚本 {script}")
            return
        if not sqlite_path.exists():
            self.log(
                job_dir,
                f"[forward-pipeline] 跳过：forward 链路耗时表需要 trace，但 {sqlite_path} 不存在",
            )
            return
        taxonomy = None
        metadata_dir = metadata_root or (job_dir / "metadata")
        metadata_dir.mkdir(parents=True, exist_ok=True)
        for candidate in (
            metadata_dir / f"{prefix}_architecture_taxonomy.json",
            package / f"{prefix}_architecture_taxonomy.json",
        ):
            if candidate.exists():
                taxonomy = candidate
                break
        command = [
            sys.executable, str(script),
            "--sqlite", str(sqlite_path),
            "--output-dir", str(package),
            "--prefix", prefix,
            "--manifest-out",
            str(metadata_dir / f"{prefix}_forward_pipeline.json"),
        ]
        if taxonomy:
            command += ["--taxonomy", str(taxonomy)]
        device = self.analysed_device(metadata_dir, package, prefix)
        if device is not None:
            # The other tables were measured on one rank; letting this script rank the
            # devices again would put a different rank's forward in the same package.
            command += ["--device", str(device)]
        chunk = self.chunked_prefill_size(metadata_dir)
        if chunk:
            command += ["--chunk-size", str(chunk)]
        # config.json + launch command declare the step's layer counts and whether a
        # draft forward runs at all, so the trace only has to reproduce them. Without
        # this the builder has to guess both from the timeline's shape.
        command += self.declaration_args(metadata_dir)
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[:1500]
            self.log(job_dir, f"[forward-pipeline] 生成 forward 链路耗时表失败，跳过（不影响任务其余产出）：{detail}")
            return
        if completed.stdout:
            self.log(job_dir, f"[forward-pipeline] {completed.stdout.strip()}")
        table = package / f"{prefix}{FORWARD_PIPELINE_SUFFIX}"
        if not table.is_file():
            self.log(job_dir, f"[forward-pipeline] 跳过：生成脚本未产出 {table}")

    def ensure_forward_pipeline(
        self, job_id: str, job_dir: Path, package: Path, prefix: str,
        sqlite_path: Path | None, metadata_root: Path | None = None,
    ) -> None:
        """Regenerate the optional seventh table from the trace when possible.

        An imported package may already ship the table without shipping a trace, and
        a package produced before the table existed can still be completed as long as
        the trace is available. When neither holds, the package simply ships without
        it -- this table is a bonus view, not a gate on task success.
        """
        table = package / f"{prefix}{FORWARD_PIPELINE_SUFFIX}"
        if table.is_file():
            return
        if sqlite_path is not None and sqlite_path.is_file():
            self.build_forward_pipeline(
                job_id, job_dir, package, prefix, sqlite_path, metadata_root,
            )
        if not table.is_file():
            self.log(
                job_dir,
                f"[forward-pipeline] 跳过：包内没有 {table.name}，也没有可用的 trace 来生成它",
            )

    def ensure_final_report(
        self, job_id: str, job_dir: Path, package_root: Path, prefix: str,
    ) -> None:
        """Make sure the package ships the human-readable `final_report.md`.

        The agent is asked to write it (SKILL.md step 10) because the conclusions and
        optimisation paths are judgement, not arithmetic. When it did not -- an
        imported package, or an agent that stopped after the tables -- generate the
        skeleton instead, so the reader always gets the four tables with their real
        numbers and explicit `<!-- TODO -->` markers for the missing prose. Never
        overwrite an existing report, and never fail the job over it.
        """
        report = package_root / "final_report.md"
        if report.is_file():
            return
        script = self.settings.skill_dir / "scripts" / "build_final_report.py"
        if not script.exists():
            self.log(job_dir, f"[final-report] 跳过：缺少生成脚本 {script}")
            return
        command = [sys.executable, str(script), str(package_root), "--prefix", prefix]
        completed = subprocess.run(command, text=True, capture_output=True)
        if completed.returncode != 0:
            detail = (completed.stderr or completed.stdout or "").strip()[:1500]
            self.log(job_dir, f"[final-report] 生成分析文档失败，跳过（不影响任务其余产出）：{detail}")
            return
        self.log(
            job_dir,
            f"[final-report] Agent 未产出分析文档，已生成待补全骨架 {report}",
        )

    @staticmethod
    def package_trace(package: Path) -> Path | None:
        """The trace shipped with a result package, if it kept one."""
        for candidate in sorted((package / "trace").glob("*.sqlite")):
            return candidate
        context_path = package / "metadata" / "context.json"
        if not context_path.is_file():
            return None
        try:
            recorded = json.loads(context_path.read_text()).get("sqlite_path")
        except (json.JSONDecodeError, OSError):
            return None
        path = Path(recorded) if recorded else None
        return path if path and path.is_file() else None

    @staticmethod
    def analysed_device(metadata_dir: Path, package: Path, prefix: str) -> int | None:
        """The device the other tables were measured on, per the stable statistics."""
        for candidate in (
            metadata_dir / f"{prefix}_stable_statistics.json",
            package / f"{prefix}_stable_statistics.json",
        ):
            if not candidate.is_file():
                continue
            try:
                device = json.loads(candidate.read_text()).get("device")
            except (json.JSONDecodeError, OSError):
                return None
            return device if isinstance(device, int) else None
        return None

    @staticmethod
    def chunked_prefill_size(metadata_dir: Path) -> int | None:
        """Read --chunked-prefill-size out of the job's launch command.

        Prefill steps run with a full chunk, so this is the step's token count --
        it cannot be derived from the trace the way a decode batch size can.
        """
        context_path = metadata_dir / "context.json"
        if not context_path.is_file():
            return None
        try:
            launch = json.loads(context_path.read_text()).get("launch_path")
        except (json.JSONDecodeError, OSError):
            return None
        if not launch or not Path(launch).is_file():
            return None
        try:
            text = Path(launch).read_text(errors="ignore")
        except OSError:
            return None
        # The value is frequently a shell arithmetic expression rather than a plain
        # integer literal (e.g. `--chunked-prefill-size $((32 * 1024))`), so pull out
        # the whole expression and evaluate it instead of expecting a bare number.
        match = re.search(
            r"chunked[-_]prefill[-_]size\s*[\s=]\s*"
            r"(\$\(\([^)]*\)\)|\d+)", text,
        )
        if not match:
            return None
        token = match.group(1)
        if token.startswith("$(("):
            expression = token[3:-2]
            if not re.fullmatch(r"[\d\s+\-*/()]+", expression):
                return None  # refuse to eval anything beyond arithmetic on literals
            try:
                return int(eval(expression, {"__builtins__": {}}, {}))
            except (SyntaxError, ZeroDivisionError, TypeError, ValueError):
                return None
        return int(token)

    @staticmethod
    def declaration_args(metadata_dir: Path) -> list[str]:
        """--model-config / --launch / --runtime-evidence / --stage for the builder.

        These are the job's own inputs, recorded in context.json when the job started,
        plus the runtime evidence the agent audited out of the trace. They declare how
        many layers a step has and whether speculative decoding is on, which the
        forward-pipeline table verifies rather than infers.
        """
        context_path = metadata_dir / "context.json"
        if not context_path.is_file():
            return []
        try:
            context = json.loads(context_path.read_text())
        except (json.JSONDecodeError, OSError):
            return []
        args: list[str] = []
        for flag, key in (("--model-config", "config_path"), ("--launch", "launch_path")):
            value = context.get(key) or context.get(key.replace("_path", ""))
            if value and Path(value).is_file():
                args += [flag, str(value)]
        stage = context.get("stage")
        if stage:
            args += ["--stage", str(stage)]
        for candidate in (
            metadata_dir / f"{context.get('prefix', 'analysis')}_runtime_evidence.json",
            metadata_dir / "runtime_evidence.json",
        ):
            if candidate.is_file():
                args += ["--runtime-evidence", str(candidate)]
                break
        return args

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
            for suffix in AGENT_CSV_SUFFIXES:
                shutil.copy2(csv_source / f"{prefix}{suffix}", csv_dir)
            pipeline_name = f"{prefix}{FORWARD_PIPELINE_SUFFIX}"
            # older packages kept the pipeline table beside the sidecars
            found = next(
                (path for path in (csv_source / pipeline_name, *source.rglob(pipeline_name))
                 if path.is_file()),
                None,
            )
            if found:
                shutil.copy2(found, csv_dir / pipeline_name)
            for sidecar in source.rglob("*.json"):
                if sidecar.name in {"analysis.json", "nsysscope-package.json"}:
                    continue
                shutil.copy2(sidecar, metadata_dir / sidecar.name)
            traces = [*source.rglob("*.sqlite")]
            for trace in traces:
                shutil.copy2(trace, trace_dir / trace.name)
            # A package predating the seventh table can still be completed from its trace.
            self.ensure_forward_pipeline(
                job_id or "", result_dir, csv_dir, prefix,
                self.package_trace(result_dir), metadata_root=metadata_dir,
            )
            source_analysis = source / "analysis.json"
            if source_analysis.exists():
                shutil.copy2(source_analysis, result_dir / "analysis.json")
            else:
                self.convert(
                    csv_dir, result_dir / "analysis.json", prefix, request,
                    job_id=job_id,
                )
            self.ensure_xlsx(csv_dir, xlsx_dir, job_id=job_id)
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
        raise RuntimeError("ZIP package does not contain one complete table directory")

    def organize_result_package(
        self, result_dir: Path, package_dir: Path, prefix: str,
        sqlite_path: Path | None, job_id: str | None = None,
    ) -> Path | None:
        """Lay the finished analysis out as a portable package.

        Delegated to the Skill's `finalize_package.py`, which is the only
        implementation of the canonical layout: a Skill run done by hand ends in the
        same directory a job does, instead of a flat one a reader has to interpret.
        Returns the trace's new path, which the caller records in context.json.
        """
        script = self.settings.skill_dir / "scripts" / "finalize_package.py"
        if not script.is_file():
            raise RuntimeError(f"analysis Skill is missing the packager: {script}")
        command = [
            shutil.which("python3") or "python3", str(script), str(result_dir),
            "--prefix", prefix,
            "--tables", str(package_dir),
        ]
        # An imported package may not ship a trace, and then there is nothing to
        # place; every other artifact is still laid out.
        if sqlite_path is not None:
            command.extend(["--trace", str(sqlite_path)])
        completed = self._run_tracked(job_id, command)
        if completed.returncode:
            raise RuntimeError(
                completed.stdout.strip() or "analysis package layout failed"
            )
        # Staging is the tool's own doing, so cleaning it up stays here rather than
        # in a script a Skill user runs.
        staged_skill = result_dir / ".comate"
        if staged_skill.exists():
            shutil.rmtree(staged_skill)
        manifest = json.loads((result_dir / "nsysscope-package.json").read_text())
        trace = manifest.get("trace")
        return result_dir / trace if trace else None

    @staticmethod
    def csv_package_dir(package: Path) -> Path:
        csv_dir = package / "csv"
        return csv_dir if csv_dir.is_dir() else package

    def validate_analysis(self, path: Path) -> None:
        """Last gate before a job is called succeeded: can the frontend render this?

        Delegated to the Skill's `validate_frontend_contract.py`, which is also what
        `validate_analysis_package.py --analysis-json` runs, so a Skill user can
        check the same contract without the service.
        """
        script = (
            self.settings.skill_dir / "scripts" / "validate_frontend_contract.py"
        )
        if not script.is_file():
            raise RuntimeError(
                f"analysis Skill is missing the contract checker: {script}"
            )
        completed = subprocess.run(
            [shutil.which("python3") or "python3", str(script), str(path)],
            text=True, capture_output=True,
        )
        if completed.returncode:
            detail = (completed.stderr or completed.stdout or "").strip()
            raise RuntimeError(detail or f"analysis.json failed the frontend contract: {path}")

    def validate_package(
        self, package: Path, prefix: str, *, analysis_path: Path | None = None,
        job_id: str | None = None, log_to: Path | None = None,
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
        # The validator reports non-fatal findings (naming stability, stage
        # composition) on stdout and still exits 0. Dropping that output made those
        # checks effectively dead, so keep them in the job log.
        output = completed.stdout.strip()
        if log_to is not None and output:
            self.log(log_to, f"[validate] {output}")

    def publish_to_popo(
        self, job_id: str, job_dir: Path, username: str, token: str | None = None,
    ) -> str:
        analysis_path = job_dir / "analysis.json"
        if not analysis_path.is_file():
            raise RuntimeError("analysis.json is missing; cannot publish")
        if token:
            self.save_popo_token(username, token)
        # A fresh slug per publish. The first upload of the pair creates the page
        # and has no --previous-slug, so a slug derived from job_id alone fails
        # the second time a job is published -- which is exactly what happens
        # after its tables are rebuilt and it needs a new page.
        slug = f"nsysscope-{job_id}-{secrets.token_hex(4)}"
        return self._publish_analysis_bytes(
            analysis_path.read_bytes(), slug=slug, username=username,
        )

    def publish_analysis_payload(
        self, analysis: dict, username: str, token: str | None = None,
    ) -> str:
        if token:
            self.save_popo_token(username, token)
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
        """Directory holding the agent's six tables.

        The requested prefix wins when it is there, but an agent occasionally names
        its tables after the model it actually found in the trace. Accepting any
        complete set means such a run reports a prefix mismatch the caller can
        recover from (see detect_prefix) instead of "did not produce the six agent
        tables", which reads like the agent failed and hides a finished analysis.
        """
        candidates = [job_dir, *[path.parent for path in job_dir.rglob("*_stage_table.csv")]]
        fallback: Path | None = None
        for path in dict.fromkeys(candidates):
            # The forward-pipeline table is generated from the trace afterwards, so only
            # the agent's own tables can identify the package directory.
            if all((path / f"{prefix}{suffix}").exists() for suffix in AGENT_CSV_SUFFIXES):
                return path
            if fallback is not None:
                continue
            for marker in sorted(path.glob("*_stage_table.csv")):
                other = marker.name.removesuffix("_stage_table.csv")
                if all(
                    (path / f"{other}{suffix}").exists() for suffix in AGENT_CSV_SUFFIXES
                ):
                    fallback = path
                    break
        if fallback is not None:
            return fallback
        raise RuntimeError("Agent run did not produce the six agent tables of the package")

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
            if all((package / f"{prefix}{suffix}").exists() for suffix in AGENT_CSV_SUFFIXES)
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
