from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .models import BUILTIN_MODEL_CONFIGS


def _paths(value: str) -> tuple[Path, ...]:
    return tuple(Path(item).expanduser().resolve() for item in value.split(":") if item)


def _discover_zulu() -> str:
    pattern = (
        ".comate-server/bin/*/extensions/baiducomate.comate/"
        "dist/zulu-cli/bin/zulu"
    )
    candidates = [
        path for path in Path.home().glob(pattern)
        if path.is_file() and os.access(path, os.X_OK)
    ]
    return str(max(candidates, key=lambda path: path.stat().st_mtime)) if candidates else ""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    allowed_roots: tuple[Path, ...]
    api_token: str
    cors_origins: tuple[str, ...]
    max_workers: int
    codex_enabled: bool
    codex_bin: str
    comate_enabled: bool
    comate_bin: str
    comate_username: str
    comate_model: str
    comate_platform: str
    comate_timeout_seconds: int
    comate_store_dir: Path
    agent_heartbeat_seconds: int
    agent_stall_timeout_seconds: int
    job_log_max_bytes: int
    job_log_line_max_bytes: int
    nsys_bin: str
    skill_dir: Path
    call_tree_skill_dir: Path
    converter: Path
    xlsx_converter: Path
    subprocess_timeout_seconds: int
    popo_username: str
    popo_upload_script: Path
    builtin_model_configs: dict[str, Path]

    @classmethod
    def from_env(cls) -> "Settings":
        project = Path(__file__).resolve().parents[1]
        default_root = Path.home().resolve()
        default_cache = Path(
            os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")
        ).expanduser().resolve()
        bundled_skill = project / "bundled" / "skills" / "sglang-nsys-static-analysis"
        bundled_call_tree_skill = (
            project / "bundled" / "skills" / "reconstruct-profiler-call-tree"
        )
        comate_bin = os.getenv("NSYSSCOPE_COMATE_BIN", "") or _discover_zulu()
        # Resolved before the constructor call because the two converters default to
        # scripts inside the selected Skill.
        skill_dir = Path(os.getenv("NSYSSCOPE_SKILL_DIR", bundled_skill)).resolve()
        return cls(
            data_dir=Path(os.getenv(
                "NSYSSCOPE_DATA_DIR", default_cache / "nsysscope" / "state",
            )).expanduser().resolve(),
            allowed_roots=_paths(os.getenv("NSYSSCOPE_ALLOWED_ROOTS", str(default_root))),
            api_token=os.getenv("NSYSSCOPE_API_TOKEN", ""),
            cors_origins=tuple(
                item.strip() for item in os.getenv(
                    "NSYSSCOPE_CORS_ORIGINS",
                    "http://localhost:3000,http://127.0.0.1:3000,"
                    "https://nsysscope-perf.z-yuanming.chatgpt.site",
                ).split(",") if item.strip()
            ),
            max_workers=max(1, int(os.getenv("NSYSSCOPE_MAX_WORKERS", "2"))),
            codex_enabled=os.getenv("NSYSSCOPE_CODEX_ENABLED", "false").lower() == "true",
            codex_bin=os.getenv("NSYSSCOPE_CODEX_BIN", "codex"),
            comate_enabled=os.getenv(
                "NSYSSCOPE_COMATE_ENABLED", "true" if comate_bin else "false",
            ).lower() == "true",
            comate_bin=comate_bin,
            comate_username=os.getenv("NSYSSCOPE_COMATE_USERNAME", ""),
            comate_model=os.getenv("NSYSSCOPE_COMATE_MODEL", ""),
            comate_platform=os.getenv("NSYSSCOPE_COMATE_PLATFORM", "internal").lower(),
            comate_timeout_seconds=max(
                60, int(os.getenv("NSYSSCOPE_COMATE_TIMEOUT_SECONDS", "7200")),
            ),
            # Where the Comate engine keeps one `chat_session_<uuid>` file per
            # conversation. The file records `workspaceDirectory`, which equals the
            # `--cwd` we pass, so a job can find its own conversation and watch it
            # advance -- the only signal that says the agent itself is still working
            # rather than that something under the job dir happened to change.
            comate_store_dir=Path(os.getenv(
                "NSYSSCOPE_COMATE_STORE_DIR", Path.home() / ".comate-engine" / "store",
            )).expanduser(),
            agent_heartbeat_seconds=max(
                5, int(os.getenv("NSYSSCOPE_AGENT_HEARTBEAT_SECONDS", "30")),
            ),
            # An agent that produces nothing, prints nothing and burns no CPU for this
            # long is stalled, not thinking: a dropped model request leaves the process
            # alive with no work pending, and waiting for comate_timeout_seconds then
            # burns hours for nothing. Long reasoning turns are covered by the CPU
            # signal, so this only has to outlast one slow tool call.
            agent_stall_timeout_seconds=max(
                0, int(os.getenv("NSYSSCOPE_AGENT_STALL_TIMEOUT_SECONDS", "1800")),
            ),
            job_log_max_bytes=max(
                1_048_576, int(os.getenv("NSYSSCOPE_JOB_LOG_MAX_BYTES", str(2 * 1024 * 1024))),
            ),
            job_log_line_max_bytes=max(
                1024, int(os.getenv("NSYSSCOPE_JOB_LOG_LINE_MAX_BYTES", str(16 * 1024))),
            ),
            nsys_bin=os.getenv("NSYSSCOPE_NSYS_BIN", "nsys"),
            skill_dir=skill_dir,
            # Optional: only used when a job supplies a torch profiler trace. A
            # missing directory disables the pre-pass instead of failing startup,
            # so prepare() deliberately does not check it.
            call_tree_skill_dir=Path(os.getenv(
                "NSYSSCOPE_CALL_TREE_SKILL_DIR",
                bundled_call_tree_skill,
            )).resolve(),
            # Both live in the Skill, not here: a standalone Skill run has to be able
            # to produce the same analysis.json and xlsx/ the tool produces, and one
            # copy is the only way that stays true. They follow the selected Skill,
            # so swapping Skill versions swaps the generators with it.
            converter=Path(os.getenv(
                "NSYSSCOPE_CONVERTER",
                skill_dir / "scripts/build_analysis_json.py",
            )).resolve(),
            xlsx_converter=Path(os.getenv(
                "NSYSSCOPE_XLSX_CONVERTER",
                skill_dir / "scripts/csv_to_xlsx.py",
            )).resolve(),
            subprocess_timeout_seconds=max(
                30, int(os.getenv("NSYSSCOPE_SUBPROCESS_TIMEOUT_SECONDS", "600")),
            ),
            popo_username=os.getenv("NSYSSCOPE_POPO_USERNAME", os.getenv("USER", "")),
            popo_upload_script=Path(os.getenv(
                "NSYSSCOPE_POPO_UPLOAD_SCRIPT",
                "/root/.comate/skills/.system/popo/scripts/upload.py",
            )).expanduser(),
            builtin_model_configs=dict(BUILTIN_MODEL_CONFIGS),
        )

    def prepare(self) -> None:
        if self.comate_platform not in {"internal", "saas"}:
            raise ValueError("NSYSSCOPE_COMATE_PLATFORM must be internal or saas")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(exist_ok=True)
        required_skill_files = (
            "SKILL.md",
            "scripts/build_static_analysis_tables.py",
            "scripts/audit_runtime_evidence.py",
            "scripts/validate_analysis_package.py",
            # The frontend contract and the workbook are generated from the Skill too,
            # so a Skill that cannot produce them is incomplete for this tool.
            "scripts/build_analysis_json.py",
            "scripts/csv_to_xlsx.py",
            "scripts/finalize_package.py",
            "scripts/validate_frontend_contract.py",
            "references/hardware-peaks.json",
        )
        missing = [
            relative for relative in required_skill_files
            if not (self.skill_dir / relative).is_file()
        ]
        if missing:
            raise ValueError(
                f"analysis Skill is incomplete at {self.skill_dir}: {', '.join(missing)}"
            )

    def resolve_allowed(self, value: str | Path, *, kind: str = "path") -> Path:
        path = Path(value).expanduser().resolve()
        if not any(path == root or path.is_relative_to(root) for root in self.allowed_roots):
            roots = ", ".join(map(str, self.allowed_roots))
            raise ValueError(f"{kind} is outside NSYSSCOPE_ALLOWED_ROOTS: {roots}")
        if not path.exists():
            raise ValueError(f"{kind} does not exist: {path}")
        return path

    def prepare_result_dir(self, value: str | Path) -> Path:
        path = Path(value).expanduser().resolve()
        if not any(path == root or path.is_relative_to(root) for root in self.allowed_roots):
            roots = ", ".join(map(str, self.allowed_roots))
            raise ValueError(f"result path is outside NSYSSCOPE_ALLOWED_ROOTS: {roots}")
        if path.exists() and not path.is_dir():
            raise ValueError(f"result path must be a directory: {path}")
        if path.exists():
            # A directory left over from a cancelled run only contains
            # logs/ (see JobRunner.wipe_job_outputs); treat that as reusable
            # so the same result_path can be resubmitted without picking a
            # new one each time.
            leftovers = {entry.name for entry in path.iterdir()}
            if leftovers - {"logs"}:
                raise ValueError(f"result path must be a new or empty directory: {path}")
        path.mkdir(parents=True, exist_ok=True)
        return path

