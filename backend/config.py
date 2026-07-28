from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
    agent_heartbeat_seconds: int
    job_log_max_bytes: int
    job_log_line_max_bytes: int
    nsys_bin: str
    skill_dir: Path
    converter: Path

    @classmethod
    def from_env(cls) -> "Settings":
        project = Path(__file__).resolve().parents[1]
        default_root = Path.home().resolve()
        comate_bin = os.getenv("NSYSSCOPE_COMATE_BIN", "") or _discover_zulu()
        return cls(
            data_dir=Path(os.getenv("NSYSSCOPE_DATA_DIR", project / ".data")).expanduser().resolve(),
            allowed_roots=_paths(os.getenv("NSYSSCOPE_ALLOWED_ROOTS", str(default_root))),
            api_token=os.getenv("NSYSSCOPE_API_TOKEN", ""),
            cors_origins=tuple(
                item.strip() for item in os.getenv(
                    "NSYSSCOPE_CORS_ORIGINS",
                    "http://localhost:3000,http://127.0.0.1:3000,"
                    "https://nsysscope-perf.z-yuanming.chatgpt.site",
                ).split(",") if item.strip()
            ),
            max_workers=max(1, int(os.getenv("NSYSSCOPE_MAX_WORKERS", "1"))),
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
            agent_heartbeat_seconds=max(
                5, int(os.getenv("NSYSSCOPE_AGENT_HEARTBEAT_SECONDS", "30")),
            ),
            job_log_max_bytes=max(
                1_048_576, int(os.getenv("NSYSSCOPE_JOB_LOG_MAX_BYTES", str(8 * 1024 * 1024))),
            ),
            job_log_line_max_bytes=max(
                1024, int(os.getenv("NSYSSCOPE_JOB_LOG_LINE_MAX_BYTES", str(16 * 1024))),
            ),
            nsys_bin=os.getenv("NSYSSCOPE_NSYS_BIN", "nsys"),
            skill_dir=Path(os.getenv(
                "NSYSSCOPE_SKILL_DIR",
                "/root/.codex/skills/sglang-nsys-static-analysis",
            )).resolve(),
            converter=Path(os.getenv(
                "NSYSSCOPE_CONVERTER",
                project / "scripts/build_analysis_json.py",
            )).resolve(),
        )

    def prepare(self) -> None:
        if self.comate_platform not in {"internal", "saas"}:
            raise ValueError("NSYSSCOPE_COMATE_PLATFORM must be internal or saas")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "jobs").mkdir(exist_ok=True)

    def resolve_allowed(self, value: str | Path, *, kind: str = "path") -> Path:
        path = Path(value).expanduser().resolve()
        if not any(path == root or path.is_relative_to(root) for root in self.allowed_roots):
            roots = ", ".join(map(str, self.allowed_roots))
            raise ValueError(f"{kind} is outside NSYSSCOPE_ALLOWED_ROOTS: {roots}")
        if not path.exists():
            raise ValueError(f"{kind} does not exist: {path}")
        return path
