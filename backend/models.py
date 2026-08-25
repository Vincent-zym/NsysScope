from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator


JobStatus = Literal[
    "queued", "exporting", "analyzing", "converting", "validating",
    "succeeded", "failed", "cancelled",
]

# Model names that ship a bundled config.json under backend/model_configs/,
# keyed by stem (e.g. "GLM5.2" -> backend/model_configs/GLM5.2.json). Read
# once at import time since the bundled set is fixed at deploy time; this
# lets a job be created without config_path when model_name matches one of
# these, filled in below before the "missing required paths" check runs.
BUILTIN_MODEL_CONFIGS: dict[str, Path] = {
    path.stem: path
    for path in sorted((Path(__file__).resolve().parent / "model_configs").glob("*.json"))
}


class JobCreate(BaseModel):
    mode: Literal["codex_skill", "existing_package"] = "codex_skill"
    agent_provider: Literal["codex", "comate"] = "codex"
    agent_model: str = Field(default="", max_length=200)
    model_name: str = Field(min_length=1, max_length=160)
    stage: Literal["prefill", "decode"]
    hardware: str = Field(min_length=1, max_length=120)
    report_path: str | None = None
    config_path: str | None = None
    launch_path: str | None = None
    source_path: str | None = None
    design_path: str | None = None
    existing_package_path: str | None = None
    result_path: str | None = None
    prefix: str = Field(default="analysis", pattern=r"^[a-zA-Z0-9_-]+$")
    notes: str = Field(default="", max_length=4000)
    enable_operator_advisor: bool = False

    @model_validator(mode="after")
    def validate_mode(self) -> "JobCreate":
        if self.mode == "existing_package":
            if not self.existing_package_path:
                raise ValueError("existing_package_path is required for existing_package mode")
            if self.existing_package_path.lower().endswith(".zip") and not self.result_path:
                raise ValueError("result_path is required when importing a ZIP package")
            return self
        # A known model name ships a bundled config.json -- fill it in before
        # checking what's missing, so the user does not have to hunt down
        # the weights directory just to point config_path at its config. An
        # explicitly supplied config_path always wins (e.g. a locally
        # patched config for the same model).
        if not self.config_path and self.model_name in BUILTIN_MODEL_CONFIGS:
            self.config_path = str(BUILTIN_MODEL_CONFIGS[self.model_name])
        required = {
            "report_path": self.report_path,
            "config_path": self.config_path,
            "launch_path": self.launch_path,
            "source_path": self.source_path,
            "result_path": self.result_path,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(f"missing required paths: {', '.join(missing)}")
        return self


class JobView(BaseModel):
    id: str
    status: JobStatus
    progress: int
    message: str
    request: JobCreate
    created_at: datetime
    updated_at: datetime
    output_dir: str
    analysis_url: str | None = None
    optimization_url: str | None = None
    popo_url: str | None = None
    error: str | None = None
    last_activity_at: datetime | None = None
    idle_seconds: int | None = None


class PublishRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    analysis: dict = Field(default_factory=dict)
    token: str | None = Field(default=None, max_length=4000)


class PublishUsername(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    token: str | None = Field(default=None, max_length=4000)
