from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator


JobStatus = Literal[
    "queued", "exporting", "analyzing", "converting", "validating",
    "succeeded", "failed", "cancelled",
]


class JobCreate(BaseModel):
    mode: Literal["codex_skill", "existing_package"] = "codex_skill"
    model_name: str = Field(min_length=1, max_length=160)
    stage: Literal["prefill", "decode"]
    hardware: str = Field(min_length=1, max_length=120)
    report_path: str | None = None
    config_path: str | None = None
    launch_path: str | None = None
    source_path: str | None = None
    design_path: str | None = None
    existing_package_path: str | None = None
    prefix: str = Field(default="analysis", pattern=r"^[a-zA-Z0-9_-]+$")
    notes: str = Field(default="", max_length=4000)

    @model_validator(mode="after")
    def validate_mode(self) -> "JobCreate":
        if self.mode == "existing_package":
            if not self.existing_package_path:
                raise ValueError("existing_package_path is required for existing_package mode")
            return self
        required = {
            "report_path": self.report_path,
            "config_path": self.config_path,
            "launch_path": self.launch_path,
            "source_path": self.source_path,
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
    error: str | None = None
    last_activity_at: datetime | None = None
    idle_seconds: int | None = None
