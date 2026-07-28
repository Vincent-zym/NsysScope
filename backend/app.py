from __future__ import annotations

import hmac
import json
import secrets
from datetime import UTC, datetime
from pathlib import Path

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from .config import Settings
from .models import JobCreate, JobView
from .runner import JobRunner
from .store import JobStore


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    settings.prepare()
    store = JobStore(settings.data_dir / "jobs.sqlite")
    runner = JobRunner(settings, store)
    app = FastAPI(title="NsysScope Analyzer API", version="0.4.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-NsysScope-Token"],
    )

    def authorize(x_nsysscope_token: str = Header(default="")) -> None:
        if settings.api_token and not hmac.compare_digest(x_nsysscope_token, settings.api_token):
            raise HTTPException(status_code=401, detail="invalid API token")

    def view(job: dict) -> JobView:
        updated = job["updated_at"]
        if isinstance(updated, str):
            updated = datetime.fromisoformat(updated)
        activity = updated
        log_path = Path(job["output_dir"]) / "job.log"
        if log_path.exists():
            log_activity = datetime.fromtimestamp(log_path.stat().st_mtime, UTC)
            activity = max(activity, log_activity)
        job["last_activity_at"] = activity
        job["idle_seconds"] = (
            max(0, int((datetime.now(UTC) - activity).total_seconds()))
            if job["status"] not in {"succeeded", "failed", "cancelled"} else None
        )
        return JobView.model_validate(job)

    @app.get("/api/health")
    def health() -> dict:
        return {
            "status": "ok",
            "codex_enabled": settings.codex_enabled,
            "providers": runner.provider_status(),
            "auth_required": bool(settings.api_token),
            "max_workers": settings.max_workers,
        }

    @app.get("/api/providers/{provider}/models", dependencies=[Depends(authorize)])
    def provider_models(provider: str) -> dict:
        if provider not in {"codex", "comate"}:
            raise HTTPException(status_code=404, detail="unknown Agent Provider")
        status = runner.provider_status()[provider]
        if not status["ready"]:
            raise HTTPException(status_code=422, detail=status["message"])
        try:
            return runner.provider_models(provider)
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

    @app.post("/api/jobs", response_model=JobView, dependencies=[Depends(authorize)])
    def create_job(request: JobCreate) -> JobView:
        if request.mode != "existing_package":
            provider = runner.provider_status()[request.agent_provider]
            if not provider["ready"]:
                raise HTTPException(status_code=422, detail=provider["message"])
        for field in (
            "report_path", "config_path", "launch_path", "source_path",
            "design_path", "existing_package_path",
        ):
            value = getattr(request, field)
            if value:
                try:
                    settings.resolve_allowed(value, kind=field)
                except ValueError as exc:
                    raise HTTPException(status_code=422, detail=str(exc)) from exc
        job_id = secrets.token_hex(8)
        output_dir = settings.data_dir / "jobs" / job_id
        output_dir.mkdir(parents=True)
        (output_dir / "request.json").write_text(
            request.model_dump_json(indent=2) + "\n",
        )
        created = store.create(job_id, request, output_dir)
        runner.submit(job_id)
        return view(created)

    @app.get("/api/jobs", response_model=list[JobView], dependencies=[Depends(authorize)])
    def list_jobs(limit: int = Query(default=30, ge=1, le=100)) -> list[JobView]:
        return [view(item) for item in store.list(limit)]

    @app.get("/api/jobs/{job_id}", response_model=JobView, dependencies=[Depends(authorize)])
    def get_job(job_id: str) -> JobView:
        try:
            return view(store.get(job_id))
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc

    @app.get("/api/jobs/{job_id}/logs", dependencies=[Depends(authorize)])
    def get_logs(
        job_id: str,
        after: int = Query(default=0, ge=0),
        limit: int = Query(default=200, ge=1, le=1000),
    ) -> dict:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        path = Path(job["output_dir"]) / "job.log"
        lines: list[str] = []
        reset = False
        total = path.stat().st_size if path.exists() else 0
        if after > total:
            after = 0
            reset = True
        next_offset = after
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(after)
                for _ in range(limit):
                    chunk = handle.readline(settings.job_log_line_max_bytes + 1)
                    if not chunk:
                        break
                    truncated = (
                        len(chunk) > settings.job_log_line_max_bytes
                        and not chunk.endswith(b"\n")
                    )
                    lines.append(
                        chunk.decode("utf-8", errors="replace").rstrip()
                        + (" …[分段]" if truncated else "")
                    )
                next_offset = handle.tell()
        return {
            "after": after,
            "next": next_offset,
            "total": total,
            "has_more": next_offset < total,
            "reset": reset,
            "lines": lines,
        }

    @app.get("/api/jobs/{job_id}/analysis", dependencies=[Depends(authorize)])
    def get_analysis(job_id: str) -> FileResponse:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if job["status"] != "succeeded":
            raise HTTPException(status_code=409, detail="analysis is not ready")
        path = Path(job["output_dir"]) / "analysis.json"
        if not path.exists():
            raise HTTPException(status_code=500, detail="analysis artifact is missing")
        return FileResponse(path, media_type="application/json", filename=f"{job_id}-analysis.json")

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobView, dependencies=[Depends(authorize)])
    def cancel_job(job_id: str) -> JobView:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return view(job)
        runner.cancel(job_id)
        return view(store.update(
            job_id, status="cancelled", progress=100, message="任务已取消",
        ))

    @app.post(
        "/api/jobs/{job_id}/retry-conversion",
        response_model=JobView,
        dependencies=[Depends(authorize)],
    )
    def retry_conversion(job_id: str) -> JobView:
        try:
            job = store.get(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="job not found") from exc
        if job["status"] != "failed":
            raise HTTPException(status_code=409, detail="only failed jobs can retry conversion")
        job_dir = Path(job["output_dir"])
        prefix = JobCreate.model_validate(job["request"]).prefix
        try:
            runner.find_package(job_dir, prefix)
        except RuntimeError as exc:
            raise HTTPException(
                status_code=409,
                detail="the failed job has no complete six-table package",
            ) from exc
        queued = store.update(
            job_id, status="converting", progress=85,
            message="转换重试已进入队列", error="",
        )
        runner.submit_conversion_retry(job_id)
        return view(queued)

    app.state.settings = settings
    app.state.store = store
    app.state.runner = runner
    return app


app = create_app()
