from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .models import JobCreate


class JobStore:
    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self._init()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init(self) -> None:
        with self.connect() as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    progress INTEGER NOT NULL,
                    message TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    output_dir TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    popo_url TEXT
                )
            """)
            columns = {row[1] for row in db.execute("PRAGMA table_info(jobs)")}
            if "popo_url" not in columns:
                db.execute("ALTER TABLE jobs ADD COLUMN popo_url TEXT")

    def create(self, job_id: str, request: JobCreate, output_dir: Path) -> dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self.connect() as db:
            db.execute(
                "INSERT INTO jobs "
                "(id, status, progress, message, request_json, output_dir, "
                "error, created_at, updated_at, popo_url) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id, "queued", 0, "任务已进入队列",
                    request.model_dump_json(), str(output_dir), None, now, now, None,
                ),
            )
        return self.get(job_id)

    def update(
        self, job_id: str, *, status: str | None = None, progress: int | None = None,
        message: str | None = None, error: str | None = None,
        popo_url: str | None = None,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if status is not None:
            fields["status"] = status
        if progress is not None:
            fields["progress"] = progress
        if message is not None:
            fields["message"] = message
        if error is not None:
            fields["error"] = error
        if popo_url is not None:
            fields["popo_url"] = popo_url
        clause = ", ".join(f"{key} = ?" for key in fields)
        with self.lock, self.connect() as db:
            db.execute(
                f"UPDATE jobs SET {clause} WHERE id = ?",
                (*fields.values(), job_id),
            )
        return self.get(job_id)

    def cancel_if_active(self, job_id: str) -> dict[str, Any]:
        """Mark a job cancelled only if it has not already reached a terminal
        state. Runs the read-check-write as one transaction under the lock so
        a run() completing concurrently cannot be overwritten after the fact,
        and this call cannot resurrect an already-finished job as cancelled.
        """
        now = datetime.now(timezone.utc).isoformat()
        with self.lock, self.connect() as db:
            db.execute(
                "UPDATE jobs SET status = 'cancelled', progress = 100, "
                "message = '任务已取消', updated_at = ? "
                "WHERE id = ? AND status NOT IN "
                "('succeeded', 'failed', 'cancelled')",
                (now, job_id),
            )
        return self.get(job_id)

    def get(self, job_id: str) -> dict[str, Any]:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        if row is None:
            raise KeyError(job_id)
        return self._decode(row)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,),
            ).fetchall()
        return [self._decode(row) for row in rows]

    @staticmethod
    def _decode(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        result["request"] = json.loads(result.pop("request_json"))
        result["analysis_url"] = (
            f"/api/jobs/{result['id']}/analysis"
            if result["status"] == "succeeded" else None
        )
        return result
