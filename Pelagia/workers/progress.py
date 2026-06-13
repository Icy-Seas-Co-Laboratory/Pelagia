from __future__ import annotations

from time import monotonic
from typing import Any

from ..services.context import AppContext


class JobProgressReporter:
    """Small helper for writing throttled, structured job progress updates."""

    def __init__(
        self,
        job: dict[str, Any],
        context: AppContext,
        *,
        stage: str,
        unit: str,
        total: int | float,
        emit_every: int = 25,
        emit_interval_s: float = 1.5,
    ) -> None:
        self.job = job
        self.context = context
        self.stage = stage
        self.unit = unit
        self.total = max(0, int(total or 0))
        self.emit_every = max(1, int(emit_every))
        self.emit_interval_s = max(0.1, float(emit_interval_s))
        self.started_at = monotonic()
        self.last_emit_at = 0.0
        self.last_completed = -1

    @property
    def job_id(self) -> str | None:
        value = self.job.get("id")
        return None if value is None else str(value)

    def start(self, message: str | None = None) -> None:
        self.update(0, message=message or f"Starting {self.stage}", force=True)

    def update(
        self,
        completed: int | float,
        *,
        failed: int | float = 0,
        skipped: int | float = 0,
        current: dict[str, Any] | None = None,
        secondary: dict[str, Any] | None = None,
        message: str | None = None,
        force: bool = False,
    ) -> None:
        job_id = self.job_id
        repository = self.context.repository
        if repository is None or job_id is None:
            return
        update_job_progress = getattr(repository, "update_job_progress", None)
        if not callable(update_job_progress):
            return
        completed_int = max(0, int(completed or 0))
        now = monotonic()
        if (
            not force
            and completed_int < self.total
            and completed_int != 0
            and completed_int - self.last_completed < self.emit_every
            and now - self.last_emit_at < self.emit_interval_s
        ):
            return

        elapsed_s = max(0.0, now - self.started_at)
        progress = {
            "schema_version": 1,
            "stage": self.stage,
            "unit": self.unit,
            "total": self.total,
            "completed": completed_int,
            "failed": max(0, int(failed or 0)),
            "skipped": max(0, int(skipped or 0)),
            "percent": (completed_int / self.total * 100.0) if self.total else None,
            "current": current or {},
            "secondary": secondary or {},
            "rates": {
                "units_per_second": (completed_int / elapsed_s) if elapsed_s > 0 else None,
            },
            "message": message,
        }
        update_job_progress(
            job_id,
            progress,
            summary=message,
            log_message=None,
        )
        self.last_emit_at = now
        self.last_completed = completed_int

    def finish(
        self,
        *,
        completed: int | float | None = None,
        failed: int | float = 0,
        skipped: int | float = 0,
        secondary: dict[str, Any] | None = None,
        message: str | None = None,
    ) -> None:
        self.update(
            self.total if completed is None else completed,
            failed=failed,
            skipped=skipped,
            secondary=secondary,
            message=message or f"Finished {self.stage}",
            force=True,
        )
