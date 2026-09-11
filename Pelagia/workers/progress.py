from __future__ import annotations

from time import monotonic
from typing import Any

from ..processing.timing import measure_phase
from ..services.context import AppContext


class JobControlInterrupt(RuntimeError):
    """Base class for expected, non-failure exits from a leased handler."""


class JobLeaseLost(JobControlInterrupt):
    """Raised when a handler no longer owns the job lease."""


class JobPauseRequested(JobControlInterrupt):
    """Raised after a handler acknowledges a pause at a safe checkpoint."""


class JobCancellationRequested(JobControlInterrupt):
    """Raised when cancellation is observed at a safe checkpoint."""


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

    @property
    def claim_identity(self) -> dict[str, str]:
        """Return the ephemeral worker claim identity when this is leased work."""
        worker_id = self.job.get("_worker_id")
        lease_token = self.job.get("_lease_token")
        if worker_id and lease_token:
            return {"worker_id": str(worker_id), "lease_token": str(lease_token)}
        return {}

    def start(self, message: str | None = None) -> None:
        self.update(0, message=message or f"Starting {self.stage}", force=True)

    def checkpoint(self) -> dict[str, Any] | None:
        """Stop leased work before the next batch when its control state changed.

        Lightweight repositories and direct handler calls do not carry a lease
        identity, so checkpoints remain backwards compatible no-ops for them.
        A missing control row for claimed work means that this worker no longer
        owns the lease (including the current immediate-cancellation behavior).
        """
        repository = self.context.repository
        identity = self.claim_identity
        if repository is None or not identity:
            return None
        get_control = getattr(repository, "get_job_control", None)
        if not callable(get_control) or self.job_id is None:
            return None
        control = get_control(self.job_id, **identity)
        if control is None:
            raise JobLeaseLost(f"Lease lost for job {self.job_id}")

        reason = str(control.get("control_reason") or "")
        status = str(control.get("status") or "")
        if reason.startswith("cancel_requested:") or status == "cancelled":
            finalize = getattr(repository, "finalize_cancelled_job", None)
            if callable(finalize):
                finalized = finalize(self.job_id, **identity)
                if finalized is None:
                    raise JobLeaseLost(f"Lease lost for job {self.job_id}")
            raise JobCancellationRequested(f"Cancellation requested for job {self.job_id}")
        if reason.startswith("pause_requested:"):
            finalize = getattr(repository, "finalize_paused_job", None)
            if not callable(finalize):
                raise JobLeaseLost(f"Cannot acknowledge pause for job {self.job_id}")
            finalized = finalize(self.job_id, **identity)
            if finalized is None:
                raise JobLeaseLost(f"Lease lost for job {self.job_id}")
            raise JobPauseRequested(f"Pause requested for job {self.job_id}")
        return control

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
        # Check before throttling so every logical batch boundary observes
        # operator controls even when no progress row needs to be emitted.
        self.checkpoint()
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
        with measure_phase("progress.database_update"):
            updated = update_job_progress(
                job_id,
                progress,
                summary=message,
                log_message=None,
                **self.claim_identity,
            )
        if updated is None and self.claim_identity:
            raise JobLeaseLost(f"Lease lost while updating progress for job {job_id}")
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
