from __future__ import annotations

import os
import socket
import time
from dataclasses import dataclass, field
from threading import Event

from ..domain import PipelineStage
from ..services.context import AppContext
from .handlers import HandlerRegistry


@dataclass(slots=True)
class Worker:
    """Simple single-process worker loop skeleton."""

    context: AppContext
    handlers: HandlerRegistry
    worker_id: str = field(default_factory=lambda: f"worker-{socket.gethostname()}")

    def _capabilities(self, stages: list[PipelineStage] | None = None) -> list[str]:
        return [stage.value for stage in stages or []]

    def _touch(
        self,
        status: str,
        *,
        stages: list[PipelineStage] | None = None,
        leased_job_id: str | None = None,
        shutdown_requested: bool | None = None,
    ) -> dict | None:
        if self.context.repository is None:
            return None
        return self.context.repository.touch_worker(
            self.worker_id,
            status=status,
            leased_job_id=leased_job_id,
            capabilities=self._capabilities(stages),
            metadata={"hostname": socket.gethostname()},
            pid=os.getpid(),
            shutdown_requested=shutdown_requested,
        )

    def shutdown_requested(self) -> bool:
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")
        session = self.context.repository.get_worker_session(self.worker_id)
        return bool(session and session.get("shutdown_requested"))

    def run_once(self, stages: list[PipelineStage] | None = None) -> int:
        """Claim and process currently available jobs once."""
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")

        if self.shutdown_requested():
            self._touch("stopped", stages=stages)
            return 0

        self._touch("idle", stages=stages)
        jobs = self.context.repository.claim_jobs(self.worker_id, stages=stages)
        for job in jobs:
            self._touch("working", stages=stages, leased_job_id=str(job["id"]))
            try:
                result = self.handlers.handle(job, self.context)
                self.context.repository.complete_job(job["id"], result=result)
            except Exception as exc:
                self.context.repository.record_failure(job["id"], str(exc), retryable=True)
            finally:
                self._touch("idle", stages=stages)
        return len(jobs)

    def run_forever(
        self,
        stages: list[PipelineStage] | None = None,
        *,
        idle_sleep_seconds: float = 2.0,
        requeue_interval_seconds: float = 30.0,
        stop_event: Event | None = None,
    ) -> None:
        """Run this worker until signaled or externally requested to shut down."""
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")

        stop = stop_event or Event()
        last_requeue_at = 0.0
        self._touch("idle", stages=stages, shutdown_requested=False)
        try:
            while not stop.is_set():
                if self.shutdown_requested():
                    break

                now = time.monotonic()
                if now - last_requeue_at >= requeue_interval_seconds:
                    self.context.repository.requeue_expired_jobs()
                    last_requeue_at = now

                claimed = self.run_once(stages=stages)
                if claimed == 0:
                    stop.wait(idle_sleep_seconds)
        finally:
            self._touch("stopped", stages=stages, shutdown_requested=False)
