from __future__ import annotations

import socket
from dataclasses import dataclass, field

from ..domain import PipelineStage
from ..services.context import AppContext
from .handlers import HandlerRegistry


@dataclass(slots=True)
class Worker:
    """Simple single-process worker loop skeleton."""

    context: AppContext
    handlers: HandlerRegistry
    worker_id: str = field(default_factory=lambda: f"worker-{socket.gethostname()}")

    def run_once(self, stages: list[PipelineStage] | None = None) -> int:
        """Claim and process currently available jobs once."""
        if self.context.repository is None:
            raise RuntimeError("Worker requires a PostgresRepository.")

        jobs = self.context.repository.claim_jobs(self.worker_id, stages=stages)
        for job in jobs:
            try:
                result = self.handlers.handle(job, self.context)
                self.context.repository.complete_job(job["id"], result=result)
            except Exception as exc:
                self.context.repository.record_failure(job["id"], str(exc), retryable=True)
        return len(jobs)
