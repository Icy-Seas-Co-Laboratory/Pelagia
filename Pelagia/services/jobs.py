from __future__ import annotations

from typing import Any, Sequence

from ..domain import JobStatus, PipelineStage
from ..storage.postgres import PostgresRepository
from .job_commands import command_from_payload, command_model


class JobService:
    """High-level job operations shared by API, CLI, and workers."""

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def enqueue(
        self,
        stage: PipelineStage,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_id: str | None = None,
        payload: dict[str, Any] | None = None,
        depends_on: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Create a queued processing job."""
        resolved_payload = dict(payload or {})
        if command_model(stage) is not None:
            resolved_payload = command_from_payload(stage, resolved_payload).to_payload()
        return self.repository.create_job(
            stage,
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            status=JobStatus.QUEUED,
            payload=resolved_payload,
            depends_on=depends_on or [],
        )

    def claim(self, worker_id: str, stages: Sequence[PipelineStage] | None = None) -> list[dict[str, Any]]:
        """Claim available jobs for a worker."""
        return self.repository.claim_jobs(worker_id=worker_id, stages=stages)
