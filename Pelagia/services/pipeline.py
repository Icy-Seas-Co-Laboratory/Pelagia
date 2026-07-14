"""Use cases shared by HTTP endpoints, CLI commands, and worker stages."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..domain import JobStatus, PipelineStage
from ..processing.frame_correction import generate_background_for_frames
from ..services.context import AppContext
from .jobs import JobService


class PipelineService:
    """Own pipeline submission and synchronous frame-processing use cases."""

    def __init__(self, context: AppContext):
        self.context = context
        if context.repository is None:
            raise RuntimeError("Pipeline operations require a PostgresRepository.")
        self.jobs = JobService(context.repository)

    def queue(
        self,
        stage: PipelineStage | str,
        *,
        project_id: str,
        run_id: str | None = None,
        asset_id: str | None = None,
        payload: dict[str, Any] | None = None,
        depends_on: Sequence[str] | None = None,
        priority: int | None = None,
        status: JobStatus = JobStatus.QUEUED,
        max_attempts: int | None = None,
        summary: str | None = None,
    ) -> dict[str, Any]:
        """Validate a typed command and submit it to the shared job queue."""
        return self.jobs.enqueue(
            PipelineStage(stage),
            project_id=project_id,
            run_id=run_id,
            asset_id=asset_id,
            payload=payload,
            depends_on=depends_on,
            priority=priority,
            status=status,
            max_attempts=max_attempts,
            summary=summary,
        )

    def generate_background(
        self,
        frame_ids: Sequence[str],
        *,
        payload_kind: str | None = None,
        encoding: str | None = None,
        quality: int | None = None,
    ) -> dict[str, Any]:
        """Generate background payloads for already-stored project frames."""
        return generate_background_for_frames(
            list(frame_ids),
            context=self.context,
            payload_kind=payload_kind,
            encoding=encoding,
            quality=quality,
        )
