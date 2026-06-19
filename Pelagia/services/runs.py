from __future__ import annotations

from ..domain import PlannedRun
from ..storage.postgres import PostgresRepository


class RunService:
    """Coordinates run registration and lifecycle operations."""

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def register_planned_run(self, planned_run: PlannedRun, *, project_id: str) -> dict:
        """Persist a planned run, its assets, and its initial jobs."""
        return self.repository.register_planned_run(planned_run, project_id=project_id)
