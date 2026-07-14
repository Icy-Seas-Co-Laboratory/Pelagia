from __future__ import annotations

from ..storage.postgres import PostgresRepository


class AssetService:
    """Coordinates asset metadata and blob payload storage."""

    def __init__(self, repository: PostgresRepository):
        self.repository = repository
        self.catalog = getattr(repository, "catalog", repository)

    def list_assets(self, run_id: str) -> list[dict]:
        """List assets registered for a run."""
        return self.catalog.list_assets(run_id)
