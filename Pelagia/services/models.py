from __future__ import annotations

from ..domain import ModelRecord
from ..storage.postgres import PostgresRepository


class ModelService:
    """Coordinates model metadata registration and lookup."""

    def __init__(self, repository: PostgresRepository):
        self.repository = repository

    def register_model(self, model: ModelRecord) -> dict:
        """Register or update model metadata."""
        return self.repository.register_model(model)
