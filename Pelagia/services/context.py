from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import CoreConfig
from ..observability import DatabaseLogger
from ..storage.blob_store import BlobStore, create_kvstore, initialize_kvstore
from ..storage.postgres import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_KEY, PostgresRepository


@dataclass(slots=True)
class AppContext:
    """Dependency container used by interfaces and workers."""

    config: CoreConfig
    repository: PostgresRepository | None = None
    kvstore: BlobStore | None = None
    logger: DatabaseLogger | None = None
    active_project_id: str | None = None
    _project_kvstores: dict[str, BlobStore] = field(default_factory=dict)
    _project_settings_cache: dict[str, tuple[dict[str, Any], float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: CoreConfig | None = None) -> "AppContext":
        """Create a context with configured storage adapters."""
        resolved = config or CoreConfig.load()
        kvstore = create_kvstore(resolved.kvstore.root_path, resolved.kvstore)
        repository = PostgresRepository(resolved)
        logger = DatabaseLogger(repository)
        return cls(config=resolved, repository=repository, kvstore=kvstore, logger=logger)

    def kvstore_for_project(self, project_id: str | None, *, initialize: bool = True) -> BlobStore | None:
        """Return the physical KVStore for a project."""
        if self.kvstore is None:
            return None
        if project_id is None or str(project_id) == DEFAULT_PROJECT_ID:
            return self.kvstore

        resolved_project_id = str(project_id)
        if self._project_uses_default_kvstore(resolved_project_id):
            return self.kvstore
        cached = self._project_kvstores.get(resolved_project_id)
        if cached is not None:
            return cached

        store = create_kvstore(self._kvstore_root_for_project(resolved_project_id), self.config.kvstore)
        if initialize and not store.initialized:
            initialize_kvstore(store, self.config.kvstore)
        self._project_kvstores[resolved_project_id] = store
        return store

    def for_project(self, project_id: str | None) -> "AppContext":
        if project_id is None or project_id == self.active_project_id:
            return self
        return replace(
            self,
            kvstore=self.kvstore_for_project(project_id),
            active_project_id=str(project_id),
        )

    def _project_uses_default_kvstore(self, project_id: str) -> bool:
        if self.repository is None:
            return False
        project = self.repository.get_project(project_id)
        return bool(
            project is not None
            and project.get("project_key") == DEFAULT_PROJECT_KEY
            and not project.get("kvstore_root_path")
        )

    def _kvstore_root_for_project(self, project_id: str) -> Path:
        if self.repository is not None:
            project = self.repository.get_project(project_id)
            if project is not None and project.get("kvstore_root_path"):
                return Path(str(project["kvstore_root_path"]))
        return Path(self.config.kvstore.root_path) / "projects" / project_id
