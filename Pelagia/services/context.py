from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

from ..config import CoreConfig
from ..observability import DatabaseLogger
from ..storage.blob_store import BlobStore, create_kvstore, initialize_kvstore
from ..storage.postgres import PostgresRepository


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
        """Create a context without opening a KVStore until a project is selected."""
        resolved = config or CoreConfig.load()
        repository = PostgresRepository(resolved)
        logger = DatabaseLogger(repository)
        return cls(config=resolved, repository=repository, kvstore=None, logger=logger)

    def kvstore_for_project(self, project_id: str | None, *, initialize: bool = True) -> BlobStore | None:
        """Return the physical KVStore for a project."""
        if project_id is None:
            return self.kvstore
        resolved_project_id = str(project_id)
        cached = self._project_kvstores.get(resolved_project_id)
        if cached is not None:
            return cached
        if self.repository is None:
            return None
        project = self.repository.get_project(resolved_project_id)
        if project is None:
            return None
        root_path = project.get("kvstore_root_path")
        if not root_path:
            if self.kvstore is not None:
                return self.kvstore
            return None
        if self.active_project_id == resolved_project_id and self.kvstore is not None:
            return self.kvstore
        store = create_kvstore(root_path, self.config.kvstore)
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
