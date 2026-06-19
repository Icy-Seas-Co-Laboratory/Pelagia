from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from ..config import CoreConfig
from ..observability import DatabaseLogger
from ..storage.kvstore import KVStore
from ..storage.postgres import DEFAULT_PROJECT_ID, DEFAULT_PROJECT_KEY, PostgresRepository


@dataclass(slots=True)
class AppContext:
    """Dependency container used by interfaces and workers."""

    config: CoreConfig
    repository: PostgresRepository | None = None
    kvstore: KVStore | None = None
    logger: DatabaseLogger | None = None
    active_project_id: str | None = None
    _project_kvstores: dict[str, KVStore] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: CoreConfig | None = None) -> "AppContext":
        """Create a context with configured storage adapters."""
        resolved = config or CoreConfig.load()
        kvstore = KVStore(resolved.kvstore.root_path)
        repository = PostgresRepository(resolved)
        logger = DatabaseLogger(repository)
        return cls(config=resolved, repository=repository, kvstore=kvstore, logger=logger)

    def kvstore_for_project(self, project_id: str | None) -> KVStore | None:
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

        store = KVStore(self._kvstore_root_for_project(resolved_project_id))
        if not store.initialized:
            config = self.config.kvstore
            store.initialize(
                hash_algorithm=config.hash_algorithm,
                prefix_length=config.prefix_length,
                max_db_bytes=config.max_db_bytes,
                max_rows=config.max_rows,
            )
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
