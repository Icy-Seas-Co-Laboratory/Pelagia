from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..config import CoreConfig
from ..observability import DatabaseLogger
from ..storage.blob_store import BlobStore, create_kvstore, create_skerrykv_store, initialize_kvstore
from ..storage.skerrykv import SkerryKVStore
from ..storage.postgres import DEFAULT_PROJECT_ID, PostgresRepository


@dataclass(slots=True)
class AppContext:
    """Dependency container used by interfaces and workers."""

    config: CoreConfig
    repository: PostgresRepository | None = None
    kvstore: BlobStore | None = None
    hot_storage: BlobStore | None = None
    logger: DatabaseLogger | None = None
    oracle: Any | None = None
    active_project_id: str | None = None
    _project_kvstores: dict[str, BlobStore] = field(default_factory=dict)
    _project_settings_cache: dict[str, tuple[dict[str, Any], float]] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: CoreConfig | None = None) -> "AppContext":
        """Create a context without opening a KVStore until a project is selected."""
        resolved = config or CoreConfig.load()
        repository = PostgresRepository(resolved)
        logger = DatabaseLogger(repository)
        hot_storage = create_skerrykv_store(resolved.hot_storage)
        # Repository methods resolve persisted ROI/mask references through this
        # service without making HTTP concerns part of SQL code.
        repository.hot_store = hot_storage
        return cls(
            config=resolved,
            repository=repository,
            kvstore=create_skerrykv_store(resolved.cold_storage),
            hot_storage=hot_storage,
            logger=logger,
        )

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
        if resolved_project_id == DEFAULT_PROJECT_ID and self.kvstore is not None:
            return self.kvstore
        if self.active_project_id == resolved_project_id and self.kvstore is not None:
            return self.kvstore
        # Test and one-off callers may explicitly inject a legacy store. Keep
        # that isolated path available while normal contexts use SkerryKV.
        if self.kvstore is not None and not isinstance(self.kvstore, SkerryKVStore):
            root_path = project.get("kvstore_root_path") or (
                Path(self.config.kvstore.directory) / "projects" / resolved_project_id
            )
            store = create_kvstore(root_path, self.config.kvstore)
            if initialize and not store.initialized:
                initialize_kvstore(store, self.config.kvstore)
            self._project_kvstores[resolved_project_id] = store
            return store
        del initialize
        store = create_skerrykv_store(self.config.cold_storage)
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

    def close(self) -> None:
        """Release process-owned service resources."""

        stores = list(self._project_kvstores.values())
        if self.kvstore is not None:
            stores.append(self.kvstore)
        if self.hot_storage is not None:
            stores.append(self.hot_storage)
        seen: set[int] = set()
        for store in stores:
            if id(store) not in seen:
                seen.add(id(store))
                close = getattr(store, "close", None)
                if close is not None:
                    close()
        if self.repository is not None:
            self.repository.close()
