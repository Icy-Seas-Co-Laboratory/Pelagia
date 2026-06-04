from __future__ import annotations

from dataclasses import dataclass

from ..config import CoreConfig
from ..observability import DatabaseLogger
from ..storage.kvstore import KVStore
from ..storage.postgres import PostgresRepository


@dataclass(slots=True)
class AppContext:
    """Dependency container used by interfaces and workers."""

    config: CoreConfig
    repository: PostgresRepository | None = None
    kvstore: KVStore | None = None
    logger: DatabaseLogger | None = None

    @classmethod
    def from_config(cls, config: CoreConfig | None = None) -> "AppContext":
        """Create a context with configured storage adapters."""
        resolved = config or CoreConfig.load()
        kvstore = KVStore(resolved.kvstore.root_path)
        repository = PostgresRepository(resolved)
        logger = DatabaseLogger(repository)
        return cls(config=resolved, repository=repository, kvstore=kvstore, logger=logger)
