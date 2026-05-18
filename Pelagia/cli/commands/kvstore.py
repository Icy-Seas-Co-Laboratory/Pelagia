from __future__ import annotations

from ...config import CoreConfig
from ...services.stores import StoreService


def initialize_kvstore(config: CoreConfig | None = None) -> str:
    """Initialize the configured KVStore and return its path."""
    resolved = config or CoreConfig.from_env()
    service = StoreService.from_config(resolved.kvstore)
    service.ensure_initialized(resolved.kvstore)
    return str(service.store.root_path)
