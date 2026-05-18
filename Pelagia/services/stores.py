from __future__ import annotations

from ..config import KVStoreConfig
from ..storage.kvstore import KVStore


class StoreService:
    """Service facade for storing and loading large payload bytes."""

    def __init__(self, store: KVStore):
        self.store = store

    @classmethod
    def from_config(cls, config: KVStoreConfig) -> "StoreService":
        """Create the service for a configured KVStore root."""
        return cls(KVStore(config.root_path))

    def ensure_initialized(self, config: KVStoreConfig) -> None:
        """Initialize the store if it has not been initialized yet."""
        if self.store.initialized:
            return
        self.store.initialize(
            hash_algorithm=config.hash_algorithm,
            prefix_length=config.prefix_length,
            max_db_bytes=config.max_db_bytes,
            max_rows=config.max_rows,
        )

    def put_payload(self, payload: bytes | bytearray | str) -> str:
        """Store payload data and return its canonical content key."""
        return self.store.put_store(payload)

    def get_payload(self, key: str) -> bytes:
        """Load payload data by canonical content key."""
        return self.store.get_store(key)
