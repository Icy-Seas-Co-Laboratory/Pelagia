from __future__ import annotations

from ..config import KVStoreConfig
from ..storage.blob_store import BlobStore, create_kvstore, initialize_kvstore


class StoreService:
    """Service facade for storing and loading large payload bytes."""

    def __init__(self, store: BlobStore):
        self.store = store

    @classmethod
    def from_config(cls, config: KVStoreConfig) -> "StoreService":
        """Create the service for a configured KVStore root."""
        return cls(create_kvstore(config.root_path, config))

    def ensure_initialized(self, config: KVStoreConfig) -> None:
        """Initialize the store if it has not been initialized yet."""
        if self.store.initialized:
            return
        initialize_kvstore(self.store, config)

    def put_payload(self, payload: bytes | bytearray | str) -> str:
        """Store payload data and return its canonical content key."""
        return self.store.put_store(payload)

    def get_payload(self, key: str) -> bytes:
        """Load payload data by canonical content key."""
        return self.store.get_store(key)
