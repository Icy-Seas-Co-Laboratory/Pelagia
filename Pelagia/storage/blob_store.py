from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import KVStoreConfig
from .kvstore import KVStore
from .kvstore2 import KVStore2

BlobStore = KVStore | KVStore2


def create_kvstore(root_path: str | Path, config: KVStoreConfig) -> BlobStore:
    """Create the configured blob store implementation without initializing it."""
    if config.backend == "kvstore2":
        return KVStore2(root_path)
    return KVStore(root_path)


def initialize_kvstore(store: BlobStore, config: KVStoreConfig, *, overwrite: bool = False) -> None:
    """Initialize a configured blob store, hiding backend-specific options."""
    if isinstance(store, KVStore2):
        store.initialize(
            hash_algorithm=config.hash_algorithm,
            prefix_length=config.prefix_length,
            max_blob_bytes=config.max_blob_bytes,
            overwrite=overwrite,
        )
        return
    store.initialize(
        hash_algorithm=config.hash_algorithm,
        prefix_length=config.prefix_length,
        max_db_bytes=config.max_db_bytes,
        max_rows=config.max_rows,
        overwrite=overwrite,
    )


def reset_kvstore(store: BlobStore, config: KVStoreConfig) -> dict[str, Any]:
    """Reset a configured blob store, hiding backend-specific options."""
    if isinstance(store, KVStore2):
        return store.reset(
            hash_algorithm=config.hash_algorithm,
            prefix_length=config.prefix_length,
            max_blob_bytes=config.max_blob_bytes,
        )
    return store.reset(
        hash_algorithm=config.hash_algorithm,
        prefix_length=config.prefix_length,
        max_db_bytes=config.max_db_bytes,
        max_rows=config.max_rows,
    )
