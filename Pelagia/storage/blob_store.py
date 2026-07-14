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


def named_kvstore_path(directory: str | Path, store_name: str) -> Path:
    """Return a named KVStore path rooted directly under ``directory``."""
    resolved_directory = Path(directory).expanduser().resolve(strict=False)
    normalized_name = str(store_name).strip()
    if not normalized_name or normalized_name in {".", ".."} or Path(normalized_name).name != normalized_name:
        raise ValueError("KVStore name must be a single non-empty directory name.")
    return resolved_directory / normalized_name


def create_named_kvstore(
    directory: str | Path,
    store_name: str,
    config: KVStoreConfig,
) -> BlobStore:
    """Create a store handle at ``directory / store_name`` without initializing it."""
    return create_kvstore(named_kvstore_path(directory, store_name), config)


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
