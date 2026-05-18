"""Compatibility wrapper for the package KVStore implementation."""

from Pelagia.storage.kvstore import (
    KVStore,
    KVStoreAlreadyInitializedError,
    KVStoreConfigError,
    KVStoreError,
    KVStoreIntegrityError,
    KVStoreLockError,
    KVStoreNotInitializedError,
    KVStoreRotationError,
)

__all__ = [
    "KVStore",
    "KVStoreError",
    "KVStoreNotInitializedError",
    "KVStoreAlreadyInitializedError",
    "KVStoreConfigError",
    "KVStoreIntegrityError",
    "KVStoreLockError",
    "KVStoreRotationError",
]
