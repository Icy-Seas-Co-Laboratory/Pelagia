"""Compatibility wrapper for the package KVStore implementation."""

from Pelagia.storage.kvstore import (
    KVStore,
    KVStoreAlreadyInitializedError,
    KVStoreConfigError,
    KVStoreError,
    KVStoreIntegrityError,
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
    "KVStoreRotationError",
]
