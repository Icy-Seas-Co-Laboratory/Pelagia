from __future__ import annotations

import hashlib
import importlib.util
import os
import sqlite3

import pytest

from Pelagia.storage.kvstore import (
    KVStore,
    KVStoreAlreadyInitializedError,
    KVStoreConfigError,
    KVStoreIntegrityError,
    KVStoreLockError,
)


def sha256_key(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def payloads_with_same_prefix(count: int, prefix_length: int = 1) -> list[bytes]:
    buckets: dict[str, list[bytes]] = {}
    for index in range(10_000):
        payload = f"payload-{index}".encode("ascii")
        prefix = sha256_key(payload)[:prefix_length]
        buckets.setdefault(prefix, []).append(payload)
        if len(buckets[prefix]) == count:
            return buckets[prefix]
    raise AssertionError("Unable to find test payloads with matching prefix.")


def test_creating_new_store_leaves_uninitialized(tmp_path):
    store = KVStore(tmp_path / "store")

    assert store.initialized is False


def test_root_path_expands_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))

    store = KVStore("~/store")

    assert store.root_path == (tmp_path / "store").resolve()


def test_root_path_expands_environment_variables(monkeypatch, tmp_path):
    monkeypatch.setenv("PELAGIA_TEST_STORE_ROOT", str(tmp_path))

    store = KVStore("$PELAGIA_TEST_STORE_ROOT/nested")

    assert store.root_path == (tmp_path / "nested").resolve()


def test_root_path_resolves_relative_paths(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    store = KVStore("relative-store")

    assert store.root_path == (tmp_path / "relative-store").resolve()


def test_top_level_kvstore_import_remains_compatible():
    from kvstore import KVStore as CompatibilityKVStore

    assert CompatibilityKVStore is KVStore


def test_initialize_creates_config_manifest_prefixes_and_initial_dbs(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    assert store.initialized is True
    assert (store.root_path / "config.json").exists()
    assert (store.root_path / "manifest.json").exists()
    for prefix in "0123456789abcdef":
        assert (store.root_path / prefix).is_dir()
        assert (store.root_path / prefix / "store_000001.sqlite").exists()


def test_loading_existing_store_from_config(tmp_path):
    root = tmp_path / "store"
    KVStore(root).initialize(prefix_length=1)

    loaded = KVStore(root)

    assert loaded.initialized is True
    assert loaded.config is not None
    assert loaded.config["prefix_length"] == 1


def test_initialize_existing_store_without_overwrite_raises(tmp_path):
    root = tmp_path / "store"
    KVStore(root).initialize(prefix_length=1)

    with pytest.raises(KVStoreAlreadyInitializedError):
        KVStore(root).initialize(prefix_length=1)


def test_store_and_retrieve_bytes(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    payload = b"hello world"
    key = store.put_store(payload)

    assert key == sha256_key(payload)
    assert store.get_store(key) == payload


def test_store_and_retrieve_string_as_utf8_bytes(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    key = store.put_store("hello snow")

    assert store.get_store(key) == "hello snow".encode("utf-8")


def test_put_store_accepts_none_key_and_payload(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    payload = b"auto key"
    key = store.put_store(None, payload)

    assert key == sha256_key(payload)


def test_rejects_explicit_key_that_does_not_match_payload(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    wrong_key = sha256_key(b"other")
    with pytest.raises(KVStoreIntegrityError):
        store.put_store(wrong_key, b"payload")


def test_key_exists_true_and_false(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    key = store.put_store(b"present")

    assert store.key_exists(key) is True
    assert store.key_exists(sha256_key(b"missing")) is False


def test_key_delete_removes_existing_key(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    key = store.put_store(b"present")
    store.key_delete(key)

    assert store.key_exists(key) is False
    with pytest.raises(KeyError):
        store.get_store(key)


def test_key_delete_missing_raises_key_error(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    with pytest.raises(KeyError):
        store.key_delete(sha256_key(b"missing"))


def test_get_store_missing_raises_key_error(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    with pytest.raises(KeyError):
        store.get_store(sha256_key(b"missing"))


def test_idempotent_writes_of_same_payload(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    first = store.put_store(b"same")
    second = store.put_store(b"same")

    assert first == second
    assert store.status()["total_stored_blobs"] == 1


def test_prefix_write_lock_timeout(tmp_path):
    store = KVStore(tmp_path / "store", lock_timeout_s=0.01, lock_poll_interval_s=0.001, stale_lock_seconds=None)
    store.initialize(prefix_length=1)
    payload = b"locked payload"
    lock_path = store.root_path / sha256_key(payload)[:1] / ".kvstore.lock"
    lock_path.write_text("held by test", encoding="utf-8")

    with pytest.raises(KVStoreLockError):
        store.put_store(payload)


def test_stale_prefix_lock_is_removed(tmp_path):
    store = KVStore(tmp_path / "store", lock_timeout_s=0.5, lock_poll_interval_s=0.001, stale_lock_seconds=0)
    store.initialize(prefix_length=1)
    payload = b"stale lock payload"
    lock_path = store.root_path / sha256_key(payload)[:1] / ".kvstore.lock"
    lock_path.write_text("stale", encoding="utf-8")
    os.utime(lock_path, (0, 0))

    key = store.put_store(payload)

    assert store.get_store(key) == payload
    assert not lock_path.exists()


def test_integrity_error_on_same_key_with_different_payload_via_existing_row(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)
    key = store.put_store(b"original")
    prefix = key[:1]
    db_path = store.root_path / prefix / "store_000001.sqlite"

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE blobs SET payload = ?, size = ? WHERE key = ?", (b"changed", 7, key))

    with pytest.raises(KVStoreIntegrityError):
        store.put_store(key, b"original")


def test_status_reports_accurate_counts(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)
    store.put_store(b"a")
    store.put_store(b"bb")

    status = store.status()

    assert status["initialized"] is True
    assert status["prefix_directory_count"] == 16
    assert status["total_sqlite_files"] == 16
    assert status["total_stored_blobs"] == 2
    assert status["total_stored_payload_bytes"] == 3


def test_check_health_passes_for_valid_store(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    health = store.check_health()

    assert health["healthy"] is True
    assert health["errors"] == []


def test_check_health_reports_missing_sqlite_file(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)
    (store.root_path / "a" / "store_000001.sqlite").unlink()

    health = store.check_health()

    assert health["healthy"] is False
    assert any("Prefix a has no SQLite" in error for error in health["errors"])


def test_sha256_key_validation(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)

    with pytest.raises(KVStoreConfigError):
        store.key_exists("abc")
    with pytest.raises(KVStoreConfigError):
        store.key_exists("G" * 64)


def test_blake3_behavior_if_dependency_is_installed(tmp_path):
    if importlib.util.find_spec("blake3") is None:
        with pytest.raises(KVStoreConfigError):
            KVStore(tmp_path / "store").initialize(hash_algorithm="blake3", prefix_length=1)
        return

    store = KVStore(tmp_path / "store")
    store.initialize(hash_algorithm="blake3", prefix_length=1)
    key = store.put_store(b"payload")

    assert len(key) == 64
    assert store.get_store(key) == b"payload"


def test_prefix_routing_for_aa_with_prefix_length_two(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=2)

    key = "aa" + "0" * 62

    assert store._prefix_dir_for_key(key) == store.root_path / "aa"


def test_rotation_by_row_count_and_retrieval_from_older_db(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1, max_rows=1)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_store(first_payload)
    second = store.put_store(second_payload)
    first_prefix = first[:1]

    files = store._sqlite_files_for_prefix(first_prefix)

    assert len(files) == 2
    assert store.get_store(first) == first_payload
    assert store.get_store(second) == second_payload


def test_rotation_by_file_size(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1, max_db_bytes=1)

    key = store.put_store(b"large enough")

    assert len(store._sqlite_files_for_prefix(key[:1])) >= 2


def test_key_exists_finds_keys_in_older_rotated_files(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1, max_rows=1)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_store(first_payload)
    store.put_store(second_payload)

    assert store.key_exists(first) is True


def test_key_delete_removes_keys_from_older_rotated_files(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1, max_rows=1)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_store(first_payload)
    second = store.put_store(second_payload)
    store.key_delete(first)

    assert store.key_exists(first) is False
    assert store.get_store(second) == second_payload


def test_idempotent_writes_detect_existing_keys_across_rotated_files(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1, max_rows=1)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_store(first_payload)
    store.put_store(second_payload)
    again = store.put_store(first_payload)

    assert again == first


def test_sqlite_filenames_increment_monotonically(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1, max_db_bytes=1)

    key = store.put_store(b"one")
    store.put_store(b"two")
    indexes = [
        store._sqlite_index_from_filename(path.name)
        for path in store._sqlite_files_for_prefix(key[:1])
    ]

    assert indexes == sorted(indexes)
    assert indexes[0] == 1


def test_health_detects_invalid_rotated_sqlite_filename(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1)
    bad_path = store.root_path / "a" / "store_bad.sqlite"
    bad_path.write_bytes(b"not sqlite")

    health = store.check_health()

    assert health["healthy"] is False
    assert any("Invalid SQLite filename" in error for error in health["errors"])


def test_status_reports_sqlite_file_count_after_rotation(tmp_path):
    store = KVStore(tmp_path / "store")
    store.initialize(prefix_length=1, max_db_bytes=1)

    store.put_store(b"one")
    store.put_store(b"two")

    assert store.status()["total_sqlite_files"] >= 18
