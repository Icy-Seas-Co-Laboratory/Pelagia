from __future__ import annotations

import gc
import hashlib
import io
import json
import os
import sqlite3
import threading
import weakref

import pytest

from Pelagia.storage.kvstore2 import (
    KVStore2,
    KVStore2AlreadyInitializedError,
    KVStore2ConfigError,
    KVStore2IntegrityError,
    KVStore2LockError,
    KVStore2Error,
    MAX_CACHED_CONNECTIONS_PER_THREAD,
    RECORD_HEADER_PREFIX,
    RECORD_HEADER_VERSION,
    RECORD_MAGIC,
)


def test_connections_reuse_and_roll_back_on_error(tmp_path):
    store = KVStore2(tmp_path / "store")
    path = store.root_path / "index.sqlite"
    with store._connect(path) as first:
        first.execute("CREATE TABLE example (value INTEGER)")
    with pytest.raises(ValueError), store._connect(path) as second:
        assert second is first
        second.execute("INSERT INTO example VALUES (1)")
        raise ValueError("rollback")
    with store._connect(path) as connection:
        assert connection is first
        assert connection.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 0
    store.close()
    with pytest.raises(sqlite3.ProgrammingError):
        first.execute("SELECT 1")
    with pytest.raises(KVStore2Error, match="closed"):
        with store._connect(path):
            pass


def test_connection_cache_evicts_least_recently_used(tmp_path):
    store = KVStore2(tmp_path / "store")
    paths = [store.root_path / f"{index}.sqlite" for index in range(MAX_CACHED_CONNECTIONS_PER_THREAD + 1)]
    with store._connect(paths[0]) as oldest:
        pass
    for path in paths[1:]:
        with store._connect(path):
            pass
    with pytest.raises(sqlite3.ProgrammingError):
        oldest.execute("SELECT 1")
    store.close()


def test_connections_are_thread_isolated_and_closed_by_store(tmp_path):
    store = KVStore2(tmp_path / "store")
    path = store.root_path / "index.sqlite"
    handles = []
    ready = threading.Barrier(3)

    def worker():
        with store._connect(path) as connection:
            handles.append(connection)
            ready.wait()
        ready.wait()

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    ready.wait()
    assert handles[0] is not handles[1]
    ready.wait()
    for thread in threads:
        thread.join()
    store.close()
    for connection in handles:
        with pytest.raises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


@pytest.mark.parametrize("cleanup", ["close", "next_acquisition"])
def test_finished_thread_cache_is_explicitly_closed(tmp_path, cleanup):
    store = KVStore2(tmp_path / "store")
    handles = []

    def worker():
        with store._connect(store.root_path / "worker.sqlite") as connection:
            handles.append(connection)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    thread_ref = weakref.ref(thread)
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        del thread
        # The store retains the thread's cache until it can close its connections.
        assert thread_ref() is not None
        if cleanup == "close":
            store.close()
        else:
            with store._connect(store.root_path / "main.sqlite"):
                pass
        with pytest.raises(sqlite3.ProgrammingError):
            handles[0].execute("SELECT 1")
        assert thread_ref() is None
    finally:
        store.close()
        if gc_was_enabled:
            gc.enable()


def test_nested_connections_are_distinct_and_reset_reopens_indexes(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    path = store.root_path / "0" / "0-index.sqlite"
    with store._connect(path) as outer:
        with store._connect(path) as inner:
            assert inner is not outer
    key = store.put_store(b"before reset")
    store.reset(prefix_length=1)
    assert not store.key_exists(key)
    assert store.get_store(store.put_store(b"after reset")) == b"after reset"
    store.close()


def test_replaced_index_is_reopened(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    key = store.put_store(b"original")
    assert store.key_exists(key)
    other = KVStore2(store.root_path)
    other.reset(prefix_length=1)
    assert not store.key_exists(key)
    store.close()
    other.close()


def test_close_waits_for_active_connection(tmp_path):
    store = KVStore2(tmp_path / "store")
    acquired = threading.Event()
    release = threading.Event()
    closed = threading.Event()

    def worker():
        with store._connect(store.root_path / "index.sqlite") as connection:
            acquired.set()
            assert release.wait(5)
            connection.execute("SELECT 1")

    thread = threading.Thread(target=worker)
    thread.start()
    assert acquired.wait(5)
    closer = threading.Thread(target=lambda: (store.close(), closed.set()))
    closer.start()
    assert not closed.wait(0.05)
    release.set()
    thread.join(5)
    closer.join(5)
    assert closed.is_set()


@pytest.mark.parametrize("blocked_phase", ["open", "eviction_close"])
def test_blocked_sqlite_work_does_not_block_other_prefix_and_close_waits(
    tmp_path, monkeypatch, blocked_phase
):
    from Pelagia.storage import kvstore2

    store = KVStore2(tmp_path / "store")
    blocked_path = store.root_path / "blocked.sqlite"
    entered = threading.Event()
    release = threading.Event()
    other_done = threading.Event()
    closed = threading.Event()
    errors = []
    original_connect = sqlite3.connect

    class BlockingConnection(sqlite3.Connection):
        def close(self):
            entered.set()
            if not release.wait(5):
                raise TimeoutError("blocked close was not released")
            super().close()

    def connect(path, **kwargs):
        if path == blocked_path:
            if blocked_phase == "open":
                entered.set()
                if not release.wait(5):
                    raise TimeoutError("blocked open was not released")
            else:
                kwargs["factory"] = BlockingConnection
        return original_connect(path, **kwargs)

    monkeypatch.setattr(kvstore2.sqlite3, "connect", connect)
    monkeypatch.setattr(kvstore2, "MAX_CACHED_CONNECTIONS_PER_THREAD", 1)

    def blocked_worker():
        try:
            with store._connect(blocked_path):
                pass
            if blocked_phase == "eviction_close":
                with store._connect(store.root_path / "evict.sqlite"):
                    pass
        except BaseException as exc:
            errors.append(exc)

    def other_worker():
        try:
            with store._connect(store.root_path / "other.sqlite"):
                pass
            other_done.set()
        except BaseException as exc:
            errors.append(exc)

    def close_store():
        try:
            store.close()
            closed.set()
        except BaseException as exc:
            errors.append(exc)

    blocked = threading.Thread(target=blocked_worker)
    other = threading.Thread(target=other_worker)
    closer = threading.Thread(target=close_store)
    blocked.start()
    try:
        assert entered.wait(5)
        other.start()
        assert other_done.wait(2), "unrelated connection was blocked"
        closer.start()
        assert not closed.wait(0.05), "close returned before detached work finished"
    finally:
        release.set()
        blocked.join(5)
        if other.ident is not None:
            other.join(5)
        if closer.ident is not None:
            closer.join(5)
        else:
            store.close()
    assert not blocked.is_alive()
    assert not other.is_alive()
    assert not closer.is_alive()
    assert closed.is_set()
    assert not errors


def test_context_close_closes_project_store_connections(tmp_path):
    from Pelagia.config import CoreConfig
    from Pelagia.services.context import AppContext

    store = KVStore2(tmp_path / "store")
    with store._connect(store.root_path / "index.sqlite") as connection:
        pass
    context = AppContext(config=CoreConfig(), kvstore=store, _project_kvstores={"p": store})
    context.close()
    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


@pytest.mark.skipif(not hasattr(os, "fork"), reason="requires fork")
def test_forked_child_does_not_reuse_parent_connection(tmp_path):
    store = KVStore2(tmp_path / "store")
    path = store.root_path / "index.sqlite"
    with store._connect(path) as parent_connection:
        parent_connection.execute("CREATE TABLE example (value INTEGER)")
    child = os.fork()
    if child == 0:
        try:
            with store._connect(path) as connection:
                assert connection is not parent_connection
                connection.execute("INSERT INTO example VALUES (1)")
            store.close()
        except BaseException:
            os._exit(1)
        os._exit(0)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    with store._connect(path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM example").fetchone()[0] == 1
    store.close()


def sha256_key(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def next_power_of_two(value: int) -> int:
    if value <= 0:
        return 0
    return 1 << (value - 1).bit_length()


def payloads_with_same_prefix(count: int, prefix_length: int = 1) -> list[bytes]:
    buckets: dict[str, list[bytes]] = {}
    for index in range(100_000):
        payload = f"payload-{index}".encode("ascii")
        prefix = sha256_key(payload)[:prefix_length]
        buckets.setdefault(prefix, []).append(payload)
        if len(buckets[prefix]) == count:
            return buckets[prefix]
    raise AssertionError("Unable to find test payloads with matching prefix.")


def read_record_header(blob_path, header_offset: int):
    with blob_path.open("rb") as blob_file:
        blob_file.seek(header_offset)
        fixed = blob_file.read(RECORD_HEADER_PREFIX.size)
        magic, version, header_json_size = RECORD_HEADER_PREFIX.unpack(fixed)
        header = json.loads(blob_file.read(header_json_size).decode("utf-8"))
    return magic, version, RECORD_HEADER_PREFIX.size + header_json_size, header


def test_creating_new_store_leaves_uninitialized(tmp_path):
    store = KVStore2(tmp_path / "store")

    assert store.initialized is False


def test_initialize_creates_config_manifest_prefix_indexes_and_blob_files(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)

    assert store.initialized is True
    assert (store.root_path / "config.json").exists()
    assert (store.root_path / "manifest.json").exists()
    for prefix in "0123456789abcdef":
        assert (store.root_path / prefix).is_dir()
        assert (store.root_path / prefix / f"{prefix}-index.sqlite").exists()
        assert (store.root_path / prefix / f"{prefix}-shard_001.blob").exists()


def test_loading_existing_store_from_config(tmp_path):
    root = tmp_path / "store"
    KVStore2(root).initialize(prefix_length=1)

    loaded = KVStore2(root)

    assert loaded.initialized is True
    assert loaded.config is not None
    assert loaded.config["layout"] == "sqlite-index-blob-shard"
    assert loaded.config["prefix_length"] == 1


def test_loading_legacy_kvstore_root_reports_backend_mismatch(tmp_path):
    from Pelagia.storage.kvstore import KVStore

    root = tmp_path / "store"
    KVStore(root).initialize(prefix_length=1)

    with pytest.raises(KVStore2ConfigError, match="legacy KVStore config"):
        KVStore2(root)


def test_initialize_existing_store_without_overwrite_raises(tmp_path):
    root = tmp_path / "store"
    KVStore2(root).initialize(prefix_length=1)

    with pytest.raises(KVStore2AlreadyInitializedError):
        KVStore2(root).initialize(prefix_length=1)


def test_store_and_retrieve_bytes(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    payload = b"hello world"

    key = store.put_store(payload)

    assert key == sha256_key(payload)
    assert store.get_store(key) == payload


def test_store_and_retrieve_string_as_utf8_bytes(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)

    key = store.put_store("hello snow")

    assert store.get_store(key) == "hello snow".encode("utf-8")


def test_put_stream_stores_without_materializing_payload(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    payload = b"streamed payload" * 17

    key = store.put_stream(io.BytesIO(payload), chunk_size=5)

    assert key == sha256_key(payload)
    assert store.get_store(key) == payload


def test_get_stream_reads_only_indexed_payload_range(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    payload = b"bounded stream"
    key = store.put_store(payload)

    with store.get_stream(key) as reader:
        assert reader.read(3) == payload[:3]
        assert reader.read() == payload[3:]
        assert reader.read() == b""


def test_put_stream_idempotent_write_does_not_append_again(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    payload = b"same stream payload"

    first = store.put_stream(io.BytesIO(payload), chunk_size=4)
    second = store.put_stream(io.BytesIO(payload), chunk_size=3)
    prefix = first[:1]

    with sqlite3.connect(store.root_path / prefix / f"{prefix}-index.sqlite") as connection:
        row = connection.execute(
            "SELECT COUNT(*), COALESCE(MAX(offset + size), 0) FROM blobs"
        ).fetchone()

    assert first == second
    assert row[0] == 1
    assert row[1] > len(payload)


def test_put_stream_rotation_is_per_prefix(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1, max_blob_bytes=256)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_stream(io.BytesIO(first_payload), chunk_size=3)
    second = store.put_stream(io.BytesIO(second_payload), chunk_size=4)
    prefix = first[:1]

    assert second[:1] == prefix
    assert store.get_store(first) == first_payload
    assert store.get_store(second) == second_payload
    assert (store.root_path / prefix / f"{prefix}-shard_002.blob").exists()


def test_index_records_offset_and_size_for_direct_reads(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_store(first_payload)
    second = store.put_store(second_payload)
    prefix = first[:1]

    with sqlite3.connect(store.root_path / prefix / f"{prefix}-index.sqlite") as connection:
        rows = connection.execute(
            """
            SELECT key, header_offset, header_size, offset, size
            FROM blobs
            ORDER BY header_offset
            """
        ).fetchall()
        state = connection.execute(
            "SELECT data_end, physical_size FROM shard_state WHERE shard_index = 1"
        ).fetchone()

    assert [row[0] for row in rows] == [first, second]
    first_header_offset, first_header_size, first_offset, first_size = rows[0][1:]
    second_header_offset, second_header_size, second_offset, second_size = rows[1][1:]
    assert (first_header_offset, first_offset, first_size) == (
        0,
        first_header_size,
        len(first_payload),
    )
    assert (second_header_offset, second_offset, second_size) == (
        first_header_size + len(first_payload),
        second_header_offset + second_header_size,
        len(second_payload),
    )
    data_end = second_header_offset + second_header_size + len(second_payload)
    assert state == (
        data_end,
        next_power_of_two(data_end),
    )


def test_blob_record_includes_magic_header(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    payload = b"headered payload"

    key = store.put_store(payload)
    prefix = key[:1]
    blob_path = store.root_path / prefix / f"{prefix}-shard_001.blob"
    with sqlite3.connect(store.root_path / prefix / f"{prefix}-index.sqlite") as connection:
        row = connection.execute(
            """
            SELECT header_offset, header_size, offset, size
            FROM blobs
            WHERE key = ?
            """,
            (key,),
        ).fetchone()

    magic, version, header_size, header = read_record_header(blob_path, row[0])

    assert magic == RECORD_MAGIC
    assert version == RECORD_HEADER_VERSION
    assert header_size == row[1]
    assert row[2] == row[0] + row[1]
    assert row[3] == len(payload)
    assert header["key"] == key
    assert header["payload_size"] == len(payload)
    assert header["checksum"] == key


def test_blob_file_is_power_of_two_padded_and_offsets_ignore_padding(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_store(first_payload)
    store.put_store(second_payload)
    prefix = first[:1]
    blob_path = store.root_path / prefix / f"{prefix}-shard_001.blob"

    with sqlite3.connect(store.root_path / prefix / f"{prefix}-index.sqlite") as connection:
        data_end = connection.execute(
            "SELECT data_end FROM shard_state WHERE shard_index = 1"
        ).fetchone()[0]

    assert blob_path.stat().st_size == next_power_of_two(data_end)
    assert store.get_store(first) == first_payload


def test_blob_rotation_is_per_prefix(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1, max_blob_bytes=256)
    first_payload, second_payload = payloads_with_same_prefix(2)

    first = store.put_store(first_payload)
    second = store.put_store(second_payload)
    prefix = first[:1]

    assert second[:1] == prefix
    assert (store.root_path / prefix / f"{prefix}-shard_001.blob").stat().st_size == 256
    assert (store.root_path / prefix / f"{prefix}-shard_002.blob").stat().st_size == 256
    with sqlite3.connect(store.root_path / prefix / f"{prefix}-index.sqlite") as connection:
        rows = connection.execute(
            "SELECT key, shard_index, offset, size FROM blobs ORDER BY shard_index, offset"
        ).fetchall()
        states = connection.execute(
            "SELECT shard_index, data_end, physical_size FROM shard_state ORDER BY shard_index"
        ).fetchall()

    first_data_end = states[0][1]
    second_data_end = states[1][1]
    assert rows == [
        (first, 1, rows[0][2], len(first_payload)),
        (second, 2, rows[1][2], len(second_payload)),
    ]
    assert first_data_end > len(first_payload)
    assert second_data_end > len(second_payload)
    assert states[0][2] == 256
    assert states[1][2] == 256
    assert store.get_store(first) == first_payload
    assert store.get_store(second) == second_payload


def test_idempotent_write_does_not_append_again(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    payload = b"same"

    first = store.put_store(payload)
    second = store.put_store(payload)
    prefix = first[:1]

    with sqlite3.connect(store.root_path / prefix / f"{prefix}-index.sqlite") as connection:
        row = connection.execute(
            "SELECT COUNT(*), COALESCE(MAX(offset + size), 0) FROM blobs"
        ).fetchone()
        state = connection.execute(
            "SELECT data_end FROM shard_state WHERE shard_index = 1"
        ).fetchone()

    assert first == second
    assert row[0] == 1
    assert row[1] == state[0]
    assert state[0] > len(payload)


def test_key_exists_true_and_false(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)

    key = store.put_store(b"present")

    assert store.key_exists(key) is True
    assert store.key_exists(sha256_key(b"missing")) is False


def test_key_delete_removes_index_entry_but_keeps_blob_capacity(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    payload = b"present"
    key = store.put_store(payload)
    prefix = key[:1]
    blob_path = store.root_path / prefix / f"{prefix}-shard_001.blob"
    size_before = blob_path.stat().st_size

    store.key_delete(key)

    assert store.key_exists(key) is False
    assert blob_path.stat().st_size == size_before
    with pytest.raises(KeyError):
        store.get_store(key)


def test_key_delete_missing_raises_key_error(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)

    with pytest.raises(KeyError):
        store.key_delete(sha256_key(b"missing"))


def test_get_store_missing_raises_key_error(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)

    with pytest.raises(KeyError):
        store.get_store(sha256_key(b"missing"))


def test_prefix_write_lock_timeout(tmp_path):
    store = KVStore2(
        tmp_path / "store",
        lock_timeout_s=0.01,
        lock_poll_interval_s=0.001,
        stale_lock_seconds=None,
    )
    store.initialize(prefix_length=1)
    payload = b"locked payload"
    lock_path = store.root_path / sha256_key(payload)[:1] / ".kvstore.lock"
    lock_path.write_text("held by test", encoding="utf-8")

    with pytest.raises(KVStore2LockError):
        store.put_store(payload)


def test_stale_prefix_lock_is_removed(tmp_path):
    store = KVStore2(
        tmp_path / "store",
        lock_timeout_s=0.5,
        lock_poll_interval_s=0.001,
        stale_lock_seconds=0,
    )
    store.initialize(prefix_length=1)
    payload = b"stale lock payload"
    lock_path = store.root_path / sha256_key(payload)[:1] / ".kvstore.lock"
    lock_path.write_text("stale", encoding="utf-8")
    os.utime(lock_path, (0, 0))

    key = store.put_store(payload)

    assert store.get_store(key) == payload
    assert not lock_path.exists()


def test_status_reports_counts_and_padding(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    first_payload, second_payload = payloads_with_same_prefix(2)
    store.put_store(first_payload)
    store.put_store(second_payload)

    status = store.status()

    assert status["initialized"] is True
    assert status["layout"] == "sqlite-index-blob-shard"
    assert status["prefix_directory_count"] == 16
    assert status["total_index_files"] == 16
    assert status["total_index_file_bytes"] > 0
    assert status["total_blob_files"] == 16
    assert status["total_file_bytes"] == status["total_index_file_bytes"] + status["total_blob_file_bytes"]
    assert status["total_stored_blobs"] == 2
    assert status["total_stored_payload_bytes"] == len(first_payload) + len(second_payload)
    assert status["total_record_header_bytes"] > 0
    assert status["total_record_bytes"] == (
        status["total_record_header_bytes"] + status["total_stored_payload_bytes"]
    )
    assert status["total_data_end_bytes"] == status["total_record_bytes"]
    assert status["total_padding_bytes"] >= 0


def test_check_health_passes_for_valid_store(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    store.put_store(b"payload")

    health = store.check_health(verify_payloads=True)

    assert health["healthy"] is True
    assert health["errors"] == []


def test_check_health_reports_missing_blob_file(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    prefix = "a"
    (store.root_path / prefix / f"{prefix}-shard_001.blob").unlink()

    health = store.check_health()

    assert health["healthy"] is False
    assert any("Missing blob file" in error for error in health["errors"])


def test_check_health_reports_non_power_of_two_blob_size(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    key = store.put_store(b"abc")
    prefix = key[:1]
    blob_path = store.root_path / prefix / f"{prefix}-shard_001.blob"
    blob_path.write_bytes(b"abcde")

    health = store.check_health()

    assert health["healthy"] is False
    assert any("not a power of two" in error for error in health["errors"])


def test_verify_payloads_detects_corruption(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    key = store.put_store(b"original")
    prefix = key[:1]
    blob_path = store.root_path / prefix / f"{prefix}-shard_001.blob"
    with sqlite3.connect(store.root_path / prefix / f"{prefix}-index.sqlite") as connection:
        offset = connection.execute(
            "SELECT offset FROM blobs WHERE key = ?",
            (key,),
        ).fetchone()[0]

    with blob_path.open("r+b") as blob_file:
        blob_file.seek(offset)
        blob_file.write(b"changed!")

    health = store.check_health(verify_payloads=True)

    assert health["healthy"] is False
    assert any("Payload hash mismatch" in error for error in health["errors"])


def test_verify_payloads_detects_invalid_magic_header(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    key = store.put_store(b"original")
    prefix = key[:1]
    blob_path = store.root_path / prefix / f"{prefix}-shard_001.blob"

    with blob_path.open("r+b") as blob_file:
        blob_file.seek(0)
        blob_file.write(b"changed!")

    health = store.check_health(verify_payloads=True)

    assert health["healthy"] is False
    assert any("invalid magic header" in error for error in health["errors"])


def test_get_store_raises_integrity_error_for_truncated_blob(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1)
    key = store.put_store(b"payload")
    prefix = key[:1]
    blob_path = store.root_path / prefix / f"{prefix}-shard_001.blob"
    with blob_path.open("r+b") as blob_file:
        blob_file.truncate(2)

    with pytest.raises(KVStore2IntegrityError):
        store.get_store(key)


def test_max_blob_bytes_limit_is_enforced(tmp_path):
    store = KVStore2(tmp_path / "store")
    store.initialize(prefix_length=1, max_blob_bytes=4)

    with pytest.raises(KVStore2ConfigError):
        store.put_store(b"12345")
