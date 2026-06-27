from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import struct
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO, Iterable, Iterator


DEFAULT_MAX_BLOB_BYTES = 64 * 1024 * 1024
CONFIG_FILENAME = "config.json"
LOCK_FILENAME = ".kvstore.lock"
MANIFEST_FILENAME = "manifest.json"
SCHEMA_VERSION = 2
LAYOUT_NAME = "sqlite-index-blob-shard"
SUPPORTED_HASHES = {"sha256": 64, "blake3": 64}
HEX_RE = re.compile(r"^[0-9a-f]+$")
ZERO_FILL_CHUNK_BYTES = 8 * 1024 * 1024
RECORD_MAGIC = b"PKV2REC1"
RECORD_HEADER_VERSION = 1
RECORD_HEADER_PREFIX = struct.Struct(">8sHI")


class KVStore2Error(Exception):
    """Base exception for KVStore2 errors."""


class KVStore2NotInitializedError(KVStore2Error):
    """Raised when an operation requires an initialized store."""


class KVStore2AlreadyInitializedError(KVStore2Error):
    """Raised when initialization would overwrite an existing store."""


class KVStore2ConfigError(KVStore2Error):
    """Raised when store configuration is missing or invalid."""


class KVStore2IntegrityError(KVStore2Error):
    """Raised when stored or supplied data violates content-addressing rules."""


class KVStore2LockError(KVStore2Error):
    """Raised when a store lock cannot be acquired."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_root_path(root_path: str | Path) -> Path:
    raw_path = os.fspath(root_path)
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    return Path(expanded).resolve(strict=False)


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        return 0
    return 1 << (value - 1).bit_length()


def _is_power_of_two(value: int) -> bool:
    return value == 0 or (value > 0 and value & (value - 1) == 0)


class _FileLock:
    """Small cross-platform lock based on atomic lock-file creation."""

    def __init__(
        self,
        lock_path: Path,
        *,
        timeout_s: float,
        poll_interval_s: float,
        stale_after_s: float | None,
    ):
        self.lock_path = lock_path
        self.timeout_s = timeout_s
        self.poll_interval_s = poll_interval_s
        self.stale_after_s = stale_after_s
        self._acquired = False

    def __enter__(self) -> "_FileLock":
        deadline = time.monotonic() + self.timeout_s
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

        while True:
            try:
                fd = os.open(
                    self.lock_path,
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                self._remove_if_stale()
                if time.monotonic() >= deadline:
                    raise KVStore2LockError(f"Timed out waiting for lock: {self.lock_path}")
                time.sleep(self.poll_interval_s)
                continue

            with os.fdopen(fd, "w", encoding="utf-8") as lock_file:
                json.dump(
                    {
                        "pid": os.getpid(),
                        "created_at": _utc_now(),
                        "lock_path": str(self.lock_path),
                    },
                    lock_file,
                    sort_keys=True,
                )
                lock_file.write("\n")
            self._acquired = True
            return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if not self._acquired:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass
        finally:
            self._acquired = False

    def _remove_if_stale(self) -> None:
        if self.stale_after_s is None:
            return
        try:
            age_s = time.time() - self.lock_path.stat().st_mtime
        except FileNotFoundError:
            return
        if age_s < self.stale_after_s:
            return
        try:
            self.lock_path.unlink()
        except FileNotFoundError:
            pass


class _BoundedBlobReader:
    """Read only a single indexed payload range from a blob file."""

    def __init__(self, blob_file: BinaryIO, *, key: str, remaining: int):
        self._blob_file = blob_file
        self._key = key
        self._remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self._remaining <= 0:
            return b""
        if size is None or size < 0 or size > self._remaining:
            size = self._remaining
        chunk = self._blob_file.read(size)
        if len(chunk) != size:
            raise KVStore2IntegrityError(
                f"Blob record {self._key} expected {self._remaining} more bytes, "
                f"read {len(chunk)} bytes."
            )
        self._remaining -= len(chunk)
        return chunk

    def readable(self) -> bool:
        return True

    def close(self) -> None:
        self._blob_file.close()

    @property
    def closed(self) -> bool:
        return self._blob_file.closed


class KVStore2:
    """A content-addressed blob store with per-prefix SQLite indexes.

    Blob bytes live in append-only shard files. Each prefix SQLite database stores
    the key, blob shard, offset, and payload size needed for direct reads.
    """

    def __init__(
        self,
        root_path: str | Path,
        *,
        lock_timeout_s: float = 30.0,
        lock_poll_interval_s: float = 0.05,
        stale_lock_seconds: float | None = 60 * 60,
    ):
        """Create a store handle for ``root_path`` without implicitly initializing it."""
        self.root_path = _normalize_root_path(root_path)
        self.config_path = self.root_path / CONFIG_FILENAME
        self.config: dict[str, Any] | None = None
        self.initialized = False
        self.lock_timeout_s = lock_timeout_s
        self.lock_poll_interval_s = lock_poll_interval_s
        self.stale_lock_seconds = stale_lock_seconds

        if self.config_path.exists():
            self.config = self._load_config()
            self.initialized = True

    def initialize(
        self,
        hash_algorithm: str = "sha256",
        prefix_length: int = 2,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
        overwrite: bool = False,
    ) -> None:
        """Initialize a new KVStore2 on disk."""
        hash_algorithm = hash_algorithm.lower()
        self._validate_hash_algorithm(hash_algorithm)
        self._validate_prefix_length(prefix_length)
        self._validate_max_blob_bytes(max_blob_bytes)

        self.root_path.mkdir(parents=True, exist_ok=True)
        with self._lock_for_path(self.root_path / LOCK_FILENAME):
            content_paths = [
                path for path in self.root_path.iterdir()
                if path.name != LOCK_FILENAME
            ]
            if content_paths:
                if not overwrite:
                    raise KVStore2AlreadyInitializedError(
                        f"Store path is not empty: {self.root_path}"
                    )
                for path in content_paths:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

            created_at = _utc_now()
            self.config = {
                "version": SCHEMA_VERSION,
                "layout": LAYOUT_NAME,
                "hash_algorithm": hash_algorithm,
                "prefix_length": prefix_length,
                "key_length": SUPPORTED_HASHES[hash_algorithm],
                "manifest_filename": MANIFEST_FILENAME,
                "index_filename_pattern": "{prefix}-index.sqlite",
                "blob_filename_pattern": "{prefix}-shard_{index:03d}.blob",
                "initial_shard_index": 1,
                "created_at": created_at,
                "padding": {
                    "strategy": "power_of_two",
                    "zero_fill": True,
                },
                "record_header": {
                    "magic": RECORD_MAGIC.decode("ascii"),
                    "version": RECORD_HEADER_VERSION,
                },
                "limits": {
                    "max_blob_bytes": max_blob_bytes,
                },
            }
            self.initialized = True
            self.config_path = self.root_path / CONFIG_FILENAME

            self._write_config()
            self._write_manifest(created_at=created_at)
            for prefix in self._expected_prefixes():
                self._initialize_prefix(prefix, created_at=created_at)

    def reset(
        self,
        hash_algorithm: str = "sha256",
        prefix_length: int = 2,
        max_blob_bytes: int = DEFAULT_MAX_BLOB_BYTES,
    ) -> dict[str, Any]:
        """Delete all stored payloads and reinitialize the store in place."""
        before = self.status()
        self.initialize(
            hash_algorithm=hash_algorithm,
            prefix_length=prefix_length,
            max_blob_bytes=max_blob_bytes,
            overwrite=True,
        )
        after = self.status()
        return {
            "root_path": str(self.root_path),
            "previous": before,
            "current": after,
        }

    def put_store(self, payload: bytes | bytearray | str) -> str:
        """Store ``payload`` and return its content-addressed key."""
        self._require_initialized()
        payload_bytes = self._normalize_payload(payload)
        key = self._hash_payload(payload_bytes)
        prefix = self._prefix_for_key(key)

        with self._lock_for_prefix(prefix):
            index_path = self._index_path_for_prefix(prefix)
            with self._connect(index_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    existing = self._fetch_blob_record(connection, key)
                    if existing is not None:
                        existing_payload = self._read_payload_from_record(prefix, existing)
                        if existing_payload != payload_bytes:
                            raise KVStore2IntegrityError(
                                f"Key {key} already exists with different payload bytes."
                            )
                        connection.commit()
                        return key

                    created_at = _utc_now()
                    header = self._record_header_bytes(
                        key=key,
                        payload_size=len(payload_bytes),
                        created_at=created_at,
                    )
                    record_size = len(header) + len(payload_bytes)
                    shard_index, state = self._active_shard_state(connection, prefix)
                    new_physical_size = self._planned_physical_size(
                        prefix,
                        shard_index,
                        state,
                        record_size,
                    )
                    if new_physical_size > self._max_blob_bytes() and int(state["data_end"]) > 0:
                        shard_index = int(state["shard_index"]) + 1
                        state = self._fetch_shard_state(connection, shard_index)
                        if state is None:
                            state = self._create_shard_state(connection, prefix, shard_index)
                        new_physical_size = self._planned_physical_size(
                            prefix,
                            shard_index,
                            state,
                            record_size,
                        )
                    self._validate_blob_limit(new_physical_size)

                    state = self._fetch_shard_state(connection, shard_index)
                    if state is None:
                        state = self._create_shard_state(connection, prefix, shard_index)
                    data_end = int(state["data_end"])
                    header_offset = data_end
                    offset = header_offset + len(header)
                    new_data_end = header_offset + record_size

                    blob_path = self._blob_path_for_prefix(prefix, shard_index)
                    self._write_payload(
                        blob_path,
                        header_offset,
                        header,
                        payload_bytes,
                        new_physical_size,
                    )
                    connection.execute(
                        """
                        INSERT INTO blobs (
                            key, shard_index, header_offset, header_size,
                            offset, size, checksum, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            shard_index,
                            header_offset,
                            len(header),
                            offset,
                            len(payload_bytes),
                            key,
                            created_at,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE shard_state
                        SET data_end = ?, physical_size = ?, updated_at = ?
                        WHERE shard_index = ?
                        """,
                        (new_data_end, new_physical_size, created_at, shard_index),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise
        return key

    def put_stream(self, payload_stream: BinaryIO, *, chunk_size: int = 1024 * 1024) -> str:
        """Store bytes from ``payload_stream`` without materializing them in memory."""
        self._require_initialized()
        if chunk_size <= 0:
            raise KVStore2ConfigError("chunk_size must be a positive integer.")

        temp_dir = self.root_path / ".tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path: Path | None = None
        try:
            hasher = self._new_hasher()
            total_size = 0
            with tempfile.NamedTemporaryFile(dir=temp_dir, delete=False) as temp_file:
                temp_path = Path(temp_file.name)
                while True:
                    chunk = payload_stream.read(chunk_size)
                    if chunk == b"":
                        break
                    if isinstance(chunk, str):
                        raise TypeError("payload_stream must yield bytes.")
                    if not isinstance(chunk, (bytes, bytearray)):
                        raise TypeError("payload_stream must yield bytes.")
                    chunk_bytes = bytes(chunk)
                    hasher.update(chunk_bytes)
                    total_size += len(chunk_bytes)
                    temp_file.write(chunk_bytes)
                temp_file.flush()
                os.fsync(temp_file.fileno())

            key = hasher.hexdigest()
            self._put_file_payload(key, temp_path, total_size)
            return key
        finally:
            if temp_path is not None:
                try:
                    temp_path.unlink()
                except FileNotFoundError:
                    pass

    def get_store(self, key: str) -> bytes:
        """Return payload bytes for ``key`` or raise ``KeyError`` if it is missing."""
        self._require_initialized()
        self._validate_key(key)
        prefix = self._prefix_for_key(key)
        with self._connect(self._index_path_for_prefix(prefix)) as connection:
            record = self._fetch_blob_record(connection, key)
        if record is None:
            raise KeyError(key)
        return self._read_payload_from_record(prefix, record)

    @contextmanager
    def get_stream(self, key: str) -> Iterator[_BoundedBlobReader]:
        """Yield a bounded binary reader for ``key``."""
        self._require_initialized()
        self._validate_key(key)
        prefix = self._prefix_for_key(key)
        with self._connect(self._index_path_for_prefix(prefix)) as connection:
            record = self._fetch_blob_record(connection, key)
        if record is None:
            raise KeyError(key)

        blob_path = self._blob_path_for_prefix(prefix, int(record["shard_index"]))
        try:
            with blob_path.open("rb") as blob_file:
                self._read_record_header(blob_file, record)
                blob_file.seek(int(record["offset"]))
                yield _BoundedBlobReader(
                    blob_file,
                    key=key,
                    remaining=int(record["size"]),
                )
        except FileNotFoundError as exc:
            raise KVStore2IntegrityError(f"Missing blob file: {blob_path}") from exc

    def key_exists(self, key: str) -> bool:
        """Return whether ``key`` exists in the prefix index."""
        self._require_initialized()
        self._validate_key(key)
        prefix = self._prefix_for_key(key)
        with self._connect(self._index_path_for_prefix(prefix)) as connection:
            return self._fetch_blob_record(connection, key) is not None

    def key_delete(self, key: str) -> None:
        """Delete the index entry for ``key`` without reclaiming blob bytes."""
        self._require_initialized()
        self._validate_key(key)
        prefix = self._prefix_for_key(key)
        deleted = False

        with self._lock_for_prefix(prefix):
            with self._connect(self._index_path_for_prefix(prefix)) as connection:
                cursor = connection.execute("DELETE FROM blobs WHERE key = ?", (key,))
                deleted = cursor.rowcount > 0

        if not deleted:
            raise KeyError(key)

    def status(self, *, deep: bool = True) -> dict[str, Any]:
        """Return counts, byte totals, and paths for the store."""
        if not self.initialized or self.config is None:
            return {
                "root_path": str(self.root_path),
                "initialized": False,
                "config_path": str(self.config_path),
                "manifest_path": str(self.root_path / MANIFEST_FILENAME),
            }

        total_index_files = 0
        total_index_file_bytes = 0
        total_blob_files = 0
        total_blob_file_bytes = 0
        total_data_end_bytes = 0
        total_blobs = 0
        total_header_bytes = 0
        total_payload_bytes = 0
        largest_blob_file_size = 0

        for prefix in self._expected_prefixes():
            index_path = self._index_path_for_prefix(prefix)
            if index_path.exists():
                total_index_files += 1
                total_index_file_bytes += self._db_file_size(index_path)
                if deep:
                    with self._connect(index_path) as connection:
                        row = connection.execute(
                            "SELECT COUNT(*), COALESCE(SUM(size), 0) FROM blobs"
                        ).fetchone()
                        total_blobs += int(row[0])
                        total_payload_bytes += int(row[1])
                        header_row = connection.execute(
                            "SELECT COALESCE(SUM(header_size), 0) FROM blobs"
                        ).fetchone()
                        total_header_bytes += int(header_row[0])
                        state_rows = connection.execute(
                            "SELECT data_end, physical_size FROM shard_state"
                        ).fetchall()
                    for row in state_rows:
                        total_data_end_bytes += int(row[0])

            for blob_path in self._blob_files_for_prefix(prefix):
                total_blob_files += 1
                size = self._db_file_size(blob_path)
                total_blob_file_bytes += size
                largest_blob_file_size = max(largest_blob_file_size, size)

        prefix_length = int(self.config["prefix_length"])
        status = {
            "root_path": str(self.root_path),
            "initialized": True,
            "version": self.config["version"],
            "layout": self.config["layout"],
            "hash_algorithm": self.config["hash_algorithm"],
            "prefix_length": prefix_length,
            "prefix_directory_count": 16**prefix_length,
            "total_index_files": total_index_files,
            "total_index_file_bytes": total_index_file_bytes,
            "total_blob_files": total_blob_files,
            "total_blob_file_bytes": total_blob_file_bytes,
            "total_file_bytes": total_index_file_bytes + total_blob_file_bytes,
            "total_data_end_bytes": total_data_end_bytes,
            "largest_blob_file_size": largest_blob_file_size,
            "config_path": str(self.config_path),
            "manifest_path": str(self.root_path / self._manifest_filename()),
            "deep": deep,
        }
        if deep:
            status["total_stored_blobs"] = total_blobs
            status["total_record_header_bytes"] = total_header_bytes
            status["total_record_bytes"] = total_header_bytes + total_payload_bytes
            status["total_stored_payload_bytes"] = total_payload_bytes
            status["total_padding_bytes"] = max(0, total_blob_file_bytes - total_data_end_bytes)
        return status

    def check_health(self, *, verify_payloads: bool = False) -> dict[str, Any]:
        """Inspect configuration, prefix indexes, blob files, and optional payload hashes."""
        errors: list[str] = []
        warnings: list[str] = []

        if not self.config_path.exists():
            errors.append(f"Missing config file: {self.config_path}")

        try:
            config = self._load_config() if self.config_path.exists() else self.config
        except KVStore2Error as exc:
            config = self.config
            errors.append(str(exc))

        manifest_path = self.root_path / self._manifest_filename()
        if not manifest_path.exists():
            errors.append(f"Missing manifest file: {manifest_path}")

        if config is None:
            return {
                "healthy": False,
                "errors": errors or ["Store is not initialized."],
                "warnings": warnings,
                "checked_at": _utc_now(),
            }

        for prefix in self._expected_prefixes():
            prefix_dir = self._prefix_dir_for_prefix(prefix)
            if not prefix_dir.exists():
                errors.append(f"Missing prefix directory: {prefix_dir}")
                continue
            if not prefix_dir.is_dir():
                errors.append(f"Prefix path is not a directory: {prefix_dir}")
                continue

            index_path = self._index_path_for_prefix(prefix)
            if not index_path.exists():
                errors.append(f"Missing index database for prefix {prefix}: {index_path}")
                continue
            if not self._has_expected_schema(index_path):
                errors.append(f"SQLite index has unexpected schema: {index_path}")
                continue
            integrity = self._sqlite_integrity_check(index_path)
            if integrity != "ok":
                errors.append(f"SQLite integrity check failed for {index_path}: {integrity}")
                continue

            with self._connect(index_path) as connection:
                states = connection.execute(
                    "SELECT shard_index, blob_filename, data_end, physical_size FROM shard_state"
                ).fetchall()
                bounds = connection.execute(
                    """
                    SELECT shard_index, COALESCE(MAX(offset + size), 0), COUNT(*)
                    FROM blobs
                    GROUP BY shard_index
                    """
                ).fetchall()

            state_by_index = {int(row[0]): row for row in states}
            if self._initial_shard_index() not in state_by_index:
                errors.append(f"Prefix {prefix} is missing initial shard state.")

            for row in states:
                shard_index = int(row[0])
                blob_filename = str(row[1])
                data_end = int(row[2])
                physical_size = int(row[3])
                expected_blob = self._blob_filename(prefix, shard_index)
                if blob_filename != expected_blob:
                    errors.append(
                        f"Unexpected blob filename for prefix {prefix}: "
                        f"{blob_filename}; expected {expected_blob}"
                    )
                blob_path = prefix_dir / blob_filename
                if not blob_path.exists():
                    errors.append(f"Missing blob file for prefix {prefix}: {blob_path}")
                    continue
                actual_size = blob_path.stat().st_size
                if actual_size != physical_size:
                    errors.append(
                        f"Blob file size mismatch for {blob_path}: "
                        f"index={physical_size} actual={actual_size}"
                    )
                if not _is_power_of_two(actual_size):
                    errors.append(f"Blob file size is not a power of two: {blob_path}")
                if data_end > actual_size:
                    errors.append(f"Shard data_end exceeds blob size for {blob_path}")

            for row in bounds:
                shard_index = int(row[0])
                max_payload_end = int(row[1])
                count = int(row[2])
                state = state_by_index.get(shard_index)
                if state is None:
                    errors.append(
                        f"Prefix {prefix} has {count} blob rows without shard state {shard_index}."
                    )
                    continue
                if max_payload_end > int(state[2]):
                    errors.append(
                        f"Prefix {prefix} has blob rows beyond shard data_end "
                        f"for shard {shard_index}."
                    )

            if verify_payloads:
                errors.extend(self._verify_prefix_payloads(prefix))

        return {
            "healthy": not errors,
            "errors": errors,
            "warnings": warnings,
            "checked_at": _utc_now(),
            "verify_payloads": verify_payloads,
        }

    def _initialize_prefix(self, prefix: str, *, created_at: str) -> None:
        prefix_dir = self._prefix_dir_for_prefix(prefix)
        prefix_dir.mkdir(parents=True, exist_ok=True)
        blob_path = self._blob_path_for_prefix(prefix, self._initial_shard_index())
        blob_path.touch(exist_ok=True)
        with self._connect(self._index_path_for_prefix(prefix)) as connection:
            self._create_schema(connection)
            connection.execute(
                """
                INSERT OR IGNORE INTO shard_state (
                    shard_index, blob_filename, data_end, physical_size, created_at, updated_at
                )
                VALUES (?, ?, 0, 0, ?, ?)
                """,
                (
                    self._initial_shard_index(),
                    self._blob_filename(prefix, self._initial_shard_index()),
                    created_at,
                    created_at,
                ),
            )

    def _load_config(self) -> dict[str, Any]:
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KVStore2ConfigError(f"Invalid JSON config: {exc}") from exc

        self._validate_config(config)
        return config

    def _write_config(self) -> None:
        self._require_config()
        self.config_path.write_text(
            json.dumps(self.config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_manifest(self, *, created_at: str) -> None:
        self._require_config()
        prefix_length = int(self.config["prefix_length"])
        manifest = {
            "version": self.config["version"],
            "schema_version": SCHEMA_VERSION,
            "layout": self.config["layout"],
            "hash_algorithm": self.config["hash_algorithm"],
            "prefix_length": prefix_length,
            "shard_directory_count": 16**prefix_length,
            "index_filename_pattern": self.config["index_filename_pattern"],
            "blob_filename_pattern": self.config["blob_filename_pattern"],
            "record_header": dict(self.config["record_header"]),
            "padding": dict(self.config["padding"]),
            "limits": dict(self.config["limits"]),
            "created_at": created_at,
            "initialized": True,
        }
        (self.root_path / self._manifest_filename()).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS blobs (
                key TEXT PRIMARY KEY,
                shard_index INTEGER NOT NULL,
                header_offset INTEGER NOT NULL,
                header_size INTEGER NOT NULL,
                offset INTEGER NOT NULL,
                size INTEGER NOT NULL,
                checksum TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS shard_state (
                shard_index INTEGER PRIMARY KEY,
                blob_filename TEXT NOT NULL,
                data_end INTEGER NOT NULL,
                physical_size INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_blobs_shard_offset ON blobs (shard_index, offset)"
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('schema_version', ?)",
            (str(SCHEMA_VERSION),),
        )
        connection.execute(
            "INSERT OR REPLACE INTO metadata (key, value) VALUES ('layout', ?)",
            (LAYOUT_NAME,),
        )

    def _fetch_blob_record(
        self,
        connection: sqlite3.Connection,
        key: str,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT key, shard_index, header_offset, header_size, offset, size, checksum, created_at
            FROM blobs
            WHERE key = ?
            """,
            (key,),
        ).fetchone()
        if row is None:
            return None
        return {
            "key": str(row[0]),
            "shard_index": int(row[1]),
            "header_offset": int(row[2]),
            "header_size": int(row[3]),
            "offset": int(row[4]),
            "size": int(row[5]),
            "checksum": str(row[6]),
            "created_at": str(row[7]),
        }

    def _fetch_shard_state(
        self,
        connection: sqlite3.Connection,
        shard_index: int,
    ) -> dict[str, Any] | None:
        row = connection.execute(
            """
            SELECT shard_index, blob_filename, data_end, physical_size, created_at, updated_at
            FROM shard_state
            WHERE shard_index = ?
            """,
            (shard_index,),
        ).fetchone()
        if row is None:
            return None
        return {
            "shard_index": int(row[0]),
            "blob_filename": str(row[1]),
            "data_end": int(row[2]),
            "physical_size": int(row[3]),
            "created_at": str(row[4]),
            "updated_at": str(row[5]),
        }

    def _active_shard_state(
        self,
        connection: sqlite3.Connection,
        prefix: str,
    ) -> tuple[int, dict[str, Any]]:
        row = connection.execute(
            """
            SELECT shard_index, blob_filename, data_end, physical_size, created_at, updated_at
            FROM shard_state
            ORDER BY shard_index DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            shard_index = self._initial_shard_index()
            return shard_index, self._create_shard_state(connection, prefix, shard_index)
        state = {
            "shard_index": int(row[0]),
            "blob_filename": str(row[1]),
            "data_end": int(row[2]),
            "physical_size": int(row[3]),
            "created_at": str(row[4]),
            "updated_at": str(row[5]),
        }
        return int(row[0]), state

    def _create_shard_state(
        self,
        connection: sqlite3.Connection,
        prefix: str,
        shard_index: int,
    ) -> dict[str, Any]:
        created_at = _utc_now()
        blob_path = self._blob_path_for_prefix(prefix, shard_index)
        blob_path.touch(exist_ok=True)
        connection.execute(
            """
            INSERT INTO shard_state (
                shard_index, blob_filename, data_end, physical_size, created_at, updated_at
            )
            VALUES (?, ?, 0, 0, ?, ?)
            """,
            (shard_index, blob_path.name, created_at, created_at),
        )
        return {
            "shard_index": shard_index,
            "blob_filename": blob_path.name,
            "data_end": 0,
            "physical_size": 0,
            "created_at": created_at,
            "updated_at": created_at,
        }

    def _planned_physical_size(
        self,
        prefix: str,
        shard_index: int,
        state: dict[str, Any],
        append_size: int,
    ) -> int:
        new_data_end = int(state["data_end"]) + append_size
        return max(
            int(state["physical_size"]),
            self._actual_blob_file_size(prefix, shard_index),
            _next_power_of_two(new_data_end),
        )

    def _write_payload(
        self,
        blob_path: Path,
        header_offset: int,
        header: bytes,
        payload: bytes,
        physical_size: int,
    ) -> None:
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if blob_path.exists() else "w+b"
        with blob_path.open(mode) as blob_file:
            self._extend_blob_file(blob_file, physical_size)
            blob_file.seek(header_offset)
            blob_file.write(header)
            blob_file.write(payload)
            blob_file.flush()
            os.fsync(blob_file.fileno())

    def _record_header_bytes(self, *, key: str, payload_size: int, created_at: str) -> bytes:
        header = {
            "created_at": created_at,
            "key": key,
            "payload_size": payload_size,
            "schema_version": SCHEMA_VERSION,
            "checksum": key,
        }
        header_json = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return RECORD_HEADER_PREFIX.pack(
            RECORD_MAGIC,
            RECORD_HEADER_VERSION,
            len(header_json),
        ) + header_json

    def _read_record_header(self, blob_file: BinaryIO, record: dict[str, Any]) -> dict[str, Any]:
        blob_file.seek(int(record["header_offset"]))
        fixed = blob_file.read(RECORD_HEADER_PREFIX.size)
        if len(fixed) != RECORD_HEADER_PREFIX.size:
            raise KVStore2IntegrityError(f"Record {record['key']} has a truncated header prefix.")
        magic, version, header_json_size = RECORD_HEADER_PREFIX.unpack(fixed)
        if magic != RECORD_MAGIC:
            raise KVStore2IntegrityError(f"Record {record['key']} has an invalid magic header.")
        if version != RECORD_HEADER_VERSION:
            raise KVStore2IntegrityError(
                f"Record {record['key']} has unsupported header version {version}."
            )
        expected_header_size = RECORD_HEADER_PREFIX.size + header_json_size
        if expected_header_size != int(record["header_size"]):
            raise KVStore2IntegrityError(
                f"Record {record['key']} header size mismatch: "
                f"index={record['header_size']} actual={expected_header_size}."
            )
        header_json = blob_file.read(header_json_size)
        if len(header_json) != header_json_size:
            raise KVStore2IntegrityError(f"Record {record['key']} has a truncated JSON header.")
        try:
            header = json.loads(header_json.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise KVStore2IntegrityError(f"Record {record['key']} has an invalid JSON header.") from exc
        self._validate_record_header(record, header)
        return header

    def _validate_record_header(self, record: dict[str, Any], header: dict[str, Any]) -> None:
        if header.get("key") != record["key"]:
            raise KVStore2IntegrityError(f"Record header key mismatch for {record['key']}.")
        if int(header.get("payload_size", -1)) != int(record["size"]):
            raise KVStore2IntegrityError(f"Record header size mismatch for {record['key']}.")
        if header.get("checksum") != record["checksum"]:
            raise KVStore2IntegrityError(f"Record header checksum mismatch for {record['key']}.")
        if int(header.get("schema_version", -1)) != SCHEMA_VERSION:
            raise KVStore2IntegrityError(f"Record header schema mismatch for {record['key']}.")

    def _put_file_payload(self, key: str, payload_path: Path, payload_size: int) -> None:
        self._validate_key(key)
        prefix = self._prefix_for_key(key)

        with self._lock_for_prefix(prefix):
            index_path = self._index_path_for_prefix(prefix)
            with self._connect(index_path) as connection:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    existing = self._fetch_blob_record(connection, key)
                    if existing is not None:
                        if not self._file_payload_matches_record(prefix, existing, payload_path):
                            raise KVStore2IntegrityError(
                                f"Key {key} already exists with different payload bytes."
                            )
                        connection.commit()
                        return

                    created_at = _utc_now()
                    header = self._record_header_bytes(
                        key=key,
                        payload_size=payload_size,
                        created_at=created_at,
                    )
                    record_size = len(header) + payload_size
                    shard_index, state = self._active_shard_state(connection, prefix)
                    new_physical_size = self._planned_physical_size(
                        prefix,
                        shard_index,
                        state,
                        record_size,
                    )
                    if new_physical_size > self._max_blob_bytes() and int(state["data_end"]) > 0:
                        shard_index = int(state["shard_index"]) + 1
                        state = self._fetch_shard_state(connection, shard_index)
                        if state is None:
                            state = self._create_shard_state(connection, prefix, shard_index)
                        new_physical_size = self._planned_physical_size(
                            prefix,
                            shard_index,
                            state,
                            record_size,
                        )
                    self._validate_blob_limit(new_physical_size)

                    data_end = int(state["data_end"])
                    header_offset = data_end
                    offset = header_offset + len(header)
                    new_data_end = header_offset + record_size
                    blob_path = self._blob_path_for_prefix(prefix, shard_index)
                    self._write_payload_from_file(
                        blob_path,
                        header_offset,
                        header,
                        payload_path,
                        payload_size,
                        new_physical_size,
                    )
                    connection.execute(
                        """
                        INSERT INTO blobs (
                            key, shard_index, header_offset, header_size,
                            offset, size, checksum, created_at
                        )
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            key,
                            shard_index,
                            header_offset,
                            len(header),
                            offset,
                            payload_size,
                            key,
                            created_at,
                        ),
                    )
                    connection.execute(
                        """
                        UPDATE shard_state
                        SET data_end = ?, physical_size = ?, updated_at = ?
                        WHERE shard_index = ?
                        """,
                        (new_data_end, new_physical_size, created_at, shard_index),
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    def _write_payload_from_file(
        self,
        blob_path: Path,
        header_offset: int,
        header: bytes,
        payload_path: Path,
        payload_size: int,
        physical_size: int,
    ) -> None:
        blob_path.parent.mkdir(parents=True, exist_ok=True)
        mode = "r+b" if blob_path.exists() else "w+b"
        with payload_path.open("rb") as payload_file, blob_path.open(mode) as blob_file:
            self._extend_blob_file(blob_file, physical_size)
            blob_file.seek(header_offset)
            blob_file.write(header)
            remaining = payload_size
            while remaining > 0:
                chunk = payload_file.read(min(ZERO_FILL_CHUNK_BYTES, remaining))
                if not chunk:
                    raise KVStore2IntegrityError(
                        f"Stream payload ended early with {remaining} bytes unwritten."
                    )
                blob_file.write(chunk)
                remaining -= len(chunk)
            blob_file.flush()
            os.fsync(blob_file.fileno())

    def _file_payload_matches_record(
        self,
        prefix: str,
        record: dict[str, Any],
        payload_path: Path,
    ) -> bool:
        if payload_path.stat().st_size != int(record["size"]):
            return False
        blob_path = self._blob_path_for_prefix(prefix, int(record["shard_index"]))
        try:
            with payload_path.open("rb") as payload_file, blob_path.open("rb") as blob_file:
                self._read_record_header(blob_file, record)
                blob_file.seek(int(record["offset"]))
                remaining = int(record["size"])
                while remaining > 0:
                    expected = payload_file.read(min(ZERO_FILL_CHUNK_BYTES, remaining))
                    if not expected:
                        return False
                    actual = blob_file.read(len(expected))
                    if actual != expected:
                        return False
                    remaining -= len(expected)
        except FileNotFoundError as exc:
            raise KVStore2IntegrityError(f"Missing blob file: {blob_path}") from exc
        return True

    def _extend_blob_file(self, blob_file: Any, physical_size: int) -> None:
        current_size = blob_file.seek(0, os.SEEK_END)
        if current_size >= physical_size:
            return

        fileno = blob_file.fileno()
        remaining = physical_size - current_size
        if hasattr(os, "posix_fallocate"):
            try:
                os.posix_fallocate(fileno, current_size, remaining)
                blob_file.truncate(physical_size)
                blob_file.flush()
                os.fsync(fileno)
                return
            except OSError:
                blob_file.seek(current_size)

        blob_file.seek(current_size)
        zero_chunk = b"\0" * min(ZERO_FILL_CHUNK_BYTES, remaining)
        while remaining > 0:
            written = min(len(zero_chunk), remaining)
            blob_file.write(zero_chunk[:written])
            remaining -= written
        blob_file.flush()
        os.fsync(fileno)

    def _read_payload_from_record(self, prefix: str, record: dict[str, Any]) -> bytes:
        blob_path = self._blob_path_for_prefix(prefix, int(record["shard_index"]))
        try:
            with blob_path.open("rb") as blob_file:
                self._read_record_header(blob_file, record)
                blob_file.seek(int(record["offset"]))
                payload = blob_file.read(int(record["size"]))
        except FileNotFoundError as exc:
            raise KVStore2IntegrityError(f"Missing blob file: {blob_path}") from exc
        if len(payload) != int(record["size"]):
            raise KVStore2IntegrityError(
                f"Blob record {record['key']} expected {record['size']} bytes, "
                f"read {len(payload)} bytes."
            )
        return payload

    def _verify_prefix_payloads(self, prefix: str) -> list[str]:
        errors: list[str] = []
        index_path = self._index_path_for_prefix(prefix)
        if not index_path.exists():
            return errors
        with self._connect(index_path) as connection:
            rows = connection.execute(
                """
                SELECT key, shard_index, header_offset, header_size,
                       offset, size, checksum, created_at
                FROM blobs
                """
            ).fetchall()
        for row in rows:
            record = {
                "key": str(row[0]),
                "shard_index": int(row[1]),
                "header_offset": int(row[2]),
                "header_size": int(row[3]),
                "offset": int(row[4]),
                "size": int(row[5]),
                "checksum": str(row[6]),
                "created_at": str(row[7]),
            }
            try:
                payload = self._read_payload_from_record(prefix, record)
            except KVStore2IntegrityError as exc:
                errors.append(str(exc))
                continue
            computed = self._hash_payload(payload)
            if computed != record["key"]:
                errors.append(f"Payload hash mismatch for key {record['key']}.")
        return errors

    def _hash_payload(self, payload: bytes) -> str:
        hasher = self._new_hasher()
        hasher.update(payload)
        return hasher.hexdigest()

    def _new_hasher(self) -> Any:
        self._require_config()
        algorithm = self.config["hash_algorithm"]
        if algorithm == "sha256":
            return hashlib.sha256()
        if algorithm == "blake3":
            try:
                import blake3  # type: ignore
            except ImportError as exc:
                raise KVStore2ConfigError(
                    "hash_algorithm='blake3' requires the third-party blake3 package."
                ) from exc
            return blake3.blake3()
        raise KVStore2ConfigError(f"Unsupported hash algorithm: {algorithm}")

    def _normalize_payload(self, payload: bytes | bytearray | str | None) -> bytes:
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, bytearray):
            return bytes(payload)
        if isinstance(payload, str):
            return payload.encode("utf-8")
        raise TypeError("payload must be bytes, bytearray, or str.")

    def _validate_key(self, key: str) -> None:
        self._require_config()
        if not isinstance(key, str):
            raise KVStore2ConfigError("key must be a string.")
        if len(key) != int(self.config["key_length"]):
            raise KVStore2ConfigError(
                f"key must be {self.config['key_length']} lowercase hexadecimal characters."
            )
        if key != key.lower() or not HEX_RE.match(key):
            raise KVStore2ConfigError("key must contain only lowercase hexadecimal characters.")

    def _validate_blob_limit(self, physical_size: int) -> None:
        limit = int(self.config["limits"]["max_blob_bytes"])
        if physical_size > limit:
            raise KVStore2ConfigError(
                f"Blob shard would exceed max_blob_bytes: {physical_size} > {limit}"
            )

    def _prefix_for_key(self, key: str) -> str:
        self._require_config()
        return key[: int(self.config["prefix_length"])]

    def _prefix_dir_for_prefix(self, prefix: str) -> Path:
        return self.root_path / prefix

    def _index_filename(self, prefix: str) -> str:
        self._require_config()
        return self.config["index_filename_pattern"].format(prefix=prefix)

    def _index_path_for_prefix(self, prefix: str) -> Path:
        return self._prefix_dir_for_prefix(prefix) / self._index_filename(prefix)

    def _blob_filename(self, prefix: str, shard_index: int) -> str:
        self._require_config()
        return self.config["blob_filename_pattern"].format(
            prefix=prefix,
            index=shard_index,
        )

    def _blob_path_for_prefix(self, prefix: str, shard_index: int) -> Path:
        return self._prefix_dir_for_prefix(prefix) / self._blob_filename(prefix, shard_index)

    def _blob_files_for_prefix(self, prefix: str) -> list[Path]:
        prefix_dir = self._prefix_dir_for_prefix(prefix)
        if not prefix_dir.exists():
            return []
        pattern = re.compile(rf"^{re.escape(prefix)}-shard_[0-9]{{3}}\.blob$")
        return sorted(
            path for path in prefix_dir.iterdir()
            if path.is_file() and pattern.match(path.name)
        )

    def _actual_blob_file_size(self, prefix: str, shard_index: int) -> int:
        path = self._blob_path_for_prefix(prefix, shard_index)
        return path.stat().st_size if path.exists() else 0

    def _initial_shard_index(self) -> int:
        self._require_config()
        return int(self.config["initial_shard_index"])

    def _max_blob_bytes(self) -> int:
        self._require_config()
        return int(self.config["limits"]["max_blob_bytes"])

    def _lock_for_prefix(self, prefix: str) -> _FileLock:
        return self._lock_for_path(self._prefix_dir_for_prefix(prefix) / LOCK_FILENAME)

    def _lock_for_path(self, lock_path: Path) -> _FileLock:
        return _FileLock(
            lock_path,
            timeout_s=self.lock_timeout_s,
            poll_interval_s=self.lock_poll_interval_s,
            stale_after_s=self.stale_lock_seconds,
        )

    def _expected_prefixes(self) -> Iterable[str]:
        self._require_config()
        prefix_length = int(self.config["prefix_length"])
        return ("".join(chars) for chars in itertools.product("0123456789abcdef", repeat=prefix_length))

    def _manifest_filename(self) -> str:
        if self.config is None:
            return MANIFEST_FILENAME
        return str(self.config.get("manifest_filename", MANIFEST_FILENAME))

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _db_file_size(self, path: Path) -> int:
        return path.stat().st_size if path.exists() else 0

    def _has_expected_schema(self, db_path: Path) -> bool:
        try:
            with self._connect(db_path) as connection:
                blob_rows = connection.execute("PRAGMA table_info(blobs)").fetchall()
                state_rows = connection.execute("PRAGMA table_info(shard_state)").fetchall()
        except sqlite3.DatabaseError:
            return False
        blob_columns = {row[1]: row[2].upper() for row in blob_rows}
        state_columns = {row[1]: row[2].upper() for row in state_rows}
        blob_required = {
            "key": "TEXT",
            "shard_index": "INTEGER",
            "header_offset": "INTEGER",
            "header_size": "INTEGER",
            "offset": "INTEGER",
            "size": "INTEGER",
            "checksum": "TEXT",
            "created_at": "TEXT",
        }
        state_required = {
            "shard_index": "INTEGER",
            "blob_filename": "TEXT",
            "data_end": "INTEGER",
            "physical_size": "INTEGER",
            "created_at": "TEXT",
            "updated_at": "TEXT",
        }
        return (
            all(blob_columns.get(name) == type_name for name, type_name in blob_required.items())
            and all(state_columns.get(name) == type_name for name, type_name in state_required.items())
        )

    def _sqlite_integrity_check(self, db_path: Path) -> str:
        try:
            with self._connect(db_path) as connection:
                row = connection.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as exc:
            return str(exc)
        return str(row[0]) if row else "missing integrity_check result"

    def _validate_config(self, config: dict[str, Any]) -> None:
        if "layout" not in config and config.get("version") == 1 and "rotation" in config:
            raise KVStore2ConfigError(
                "KVStore2 cannot open this root because it appears to contain a legacy "
                "KVStore config. Set [kvstore].backend = 'kvstore' for this root, choose "
                "an empty kvstore root_path for KVStore2, or migrate/reset the store before "
                "using [kvstore].backend = 'kvstore2'."
            )
        try:
            version = int(config["version"])
            layout = str(config["layout"])
            hash_algorithm = str(config["hash_algorithm"]).lower()
            prefix_length = int(config["prefix_length"])
            key_length = int(config["key_length"])
            initial_shard_index = int(config["initial_shard_index"])
            max_blob_bytes = int(config["limits"]["max_blob_bytes"])
            record_header = config["record_header"]
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise KVStore2ConfigError(f"Invalid config structure: {exc}") from exc

        if version != SCHEMA_VERSION:
            raise KVStore2ConfigError(f"Unsupported config version: {version}")
        if layout != LAYOUT_NAME:
            raise KVStore2ConfigError(f"Unsupported KVStore2 layout: {layout}")
        self._validate_hash_algorithm(hash_algorithm)
        self._validate_prefix_length(prefix_length)
        if key_length != SUPPORTED_HASHES[hash_algorithm]:
            raise KVStore2ConfigError(
                f"Invalid key_length for {hash_algorithm}: {key_length}"
            )
        if initial_shard_index != 1:
            raise KVStore2ConfigError("Only initial_shard_index=1 is supported in this draft.")
        if record_header.get("magic") != RECORD_MAGIC.decode("ascii"):
            raise KVStore2ConfigError("Unsupported record header magic.")
        if int(record_header.get("version", -1)) != RECORD_HEADER_VERSION:
            raise KVStore2ConfigError("Unsupported record header version.")
        self._validate_max_blob_bytes(max_blob_bytes)
        if config.get("index_filename_pattern") != "{prefix}-index.sqlite":
            raise KVStore2ConfigError("index_filename_pattern must be '{prefix}-index.sqlite'.")
        if config.get("blob_filename_pattern") != "{prefix}-shard_{index:03d}.blob":
            raise KVStore2ConfigError(
                "blob_filename_pattern must be '{prefix}-shard_{index:03d}.blob'."
            )

    def _validate_hash_algorithm(self, hash_algorithm: str) -> None:
        if hash_algorithm not in SUPPORTED_HASHES:
            raise KVStore2ConfigError(
                f"Unsupported hash algorithm {hash_algorithm!r}; expected sha256 or blake3."
            )
        if hash_algorithm == "blake3":
            try:
                import blake3  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise KVStore2ConfigError(
                    "hash_algorithm='blake3' requires the third-party blake3 package."
                ) from exc

    def _validate_prefix_length(self, prefix_length: int) -> None:
        if not isinstance(prefix_length, int) or not 1 <= prefix_length <= 4:
            raise KVStore2ConfigError("prefix_length must be an integer from 1 through 4.")

    def _validate_max_blob_bytes(self, max_blob_bytes: int) -> None:
        if not isinstance(max_blob_bytes, int) or max_blob_bytes <= 0:
            raise KVStore2ConfigError("max_blob_bytes must be a positive integer.")

    def _require_initialized(self) -> None:
        if not self.initialized or self.config is None:
            raise KVStore2NotInitializedError("KVStore2 is not initialized; call initialize().")

    def _require_config(self) -> None:
        if self.config is None:
            raise KVStore2NotInitializedError("KVStore2 configuration is not loaded.")
