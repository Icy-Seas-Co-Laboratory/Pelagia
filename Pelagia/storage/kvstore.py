from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_MAX_DB_BYTES = 4 * 1024 * 1024 * 1024
DEFAULT_MAX_ROWS = 1_000_000
CONFIG_FILENAME = "config.json"
LOCK_FILENAME = ".kvstore.lock"
SCHEMA_VERSION = 1
SUPPORTED_HASHES = {"sha256": 64, "blake3": 64}
HEX_RE = re.compile(r"^[0-9a-f]+$")


class KVStoreError(Exception):
    """Base exception for KVStore errors."""


class KVStoreNotInitializedError(KVStoreError):
    """Raised when an operation requires an initialized store."""


class KVStoreAlreadyInitializedError(KVStoreError):
    """Raised when initialization would overwrite an existing store."""


class KVStoreConfigError(KVStoreError):
    """Raised when store configuration is missing or invalid."""


class KVStoreIntegrityError(KVStoreError):
    """Raised when stored or supplied data violates content-addressing rules."""


class KVStoreRotationError(KVStoreError):
    """Raised when a rotated SQLite file cannot be selected or created."""


class KVStoreLockError(KVStoreError):
    """Raised when a store lock cannot be acquired."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _normalize_root_path(root_path: str | Path) -> Path:
    raw_path = os.fspath(root_path)
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    return Path(expanded).resolve(strict=False)


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
                    raise KVStoreLockError(f"Timed out waiting for lock: {self.lock_path}")
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


class KVStore:
    """A content-addressed key-value store backed by rotated SQLite files."""

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
        sqlite_basename: str = "store",
        max_db_bytes: int = DEFAULT_MAX_DB_BYTES,
        max_rows: int = DEFAULT_MAX_ROWS,
        overwrite: bool = False,
    ) -> None:
        """Initialize a new store on disk with config, manifest, shards, and schema."""
        hash_algorithm = hash_algorithm.lower()
        self._validate_hash_algorithm(hash_algorithm)
        self._validate_prefix_length(prefix_length)
        self._validate_rotation(max_db_bytes, max_rows)
        if not sqlite_basename:
            raise KVStoreConfigError("sqlite_basename must be a non-empty string.")

        self.root_path.mkdir(parents=True, exist_ok=True)
        with self._lock_for_path(self.root_path / LOCK_FILENAME):
            content_paths = [
                path for path in self.root_path.iterdir()
                if path.name != LOCK_FILENAME
            ]
            if content_paths:
                if not overwrite:
                    raise KVStoreAlreadyInitializedError(
                        f"Store path is not empty: {self.root_path}"
                    )
                for path in content_paths:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()

            created_at = _utc_now()
            config = {
                "version": 1,
                "hash_algorithm": hash_algorithm,
                "prefix_length": prefix_length,
                "sqlite_basename": sqlite_basename,
                "sqlite_extension": ".sqlite",
                "sqlite_filename_pattern": f"{sqlite_basename}_{{index:06d}}.sqlite",
                "manifest_filename": "manifest.json",
                "created_at": created_at,
                "key_length": SUPPORTED_HASHES[hash_algorithm],
                "rotation": {
                    "max_db_bytes": max_db_bytes,
                    "max_rows": max_rows,
                },
            }

            self.config = config
            self.initialized = True
            self.config_path = self.root_path / CONFIG_FILENAME

            self._write_config()
            self._write_manifest(created_at=created_at)

            for prefix in self._expected_prefixes():
                prefix_dir = self._prefix_dir_for_prefix(prefix)
                prefix_dir.mkdir(parents=True, exist_ok=True)
                self._create_schema(prefix_dir / self._sqlite_filename(1))

    def get_store(self, key: str) -> bytes:
        """Return payload bytes for ``key`` or raise ``KeyError`` if it is missing."""
        self._require_initialized()
        self._validate_key(key)
        prefix = self._prefix_for_key(key)
        found = self._find_key_in_prefix(key, prefix)
        if found is None:
            raise KeyError(key)
        return found[1]

    def key_delete(self, key: str) -> None:
        """Delete ``key`` from the store or raise ``KeyError`` if it is missing."""
        self._require_initialized()
        self._validate_key(key)
        prefix = self._prefix_for_key(key)
        deleted = False

        with self._lock_for_prefix(prefix):
            for db_path in reversed(self._sqlite_files_for_prefix(prefix)):
                with self._connect(db_path) as connection:
                    cursor = connection.execute("DELETE FROM blobs WHERE key = ?", (key,))
                deleted = deleted or cursor.rowcount > 0

        if not deleted:
            raise KeyError(key)

    def put_store(self, key: str | bytes | bytearray | None = None, payload: bytes | bytearray | str | None = None) -> str:
        """Store ``payload`` and return its content-addressed key.

        Call as ``put_store(payload)`` to compute the key automatically, or as
        ``put_store(key, payload)`` / ``put_store(None, payload)`` for explicit
        compatibility with older two-argument call sites.
        """
        self._require_initialized()

        explicit_key: str | None
        if payload is None:
            explicit_key = None
            payload_value = key
        else:
            explicit_key = key if isinstance(key, str) else None
            payload_value = payload

        payload_bytes = self._normalize_payload(payload_value)
        computed_key = self._hash_payload(payload_bytes)

        if explicit_key is not None:
            self._validate_key(explicit_key)
            if explicit_key != computed_key:
                raise KVStoreIntegrityError(
                    "Explicit key does not match the configured hash of the payload."
                )
            key_to_store = explicit_key
        else:
            key_to_store = computed_key

        prefix = self._prefix_for_key(key_to_store)
        with self._lock_for_prefix(prefix):
            found = self._find_key_in_prefix(key_to_store, prefix)
            if found is not None:
                _, existing_payload = found
                if existing_payload != payload_bytes:
                    raise KVStoreIntegrityError(
                        f"Key {key_to_store} already exists with different payload bytes."
                    )
                return key_to_store

            db_path = self._active_db_path_for_prefix(prefix)
            with self._connect(db_path) as connection:
                connection.execute(
                    """
                    INSERT INTO blobs (key, payload, size, created_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (key_to_store, payload_bytes, len(payload_bytes), _utc_now()),
                )
        return key_to_store

    def key_exists(self, key: str) -> bool:
        """Return whether ``key`` exists in any rotated SQLite file for its prefix."""
        self._require_initialized()
        self._validate_key(key)
        prefix = self._prefix_for_key(key)
        return self._find_key_in_prefix(key, prefix) is not None

    def check_health(self) -> dict[str, Any]:
        """Inspect configuration, shard directories, SQLite filenames, and schemas."""
        errors: list[str] = []
        warnings: list[str] = []

        if not self.config_path.exists():
            errors.append(f"Missing config file: {self.config_path}")

        try:
            config = self._load_config() if self.config_path.exists() else self.config
        except KVStoreError as exc:
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

        expected_pattern = config["sqlite_filename_pattern"]
        for prefix in self._expected_prefixes():
            prefix_dir = self._prefix_dir_for_prefix(prefix)
            if not prefix_dir.exists():
                errors.append(f"Missing prefix directory: {prefix_dir}")
                continue
            if not prefix_dir.is_dir():
                errors.append(f"Prefix path is not a directory: {prefix_dir}")
                continue

            sqlite_files = self._sqlite_files_for_prefix(prefix)
            if not sqlite_files:
                errors.append(f"Prefix {prefix} has no SQLite database files.")

            indexes: list[int] = []
            for child in sorted(prefix_dir.iterdir()):
                if child.is_dir():
                    continue
                if child.suffix != self._sqlite_extension():
                    continue
                index = self._sqlite_index_from_filename(child.name)
                if index is None:
                    errors.append(
                        f"Invalid SQLite filename in prefix {prefix}: {child.name}; "
                        f"expected pattern {expected_pattern}"
                    )
                    continue
                if index < 1:
                    errors.append(f"Invalid non-positive SQLite index: {child.name}")
                    continue
                indexes.append(index)
                if not self._has_expected_schema(child):
                    errors.append(f"SQLite file has unexpected schema: {child}")
                integrity = self._sqlite_integrity_check(child)
                if integrity != "ok":
                    errors.append(f"SQLite integrity check failed for {child}: {integrity}")

            if indexes:
                ordered = sorted(indexes)
                expected = list(range(ordered[0], ordered[-1] + 1))
                if ordered != expected:
                    warnings.append(f"SQLite indexes for prefix {prefix} have gaps: {ordered}")

        return {
            "healthy": not errors,
            "errors": errors,
            "warnings": warnings,
            "checked_at": _utc_now(),
        }

    def status(self) -> dict[str, Any]:
        """Return counts, byte totals, rotation settings, and paths for the store."""
        if not self.initialized or self.config is None:
            return {
                "root_path": str(self.root_path),
                "initialized": False,
                "config_path": str(self.config_path),
                "manifest_path": str(self.root_path / "manifest.json"),
            }

        total_sqlite_files = 0
        total_blobs = 0
        total_payload_bytes = 0
        largest_sqlite_file_size = 0
        largest_sqlite_row_count = 0

        for prefix in self._expected_prefixes():
            for db_path in self._sqlite_files_for_prefix(prefix):
                total_sqlite_files += 1
                file_size = self._db_file_size(db_path)
                row_count = self._db_row_count(db_path)
                total_blobs += row_count
                total_payload_bytes += self._db_payload_bytes(db_path)
                largest_sqlite_file_size = max(largest_sqlite_file_size, file_size)
                largest_sqlite_row_count = max(largest_sqlite_row_count, row_count)

        prefix_length = int(self.config["prefix_length"])
        return {
            "root_path": str(self.root_path),
            "initialized": True,
            "hash_algorithm": self.config["hash_algorithm"],
            "prefix_length": prefix_length,
            "prefix_directory_count": 16**prefix_length,
            "total_sqlite_files": total_sqlite_files,
            "total_stored_blobs": total_blobs,
            "total_stored_payload_bytes": total_payload_bytes,
            "rotation": dict(self.config["rotation"]),
            "largest_sqlite_file_size": largest_sqlite_file_size,
            "largest_sqlite_row_count": largest_sqlite_row_count,
            "config_path": str(self.config_path),
            "manifest_path": str(self.root_path / self._manifest_filename()),
        }

    def _load_config(self) -> dict[str, Any]:
        try:
            config = json.loads(self.config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise KVStoreConfigError(f"Invalid JSON config: {exc}") from exc

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
            "hash_algorithm": self.config["hash_algorithm"],
            "prefix_length": prefix_length,
            "shard_directory_count": 16**prefix_length,
            "sqlite_filename_pattern": self.config["sqlite_filename_pattern"],
            "rotation": dict(self.config["rotation"]),
            "created_at": created_at,
            "initialized": True,
        }
        (self.root_path / self._manifest_filename()).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _hash_payload(self, payload: bytes) -> str:
        self._require_config()
        algorithm = self.config["hash_algorithm"]
        if algorithm == "sha256":
            return hashlib.sha256(payload).hexdigest()
        if algorithm == "blake3":
            try:
                import blake3  # type: ignore
            except ImportError as exc:
                raise KVStoreConfigError(
                    "hash_algorithm='blake3' requires the third-party blake3 package."
                ) from exc
            return blake3.blake3(payload).hexdigest()
        raise KVStoreConfigError(f"Unsupported hash algorithm: {algorithm}")

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
            raise KVStoreConfigError("key must be a string.")
        if len(key) != int(self.config["key_length"]):
            raise KVStoreConfigError(
                f"key must be {self.config['key_length']} lowercase hexadecimal characters."
            )
        if key != key.lower() or not HEX_RE.match(key):
            raise KVStoreConfigError("key must contain only lowercase hexadecimal characters.")

    def _prefix_for_key(self, key: str) -> str:
        self._require_config()
        return key[: int(self.config["prefix_length"])]

    def _prefix_dir_for_key(self, key: str) -> Path:
        return self._prefix_dir_for_prefix(self._prefix_for_key(key))

    def _prefix_dir_for_prefix(self, prefix: str) -> Path:
        return self.root_path / prefix

    def _lock_for_prefix(self, prefix: str) -> _FileLock:
        return self._lock_for_path(self._prefix_dir_for_prefix(prefix) / LOCK_FILENAME)

    def _lock_for_path(self, lock_path: Path) -> _FileLock:
        return _FileLock(
            lock_path,
            timeout_s=self.lock_timeout_s,
            poll_interval_s=self.lock_poll_interval_s,
            stale_after_s=self.stale_lock_seconds,
        )

    def _sqlite_filename(self, index: int) -> str:
        self._require_config()
        return self.config["sqlite_filename_pattern"].format(index=index)

    def _sqlite_index_from_filename(self, filename: str) -> int | None:
        self._require_config()
        basename = re.escape(self.config["sqlite_basename"])
        extension = re.escape(self.config["sqlite_extension"])
        match = re.match(rf"^{basename}_(?P<index>[0-9]{{6}}){extension}$", filename)
        if not match:
            return None
        return int(match.group("index"))

    def _sqlite_files_for_prefix(self, prefix: str) -> list[Path]:
        prefix_dir = self._prefix_dir_for_prefix(prefix)
        if not prefix_dir.exists():
            return []
        files = []
        for path in prefix_dir.iterdir():
            if not path.is_file():
                continue
            if self._sqlite_index_from_filename(path.name) is not None:
                files.append(path)
        return sorted(
            files,
            key=lambda path: self._sqlite_index_from_filename(path.name) or 0,
        )

    def _active_db_path_for_prefix(self, prefix: str) -> Path:
        files = self._sqlite_files_for_prefix(prefix)
        if not files:
            path = self._prefix_dir_for_prefix(prefix) / self._sqlite_filename(1)
            self._create_schema(path)
            return path

        active = files[-1]
        if self._should_rotate(active):
            active = self._next_db_path_for_prefix(prefix)
            self._create_schema(active)
        return active

    def _should_rotate(self, db_path: Path) -> bool:
        self._require_config()
        rotation = self.config["rotation"]
        return (
            self._db_file_size(db_path) >= int(rotation["max_db_bytes"])
            or self._db_row_count(db_path) >= int(rotation["max_rows"])
        )

    def _next_db_path_for_prefix(self, prefix: str) -> Path:
        files = self._sqlite_files_for_prefix(prefix)
        next_index = 1
        if files:
            last_index = self._sqlite_index_from_filename(files[-1].name)
            if last_index is None:
                raise KVStoreRotationError(f"Cannot parse SQLite index: {files[-1]}")
            next_index = last_index + 1
        return self._prefix_dir_for_prefix(prefix) / self._sqlite_filename(next_index)

    def _db_row_count(self, db_path: Path) -> int:
        if not db_path.exists():
            return 0
        with self._connect(db_path) as connection:
            row = connection.execute("SELECT COUNT(*) FROM blobs").fetchone()
        return int(row[0])

    def _db_payload_bytes(self, db_path: Path) -> int:
        if not db_path.exists():
            return 0
        with self._connect(db_path) as connection:
            row = connection.execute("SELECT COALESCE(SUM(size), 0) FROM blobs").fetchone()
        return int(row[0])

    def _db_file_size(self, db_path: Path) -> int:
        return db_path.stat().st_size if db_path.exists() else 0

    def _connect(self, db_path: Path) -> sqlite3.Connection:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(db_path)
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _create_schema(self, db_path: Path) -> None:
        with self._connect(db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS blobs (
                    key TEXT PRIMARY KEY,
                    payload BLOB NOT NULL,
                    size INTEGER NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_blobs_created_at ON blobs (created_at)"
            )

    def _expected_prefixes(self) -> Iterable[str]:
        self._require_config()
        prefix_length = int(self.config["prefix_length"])
        return ("".join(chars) for chars in itertools.product("0123456789abcdef", repeat=prefix_length))

    def _find_key_in_prefix(self, key: str, prefix: str) -> tuple[Path, bytes] | None:
        for db_path in reversed(self._sqlite_files_for_prefix(prefix)):
            with self._connect(db_path) as connection:
                row = connection.execute(
                    "SELECT payload FROM blobs WHERE key = ?",
                    (key,),
                ).fetchone()
            if row is not None:
                return db_path, bytes(row[0])
        return None

    def _manifest_filename(self) -> str:
        if self.config is None:
            return "manifest.json"
        return str(self.config.get("manifest_filename", "manifest.json"))

    def _sqlite_extension(self) -> str:
        self._require_config()
        return str(self.config["sqlite_extension"])

    def _has_expected_schema(self, db_path: Path) -> bool:
        try:
            with self._connect(db_path) as connection:
                rows = connection.execute("PRAGMA table_info(blobs)").fetchall()
        except sqlite3.DatabaseError:
            return False
        columns = {row[1]: row[2].upper() for row in rows}
        required = {
            "key": "TEXT",
            "payload": "BLOB",
            "size": "INTEGER",
            "created_at": "TEXT",
        }
        return all(columns.get(name) == type_name for name, type_name in required.items())

    def _sqlite_integrity_check(self, db_path: Path) -> str:
        try:
            with self._connect(db_path) as connection:
                row = connection.execute("PRAGMA integrity_check").fetchone()
        except sqlite3.DatabaseError as exc:
            return str(exc)
        return str(row[0]) if row else "missing integrity_check result"

    def _validate_config(self, config: dict[str, Any]) -> None:
        try:
            version = config["version"]
            hash_algorithm = str(config["hash_algorithm"]).lower()
            prefix_length = int(config["prefix_length"])
            key_length = int(config["key_length"])
            rotation = config["rotation"]
            max_db_bytes = int(rotation["max_db_bytes"])
            max_rows = int(rotation["max_rows"])
        except (KeyError, TypeError, ValueError) as exc:
            raise KVStoreConfigError(f"Invalid config structure: {exc}") from exc

        if version != 1:
            raise KVStoreConfigError(f"Unsupported config version: {version}")
        self._validate_hash_algorithm(hash_algorithm)
        self._validate_prefix_length(prefix_length)
        if key_length != SUPPORTED_HASHES[hash_algorithm]:
            raise KVStoreConfigError(
                f"Invalid key_length for {hash_algorithm}: {key_length}"
            )
        self._validate_rotation(max_db_bytes, max_rows)
        if not config.get("sqlite_basename"):
            raise KVStoreConfigError("sqlite_basename is required.")
        if config.get("sqlite_extension") != ".sqlite":
            raise KVStoreConfigError("sqlite_extension must be '.sqlite'.")
        if "{index:06d}" not in config.get("sqlite_filename_pattern", ""):
            raise KVStoreConfigError("sqlite_filename_pattern must include {index:06d}.")

    def _validate_hash_algorithm(self, hash_algorithm: str) -> None:
        if hash_algorithm not in SUPPORTED_HASHES:
            raise KVStoreConfigError(
                f"Unsupported hash algorithm {hash_algorithm!r}; expected sha256 or blake3."
            )
        if hash_algorithm == "blake3":
            try:
                import blake3  # type: ignore  # noqa: F401
            except ImportError as exc:
                raise KVStoreConfigError(
                    "hash_algorithm='blake3' requires the third-party blake3 package."
                ) from exc

    def _validate_prefix_length(self, prefix_length: int) -> None:
        if not isinstance(prefix_length, int) or not 1 <= prefix_length <= 4:
            raise KVStoreConfigError("prefix_length must be an integer from 1 through 4.")

    def _validate_rotation(self, max_db_bytes: int, max_rows: int) -> None:
        if not isinstance(max_db_bytes, int) or max_db_bytes <= 0:
            raise KVStoreConfigError("max_db_bytes must be a positive integer.")
        if not isinstance(max_rows, int) or max_rows <= 0:
            raise KVStoreConfigError("max_rows must be a positive integer.")

    def _require_initialized(self) -> None:
        if not self.initialized or self.config is None:
            raise KVStoreNotInitializedError("KVStore is not initialized; call initialize().")

    def _require_config(self) -> None:
        if self.config is None:
            raise KVStoreNotInitializedError("KVStore configuration is not loaded.")
