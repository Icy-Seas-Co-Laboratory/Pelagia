from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


ImageDataStorageEncoding = Literal["png", "raw", "zstd"]
IMAGE_DATA_STORAGE_ENCODINGS = {"png", "raw", "zstd"}


@dataclass(slots=True)
class DatabaseConfig:
    """PostgreSQL connection settings for the catalog and queue."""

    dsn: str = "postgresql://localhost/pelagia"
    schema_name: str = "pelagia"
    connect_timeout_s: int = 5
    statement_timeout_ms: int = 30_000


@dataclass(slots=True)
class QueueConfig:
    """Worker queue leasing and retry settings."""

    default_priority: int = 100
    max_attempts: int = 3
    max_claim_count: int = 1
    lease_seconds: int = 300
    heartbeat_interval_seconds: int = 30


@dataclass(slots=True)
class KVStoreConfig:
    """Large blob store settings."""

    root_path: Path = Path("./data/kvstore")
    hash_algorithm: str = "sha256"
    prefix_length: int = 1
    max_db_bytes: int = 4 * 1024 * 1024 * 1024
    max_rows: int = 1_000_000


@dataclass(slots=True)
class ImageDataStorageConfig:
    """Frame image payload storage settings."""

    encoding: ImageDataStorageEncoding = "zstd"

    def __post_init__(self) -> None:
        self.encoding = str(self.encoding).lower()
        if self.encoding not in IMAGE_DATA_STORAGE_ENCODINGS:
            raise ValueError(
                "image data storage encoding must be one of: png, raw, zstd."
            )


@dataclass(slots=True)
class APIConfig:
    """HTTP API settings."""

    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(slots=True)
class CoreConfig:
    """Top-level application config shared by CLI, API, services, and workers."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    kvstore: KVStoreConfig = field(default_factory=KVStoreConfig)
    image_data_storage: ImageDataStorageConfig = field(default_factory=ImageDataStorageConfig)
    api: APIConfig = field(default_factory=APIConfig)

    @classmethod
    def from_env(cls) -> "CoreConfig":
        """Build config from environment variables, using local defaults."""
        database_defaults = DatabaseConfig()
        queue_defaults = QueueConfig()
        kvstore_defaults = KVStoreConfig()
        image_data_storage_defaults = ImageDataStorageConfig()
        api_defaults = APIConfig()
        return cls(
            database=DatabaseConfig(
                dsn=os.getenv("PELAGIA_DATABASE_DSN", database_defaults.dsn),
                schema_name=os.getenv("PELAGIA_DATABASE_SCHEMA", database_defaults.schema_name),
                connect_timeout_s=int(os.getenv("PELAGIA_DB_CONNECT_TIMEOUT_S", str(database_defaults.connect_timeout_s))),
                statement_timeout_ms=int(os.getenv("PELAGIA_DB_STATEMENT_TIMEOUT_MS", str(database_defaults.statement_timeout_ms))),
            ),
            queue=QueueConfig(
                default_priority=int(os.getenv("PELAGIA_QUEUE_DEFAULT_PRIORITY", str(queue_defaults.default_priority))),
                max_attempts=int(os.getenv("PELAGIA_QUEUE_MAX_ATTEMPTS", str(queue_defaults.max_attempts))),
                max_claim_count=int(os.getenv("PELAGIA_QUEUE_MAX_CLAIM_COUNT", str(queue_defaults.max_claim_count))),
                lease_seconds=int(os.getenv("PELAGIA_QUEUE_LEASE_SECONDS", str(queue_defaults.lease_seconds))),
                heartbeat_interval_seconds=int(os.getenv("PELAGIA_QUEUE_HEARTBEAT_SECONDS", str(queue_defaults.heartbeat_interval_seconds))),
            ),
            kvstore=KVStoreConfig(
                root_path=Path(os.getenv("PELAGIA_KVSTORE_ROOT", str(kvstore_defaults.root_path))),
                hash_algorithm=os.getenv("PELAGIA_KVSTORE_HASH", kvstore_defaults.hash_algorithm),
                prefix_length=int(os.getenv("PELAGIA_KVSTORE_PREFIX_LENGTH", str(kvstore_defaults.prefix_length))),
                max_db_bytes=int(os.getenv("PELAGIA_KVSTORE_MAX_DB_BYTES", str(kvstore_defaults.max_db_bytes))),
                max_rows=int(os.getenv("PELAGIA_KVSTORE_MAX_ROWS", str(kvstore_defaults.max_rows))),
            ),
            image_data_storage=ImageDataStorageConfig(
                encoding=os.getenv(
                    "PELAGIA_IMAGE_DATA_STORAGE_ENCODING",
                    image_data_storage_defaults.encoding,
                ),
            ),
            api=APIConfig(
                host=os.getenv("PELAGIA_API_HOST", api_defaults.host),
                port=int(os.getenv("PELAGIA_API_PORT", str(api_defaults.port))),
            ),
        )
