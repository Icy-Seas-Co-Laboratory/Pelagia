from __future__ import annotations

import os
from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None


ImageDataStorageEncoding = Literal["png", "raw", "zstd"]
IMAGE_DATA_STORAGE_ENCODINGS = {"png", "raw", "zstd"}


@dataclass(slots=True)
class DatabaseConfig:
    """PostgreSQL connection settings for the catalog and queue."""

    dsn: str = "postgresql://postgres:postgres@localhost:5432/pelagia"
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
    def load(
        cls,
        config_path: str | Path | None = None,
        *,
        local_config_path: str | Path | None = "config.toml",
        use_env: bool = True,
    ) -> "CoreConfig":
        """
        Load configuration from packaged defaults, optional local TOML, then env vars.

        Precedence is:
        ``default.config.toml < config.toml < environment variables``.
        """
        settings = _load_packaged_default_config()

        resolved_config_path = config_path
        if resolved_config_path is None:
            resolved_config_path = os.getenv("PELAGIA_CONFIG")
        if resolved_config_path is None:
            resolved_config_path = local_config_path

        if resolved_config_path is not None:
            path = Path(resolved_config_path)
            if path.exists():
                _deep_update(settings, _load_toml_path(path))
            elif config_path is not None:
                raise FileNotFoundError(path)

        if use_env:
            _apply_env_overrides(settings)

        return _config_from_mapping(settings)

def _load_toml_bytes(data: bytes, source: str) -> dict[str, Any]:
    if tomllib is None:
        raise RuntimeError(
            f"Reading TOML config requires Python 3.11+ or the tomli package: {source}"
        )
    return tomllib.loads(data.decode("utf-8"))


def _load_packaged_default_config() -> dict[str, Any]:
    data = files(__package__).joinpath("default.config.toml").read_bytes()
    return _load_toml_bytes(data, "Pelagia/default.config.toml")


def _load_toml_path(path: Path) -> dict[str, Any]:
    return _load_toml_bytes(path.read_bytes(), str(path))


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _set_from_env(
    settings: dict[str, Any],
    section: str,
    key: str,
    env_name: str,
    cast: type = str,
) -> None:
    value = os.getenv(env_name)
    if value is None:
        return
    settings.setdefault(section, {})[key] = cast(value)


def _apply_env_overrides(settings: dict[str, Any]) -> None:
    _set_from_env(settings, "database", "dsn", "PELAGIA_DATABASE_DSN")
    _set_from_env(settings, "database", "schema_name", "PELAGIA_DATABASE_SCHEMA")
    _set_from_env(settings, "database", "connect_timeout_s", "PELAGIA_DB_CONNECT_TIMEOUT_S", int)
    _set_from_env(settings, "database", "statement_timeout_ms", "PELAGIA_DB_STATEMENT_TIMEOUT_MS", int)

    _set_from_env(settings, "queue", "default_priority", "PELAGIA_QUEUE_DEFAULT_PRIORITY", int)
    _set_from_env(settings, "queue", "max_attempts", "PELAGIA_QUEUE_MAX_ATTEMPTS", int)
    _set_from_env(settings, "queue", "max_claim_count", "PELAGIA_QUEUE_MAX_CLAIM_COUNT", int)
    _set_from_env(settings, "queue", "lease_seconds", "PELAGIA_QUEUE_LEASE_SECONDS", int)
    _set_from_env(settings, "queue", "heartbeat_interval_seconds", "PELAGIA_QUEUE_HEARTBEAT_SECONDS", int)

    _set_from_env(settings, "kvstore", "root_path", "PELAGIA_KVSTORE_ROOT", Path)
    _set_from_env(settings, "kvstore", "hash_algorithm", "PELAGIA_KVSTORE_HASH")
    _set_from_env(settings, "kvstore", "prefix_length", "PELAGIA_KVSTORE_PREFIX_LENGTH", int)
    _set_from_env(settings, "kvstore", "max_db_bytes", "PELAGIA_KVSTORE_MAX_DB_BYTES", int)
    _set_from_env(settings, "kvstore", "max_rows", "PELAGIA_KVSTORE_MAX_ROWS", int)

    _set_from_env(settings, "image_data_storage", "encoding", "PELAGIA_IMAGE_DATA_STORAGE_ENCODING")

    _set_from_env(settings, "api", "host", "PELAGIA_API_HOST")
    _set_from_env(settings, "api", "port", "PELAGIA_API_PORT", int)


def _section(settings: dict[str, Any], name: str) -> dict[str, Any]:
    value = settings.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table.")
    return value


def _config_from_mapping(settings: dict[str, Any]) -> CoreConfig:
    database = _section(settings, "database")
    queue = _section(settings, "queue")
    kvstore = _section(settings, "kvstore")
    image_data_storage = _section(settings, "image_data_storage")
    api = _section(settings, "api")

    return CoreConfig(
        database=DatabaseConfig(
            dsn=str(database.get("dsn", DatabaseConfig.dsn)),
            schema_name=str(database.get("schema_name", DatabaseConfig.schema_name)),
            connect_timeout_s=int(database.get("connect_timeout_s", DatabaseConfig.connect_timeout_s)),
            statement_timeout_ms=int(database.get("statement_timeout_ms", DatabaseConfig.statement_timeout_ms)),
        ),
        queue=QueueConfig(
            default_priority=int(queue.get("default_priority", QueueConfig.default_priority)),
            max_attempts=int(queue.get("max_attempts", QueueConfig.max_attempts)),
            max_claim_count=int(queue.get("max_claim_count", QueueConfig.max_claim_count)),
            lease_seconds=int(queue.get("lease_seconds", QueueConfig.lease_seconds)),
            heartbeat_interval_seconds=int(
                queue.get("heartbeat_interval_seconds", QueueConfig.heartbeat_interval_seconds)
            ),
        ),
        kvstore=KVStoreConfig(
            root_path=Path(kvstore.get("root_path", KVStoreConfig.root_path)),
            hash_algorithm=str(kvstore.get("hash_algorithm", KVStoreConfig.hash_algorithm)),
            prefix_length=int(kvstore.get("prefix_length", KVStoreConfig.prefix_length)),
            max_db_bytes=int(kvstore.get("max_db_bytes", KVStoreConfig.max_db_bytes)),
            max_rows=int(kvstore.get("max_rows", KVStoreConfig.max_rows)),
        ),
        image_data_storage=ImageDataStorageConfig(
            encoding=image_data_storage.get("encoding", ImageDataStorageConfig.encoding),
        ),
        api=APIConfig(
            host=str(api.get("host", APIConfig.host)),
            port=int(api.get("port", APIConfig.port)),
        ),
    )
