from __future__ import annotations

import os
import re
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


ImageDataStorageEncoding = Literal["png", "jpg", "raw", "zstd"]
IMAGE_DATA_STORAGE_ENCODINGS = {"png", "jpg", "raw", "zstd"}
_TOML_NULL_SENTINEL = "__PELAGIA_TOML_NULL__"


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
                "image data storage encoding must be one of: png, jpg, raw, zstd."
            )


@dataclass(slots=True)
class APIConfig:
    """HTTP API settings."""

    host: str = "127.0.0.1"
    port: int = 8000


@dataclass(slots=True)
class SegmentationProcessingConfig:
    """Default ROI segmentation parameters."""

    min_perimeter: float = 0.0
    max_perimeter: float | None = None
    padding: int = 0
    roi_encoding: str = "zstd"
    zstd_min_bytes: int = 1024


@dataclass(slots=True)
class VideoIngestProcessingConfig:
    """Default video-to-frame extraction parameters."""

    n_tile: int = 1
    flatfield_correction: bool = True
    flatfield_q: float = 0.9
    flatfield_axis: int = 0
    adaptive_background_subtraction: bool = False
    adaptive_background_period: int = 50
    frame_mask: bool = False
    frame_mask_path: str | None = None


@dataclass(slots=True)
class ThresholdingProcessingConfig:
    """Default thresholding parameters."""

    thresholding_maximum_value: float = 255.0


@dataclass(slots=True)
class FrameStorageProcessingConfig:
    """Default frame storage parameters."""

    image_encoding: ImageDataStorageEncoding = "zstd"
    thumbhash_max_dim: int = 100

    def __post_init__(self) -> None:
        self.image_encoding = str(self.image_encoding).lower()
        if self.image_encoding not in IMAGE_DATA_STORAGE_ENCODINGS:
            raise ValueError(
                "frame storage image_encoding must be one of: png, jpg, raw, zstd."
            )


@dataclass(slots=True)
class ThumbhashProcessingConfig:
    """Default ThumbHash preview parameters."""

    max_dim: int = 100


@dataclass(slots=True)
class ProcessingConfig:
    """Processing defaults shared by CLI, API, workers, and direct function calls."""

    segmentation: SegmentationProcessingConfig = field(default_factory=SegmentationProcessingConfig)
    video_ingest: VideoIngestProcessingConfig = field(default_factory=VideoIngestProcessingConfig)
    thresholding: ThresholdingProcessingConfig = field(default_factory=ThresholdingProcessingConfig)
    frame_storage: FrameStorageProcessingConfig = field(default_factory=FrameStorageProcessingConfig)
    thumbhash: ThumbhashProcessingConfig = field(default_factory=ThumbhashProcessingConfig)


@dataclass(slots=True)
class CoreConfig:
    """Top-level application config shared by CLI, API, services, and workers."""

    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    queue: QueueConfig = field(default_factory=QueueConfig)
    kvstore: KVStoreConfig = field(default_factory=KVStoreConfig)
    image_data_storage: ImageDataStorageConfig = field(default_factory=ImageDataStorageConfig)
    api: APIConfig = field(default_factory=APIConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)

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
    text = re.sub(
        r"(?m)(=\s*)null(\s*(?:#.*)?$)",
        rf'\1"{_TOML_NULL_SENTINEL}"\2',
        data.decode("utf-8"),
    )
    return _normalize_toml_nulls(tomllib.loads(text))


def _normalize_toml_nulls(value: Any) -> Any:
    if value == _TOML_NULL_SENTINEL:
        return None
    if isinstance(value, dict):
        return {key: _normalize_toml_nulls(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_toml_nulls(item) for item in value]
    return value


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
    target = settings
    for part in section.split("."):
        target = target.setdefault(part, {})
    target[key] = cast(value)


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

    _set_from_env(settings, "processing.segmentation", "min_perimeter", "PELAGIA_SEGMENTATION_MIN_PERIMETER", float)
    _set_from_env(settings, "processing.segmentation", "max_perimeter", "PELAGIA_SEGMENTATION_MAX_PERIMETER", float)
    _set_from_env(settings, "processing.segmentation", "padding", "PELAGIA_SEGMENTATION_PADDING", int)
    _set_from_env(settings, "processing.segmentation", "roi_encoding", "PELAGIA_SEGMENTATION_ROI_ENCODING")
    _set_from_env(settings, "processing.segmentation", "zstd_min_bytes", "PELAGIA_SEGMENTATION_ZSTD_MIN_BYTES", int)

    _set_from_env(settings, "processing.video_ingest", "n_tile", "PELAGIA_VIDEO_INGEST_N_TILE", int)
    _set_from_env(settings, "processing.video_ingest", "flatfield_correction", "PELAGIA_VIDEO_INGEST_FLATFIELD_CORRECTION", _env_bool)
    _set_from_env(settings, "processing.video_ingest", "flatfield_q", "PELAGIA_VIDEO_INGEST_FLATFIELD_Q", float)
    _set_from_env(settings, "processing.video_ingest", "flatfield_axis", "PELAGIA_VIDEO_INGEST_FLATFIELD_AXIS", int)
    _set_from_env(settings, "processing.video_ingest", "adaptive_background_subtraction", "PELAGIA_VIDEO_INGEST_ADAPTIVE_BACKGROUND_SUBTRACTION", _env_bool)
    _set_from_env(settings, "processing.video_ingest", "adaptive_background_period", "PELAGIA_VIDEO_INGEST_ADAPTIVE_BACKGROUND_PERIOD", int)
    _set_from_env(settings, "processing.video_ingest", "frame_mask", "PELAGIA_VIDEO_INGEST_FRAME_MASK", _env_bool)
    _set_from_env(settings, "processing.video_ingest", "frame_mask_path", "PELAGIA_VIDEO_INGEST_FRAME_MASK_PATH")

    _set_from_env(settings, "processing.thresholding", "thresholding_maximum_value", "PELAGIA_THRESHOLDING_MAXIMUM_VALUE", float)

    _set_from_env(settings, "processing.frame_storage", "image_encoding", "PELAGIA_FRAME_STORAGE_IMAGE_ENCODING")
    _set_from_env(settings, "processing.frame_storage", "thumbhash_max_dim", "PELAGIA_FRAME_STORAGE_THUMBHASH_MAX_DIM", int)

    _set_from_env(settings, "processing.thumbhash", "max_dim", "PELAGIA_THUMBHASH_MAX_DIM", int)


def _env_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "on"}


def _section(settings: dict[str, Any], name: str) -> dict[str, Any]:
    value: Any = settings
    for part in name.split("."):
        if not isinstance(value, dict):
            raise ValueError(f"Config section [{name}] must be a table.")
        value = value.get(part, {})
    if not isinstance(value, dict):
        raise ValueError(f"Config section [{name}] must be a table.")
    return value


def _config_from_mapping(settings: dict[str, Any]) -> CoreConfig:
    database = _section(settings, "database")
    queue = _section(settings, "queue")
    kvstore = _section(settings, "kvstore")
    image_data_storage = _section(settings, "image_data_storage")
    api = _section(settings, "api")
    segmentation = _section(settings, "processing.segmentation")
    video_ingest = _section(settings, "processing.video_ingest")
    thresholding = _section(settings, "processing.thresholding")
    frame_storage = _section(settings, "processing.frame_storage")
    thumbhash = _section(settings, "processing.thumbhash")

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
        processing=ProcessingConfig(
            segmentation=SegmentationProcessingConfig(
                min_perimeter=float(segmentation.get("min_perimeter", SegmentationProcessingConfig.min_perimeter)),
                max_perimeter=(
                    None
                    if segmentation.get("max_perimeter", None) is None
                    else float(segmentation["max_perimeter"])
                ),
                padding=int(segmentation.get("padding", SegmentationProcessingConfig.padding)),
                roi_encoding=str(segmentation.get("roi_encoding", SegmentationProcessingConfig.roi_encoding)),
                zstd_min_bytes=int(segmentation.get("zstd_min_bytes", SegmentationProcessingConfig.zstd_min_bytes)),
            ),
            video_ingest=VideoIngestProcessingConfig(
                n_tile=int(video_ingest.get("n_tile", VideoIngestProcessingConfig.n_tile)),
                flatfield_correction=bool(
                    video_ingest.get("flatfield_correction", VideoIngestProcessingConfig.flatfield_correction)
                ),
                flatfield_q=float(video_ingest.get("flatfield_q", VideoIngestProcessingConfig.flatfield_q)),
                flatfield_axis=int(video_ingest.get("flatfield_axis", VideoIngestProcessingConfig.flatfield_axis)),
                adaptive_background_subtraction=bool(
                    video_ingest.get(
                        "adaptive_background_subtraction",
                        VideoIngestProcessingConfig.adaptive_background_subtraction,
                    )
                ),
                adaptive_background_period=int(
                    video_ingest.get("adaptive_background_period", VideoIngestProcessingConfig.adaptive_background_period)
                ),
                frame_mask=bool(video_ingest.get("frame_mask", VideoIngestProcessingConfig.frame_mask)),
                frame_mask_path=video_ingest.get("frame_mask_path", VideoIngestProcessingConfig.frame_mask_path),
            ),
            thresholding=ThresholdingProcessingConfig(
                thresholding_maximum_value=float(
                    thresholding.get(
                        "thresholding_maximum_value",
                        ThresholdingProcessingConfig.thresholding_maximum_value,
                    )
                ),
            ),
            frame_storage=FrameStorageProcessingConfig(
                image_encoding=frame_storage.get(
                    "image_encoding",
                    image_data_storage.get("encoding", FrameStorageProcessingConfig.image_encoding),
                ),
                thumbhash_max_dim=int(
                    frame_storage.get("thumbhash_max_dim", FrameStorageProcessingConfig.thumbhash_max_dim)
                ),
            ),
            thumbhash=ThumbhashProcessingConfig(
                max_dim=int(thumbhash.get("max_dim", ThumbhashProcessingConfig.max_dim)),
            ),
        ),
    )
