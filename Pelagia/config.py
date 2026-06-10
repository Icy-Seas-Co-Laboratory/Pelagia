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
    prefix_length: int = 3
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
    cors_allow_origin_regex: str = (
        r"https?://("
        r"localhost|127\.0\.0\.1|\[::1\]|"
        r"10(?:\.\d{1,3}){3}|"
        r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])(?:\.\d{1,3}){2}|"
        r"192\.168(?:\.\d{1,3}){2}|"
        r"172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2}"
        r")(?::\d+)?"
    )


@dataclass(slots=True)
class LoggingConfig:
    """Operational file logging settings."""

    log_path: Path = Path("./logs")
    file_name: str = "pelagia.log"
    level: str = "INFO"
    console: bool = True
    max_bytes: int = 10 * 1024 * 1024
    backup_count: int = 10


@dataclass(slots=True)
class MaskAugmentationProcessingConfig:
    """Default binary-mask augmentation parameters."""

    enabled: bool = True
    steps: list[str] = field(default_factory=lambda: ["dilate"])
    dilate_kernel_w: int = 3
    dilate_kernel_h: int = 3
    dilate_iterations: int = 1
    erode_kernel_w: int = 3
    erode_kernel_h: int = 3
    erode_iterations: int = 1
    open_kernel_w: int = 3
    open_kernel_h: int = 3
    open_iterations: int = 1
    close_kernel_w: int = 3
    close_kernel_h: int = 3
    close_iterations: int = 1
    fill_holes: bool = False
    remove_small_components: bool = False
    min_component_area: float = 1.0
    clear_border: bool = False


@dataclass(slots=True)
class RoiAssemblyProcessingConfig:
    """Default candidate ROI assembly parameters."""

    method: str = "connected_components"
    connectivity: int = 8


@dataclass(slots=True)
class RoiFilterProcessingConfig:
    """Default candidate ROI filtering parameters."""

    min_area: float | None = None
    max_area: float | None = None
    min_perimeter: float = 50.0
    max_perimeter: float | None = None
    min_width: float | None = None
    max_width: float | None = None
    min_height: float | None = None
    max_height: float | None = None
    min_width_plus_height: float | None = None
    max_width_plus_height: float | None = None


@dataclass(slots=True)
class RoiRecordingProcessingConfig:
    """Default candidate ROI recording parameters."""

    padding: int = 50
    roi_encoding: str = "zstd"
    zstd_min_bytes: int = 16_384
    always_store_mask: bool = True
    store_roi_payload_min_area: float | None = None
    store_roi_payload_min_width: float | None = None
    store_roi_payload_min_height: float | None = None
    store_roi_payload_min_width_plus_height: float | None = None


@dataclass(slots=True)
class VideoIngestProcessingConfig:
    """Default video-to-frame extraction parameters."""

    n_tile: int = 4


@dataclass(slots=True)
class FlatfieldProcessingConfig:
    """Default flatfield correction parameters."""

    flatfield_correction: bool = True
    flatfield_q: float = 0.9
    flatfield_axis: int = 0


@dataclass(slots=True)
class ThresholdingProcessingConfig:
    """Default thresholding parameters."""

    method: str = "otsu"
    manual_threshold: float = 100.0
    thresholding_maximum_value: float = 255.0
    bounded_otsu_min_contrast: float = 50.0
    bounded_otsu_max_foreground_fraction: float = 0.9
    canny_enabled: bool = True
    canny_low_threshold: float = 30.0
    canny_high_threshold: float = 80.0
    canny_blur_kernel: int = 5
    adaptive_block_size: int = 31
    adaptive_c: float = 5.0
    percentile_background_percentile: float = 50.0
    percentile_min_contrast: float = 50.0
    hysteresis_low_threshold: float = 30.0
    hysteresis_high_threshold: float = 80.0
    hysteresis_connectivity: int = 8
    sobel_percentile: float = 90.0
    sobel_threshold: float | None = None
    sobel_kernel_size: int = 3


@dataclass(slots=True)
class PreprocessingConfig:
    """Default frame preprocessing parameters."""

    apply_mask: bool = False
    mask_path: str | None = None
    adaptive_background_subtraction: bool = False
    adaptive_background_period: int = 50
    background_correction: bool = False
    background_percentile: float = 50.0
    invert_intensity: bool = True
    crop_enabled: bool = False
    crop_x: int | None = None
    crop_y: int | None = None
    crop_w: int | None = None
    crop_h: int | None = None


@dataclass(slots=True)
class FrameStorageProcessingConfig:
    """Default frame storage parameters."""

    image_encoding: ImageDataStorageEncoding = "jpg"

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

    mask_augmentation: MaskAugmentationProcessingConfig = field(default_factory=MaskAugmentationProcessingConfig)
    roi_assembly: RoiAssemblyProcessingConfig = field(default_factory=RoiAssemblyProcessingConfig)
    roi_filter: RoiFilterProcessingConfig = field(default_factory=RoiFilterProcessingConfig)
    roi_recording: RoiRecordingProcessingConfig = field(default_factory=RoiRecordingProcessingConfig)
    video_ingest: VideoIngestProcessingConfig = field(default_factory=VideoIngestProcessingConfig)
    flatfield: FlatfieldProcessingConfig = field(default_factory=FlatfieldProcessingConfig)
    thresholding: ThresholdingProcessingConfig = field(default_factory=ThresholdingProcessingConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
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
    logging: LoggingConfig = field(default_factory=LoggingConfig)
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
    _set_from_env(settings, "api", "cors_allow_origin_regex", "PELAGIA_API_CORS_ALLOW_ORIGIN_REGEX")

    _set_from_env(settings, "logging", "log_path", "PELAGIA_LOG_PATH", Path)
    _set_from_env(settings, "logging", "file_name", "PELAGIA_LOG_FILE")
    _set_from_env(settings, "logging", "level", "PELAGIA_LOG_LEVEL")
    _set_from_env(settings, "logging", "console", "PELAGIA_LOG_CONSOLE", _env_bool)
    _set_from_env(settings, "logging", "max_bytes", "PELAGIA_LOG_MAX_BYTES", int)
    _set_from_env(settings, "logging", "backup_count", "PELAGIA_LOG_BACKUP_COUNT", int)

    _set_from_env(settings, "processing.video_ingest", "n_tile", "PELAGIA_VIDEO_INGEST_N_TILE", int)

    _set_from_env(settings, "processing.flatfield", "flatfield_correction", "PELAGIA_FLATFIELD_CORRECTION", _env_bool)
    _set_from_env(settings, "processing.flatfield", "flatfield_q", "PELAGIA_FLATFIELD_Q", float)
    _set_from_env(settings, "processing.flatfield", "flatfield_axis", "PELAGIA_FLATFIELD_AXIS", int)

    _set_from_env(settings, "processing.thresholding", "method", "PELAGIA_THRESHOLDING_METHOD")
    _set_from_env(settings, "processing.thresholding", "manual_threshold", "PELAGIA_THRESHOLDING_MANUAL_THRESHOLD", float)
    _set_from_env(settings, "processing.thresholding", "thresholding_maximum_value", "PELAGIA_THRESHOLDING_MAXIMUM_VALUE", float)
    _set_from_env(settings, "processing.thresholding", "bounded_otsu_min_contrast", "PELAGIA_THRESHOLDING_BOUNDED_OTSU_MIN_CONTRAST", float)
    _set_from_env(settings, "processing.thresholding", "bounded_otsu_max_foreground_fraction", "PELAGIA_THRESHOLDING_BOUNDED_OTSU_MAX_FOREGROUND_FRACTION", float)
    _set_from_env(settings, "processing.thresholding", "canny_enabled", "PELAGIA_THRESHOLDING_CANNY_ENABLED", _env_bool)
    _set_from_env(settings, "processing.thresholding", "canny_low_threshold", "PELAGIA_THRESHOLDING_CANNY_LOW_THRESHOLD", float)
    _set_from_env(settings, "processing.thresholding", "canny_high_threshold", "PELAGIA_THRESHOLDING_CANNY_HIGH_THRESHOLD", float)
    _set_from_env(settings, "processing.thresholding", "canny_blur_kernel", "PELAGIA_THRESHOLDING_CANNY_BLUR_KERNEL", int)
    _set_from_env(settings, "processing.thresholding", "adaptive_block_size", "PELAGIA_THRESHOLDING_ADAPTIVE_BLOCK_SIZE", int)
    _set_from_env(settings, "processing.thresholding", "adaptive_c", "PELAGIA_THRESHOLDING_ADAPTIVE_C", float)
    _set_from_env(settings, "processing.thresholding", "percentile_background_percentile", "PELAGIA_THRESHOLDING_PERCENTILE_BACKGROUND_PERCENTILE", float)
    _set_from_env(settings, "processing.thresholding", "percentile_min_contrast", "PELAGIA_THRESHOLDING_PERCENTILE_MIN_CONTRAST", float)
    _set_from_env(settings, "processing.thresholding", "hysteresis_low_threshold", "PELAGIA_THRESHOLDING_HYSTERESIS_LOW_THRESHOLD", float)
    _set_from_env(settings, "processing.thresholding", "hysteresis_high_threshold", "PELAGIA_THRESHOLDING_HYSTERESIS_HIGH_THRESHOLD", float)
    _set_from_env(settings, "processing.thresholding", "hysteresis_connectivity", "PELAGIA_THRESHOLDING_HYSTERESIS_CONNECTIVITY", int)
    _set_from_env(settings, "processing.thresholding", "sobel_percentile", "PELAGIA_THRESHOLDING_SOBEL_PERCENTILE", float)
    _set_from_env(settings, "processing.thresholding", "sobel_threshold", "PELAGIA_THRESHOLDING_SOBEL_THRESHOLD", float)
    _set_from_env(settings, "processing.thresholding", "sobel_kernel_size", "PELAGIA_THRESHOLDING_SOBEL_KERNEL_SIZE", int)

    _set_from_env(settings, "processing.mask_augmentation", "enabled", "PELAGIA_MASK_AUGMENTATION_ENABLED", _env_bool)
    _set_from_env(settings, "processing.mask_augmentation", "dilate_kernel_w", "PELAGIA_MASK_AUGMENTATION_DILATE_KERNEL_W", int)
    _set_from_env(settings, "processing.mask_augmentation", "dilate_kernel_h", "PELAGIA_MASK_AUGMENTATION_DILATE_KERNEL_H", int)
    _set_from_env(settings, "processing.mask_augmentation", "dilate_iterations", "PELAGIA_MASK_AUGMENTATION_DILATE_ITERATIONS", int)
    _set_from_env(settings, "processing.mask_augmentation", "erode_kernel_w", "PELAGIA_MASK_AUGMENTATION_ERODE_KERNEL_W", int)
    _set_from_env(settings, "processing.mask_augmentation", "erode_kernel_h", "PELAGIA_MASK_AUGMENTATION_ERODE_KERNEL_H", int)
    _set_from_env(settings, "processing.mask_augmentation", "erode_iterations", "PELAGIA_MASK_AUGMENTATION_ERODE_ITERATIONS", int)
    _set_from_env(settings, "processing.mask_augmentation", "fill_holes", "PELAGIA_MASK_AUGMENTATION_FILL_HOLES", _env_bool)
    _set_from_env(settings, "processing.mask_augmentation", "remove_small_components", "PELAGIA_MASK_AUGMENTATION_REMOVE_SMALL_COMPONENTS", _env_bool)
    _set_from_env(settings, "processing.mask_augmentation", "min_component_area", "PELAGIA_MASK_AUGMENTATION_MIN_COMPONENT_AREA", float)
    _set_from_env(settings, "processing.mask_augmentation", "clear_border", "PELAGIA_MASK_AUGMENTATION_CLEAR_BORDER", _env_bool)

    _set_from_env(settings, "processing.roi_assembly", "method", "PELAGIA_ROI_ASSEMBLY_METHOD")
    _set_from_env(settings, "processing.roi_assembly", "connectivity", "PELAGIA_ROI_ASSEMBLY_CONNECTIVITY", int)

    _set_from_env(settings, "processing.roi_filter", "min_area", "PELAGIA_ROI_FILTER_MIN_AREA", float)
    _set_from_env(settings, "processing.roi_filter", "max_area", "PELAGIA_ROI_FILTER_MAX_AREA", float)
    _set_from_env(settings, "processing.roi_filter", "min_perimeter", "PELAGIA_ROI_FILTER_MIN_PERIMETER", float)
    _set_from_env(settings, "processing.roi_filter", "max_perimeter", "PELAGIA_ROI_FILTER_MAX_PERIMETER", float)
    _set_from_env(settings, "processing.roi_filter", "min_width", "PELAGIA_ROI_FILTER_MIN_WIDTH", float)
    _set_from_env(settings, "processing.roi_filter", "max_width", "PELAGIA_ROI_FILTER_MAX_WIDTH", float)
    _set_from_env(settings, "processing.roi_filter", "min_height", "PELAGIA_ROI_FILTER_MIN_HEIGHT", float)
    _set_from_env(settings, "processing.roi_filter", "max_height", "PELAGIA_ROI_FILTER_MAX_HEIGHT", float)
    _set_from_env(settings, "processing.roi_filter", "min_width_plus_height", "PELAGIA_ROI_FILTER_MIN_WIDTH_PLUS_HEIGHT", float)
    _set_from_env(settings, "processing.roi_filter", "max_width_plus_height", "PELAGIA_ROI_FILTER_MAX_WIDTH_PLUS_HEIGHT", float)

    _set_from_env(settings, "processing.roi_recording", "padding", "PELAGIA_ROI_RECORDING_PADDING", int)
    _set_from_env(settings, "processing.roi_recording", "roi_encoding", "PELAGIA_ROI_RECORDING_ROI_ENCODING")
    _set_from_env(settings, "processing.roi_recording", "zstd_min_bytes", "PELAGIA_ROI_RECORDING_ZSTD_MIN_BYTES", int)
    _set_from_env(settings, "processing.roi_recording", "always_store_mask", "PELAGIA_ROI_RECORDING_ALWAYS_STORE_MASK", _env_bool)
    _set_from_env(settings, "processing.roi_recording", "store_roi_payload_min_area", "PELAGIA_ROI_RECORDING_STORE_ROI_PAYLOAD_MIN_AREA", float)
    _set_from_env(settings, "processing.roi_recording", "store_roi_payload_min_width", "PELAGIA_ROI_RECORDING_STORE_ROI_PAYLOAD_MIN_WIDTH", float)
    _set_from_env(settings, "processing.roi_recording", "store_roi_payload_min_height", "PELAGIA_ROI_RECORDING_STORE_ROI_PAYLOAD_MIN_HEIGHT", float)
    _set_from_env(settings, "processing.roi_recording", "store_roi_payload_min_width_plus_height", "PELAGIA_ROI_RECORDING_STORE_ROI_PAYLOAD_MIN_WIDTH_PLUS_HEIGHT", float)

    _set_from_env(settings, "processing.preprocessing", "apply_mask", "PELAGIA_PREPROCESSING_APPLY_MASK", _env_bool)
    _set_from_env(settings, "processing.preprocessing", "mask_path", "PELAGIA_PREPROCESSING_MASK_PATH")
    _set_from_env(settings, "processing.preprocessing", "adaptive_background_subtraction", "PELAGIA_PREPROCESSING_ADAPTIVE_BACKGROUND_SUBTRACTION", _env_bool)
    _set_from_env(settings, "processing.preprocessing", "adaptive_background_period", "PELAGIA_PREPROCESSING_ADAPTIVE_BACKGROUND_PERIOD", int)
    _set_from_env(settings, "processing.preprocessing", "background_correction", "PELAGIA_PREPROCESSING_BACKGROUND_CORRECTION", _env_bool)
    _set_from_env(settings, "processing.preprocessing", "background_percentile", "PELAGIA_PREPROCESSING_BACKGROUND_PERCENTILE", float)
    _set_from_env(settings, "processing.preprocessing", "invert_intensity", "PELAGIA_PREPROCESSING_INVERT_INTENSITY", _env_bool)
    _set_from_env(settings, "processing.preprocessing", "crop_enabled", "PELAGIA_PREPROCESSING_CROP_ENABLED", _env_bool)
    _set_from_env(settings, "processing.preprocessing", "crop_x", "PELAGIA_PREPROCESSING_CROP_X", int)
    _set_from_env(settings, "processing.preprocessing", "crop_y", "PELAGIA_PREPROCESSING_CROP_Y", int)
    _set_from_env(settings, "processing.preprocessing", "crop_w", "PELAGIA_PREPROCESSING_CROP_W", int)
    _set_from_env(settings, "processing.preprocessing", "crop_h", "PELAGIA_PREPROCESSING_CROP_H", int)

    _set_from_env(settings, "processing.frame_storage", "image_encoding", "PELAGIA_FRAME_STORAGE_IMAGE_ENCODING")

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
    logging = _section(settings, "logging")
    mask_augmentation = _section(settings, "processing.mask_augmentation")
    roi_assembly = _section(settings, "processing.roi_assembly")
    roi_filter = _section(settings, "processing.roi_filter")
    roi_recording = _section(settings, "processing.roi_recording")
    video_ingest = _section(settings, "processing.video_ingest")
    flatfield = _section(settings, "processing.flatfield")
    thresholding = _section(settings, "processing.thresholding")
    preprocessing = _section(settings, "processing.preprocessing")
    frame_storage = _section(settings, "processing.frame_storage")
    thumbhash = _section(settings, "processing.thumbhash")
    mask_defaults = MaskAugmentationProcessingConfig()
    assembly_defaults = RoiAssemblyProcessingConfig()
    filter_defaults = RoiFilterProcessingConfig()
    recording_defaults = RoiRecordingProcessingConfig()

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
            cors_allow_origin_regex=str(
                api.get("cors_allow_origin_regex", APIConfig.cors_allow_origin_regex)
            ),
        ),
        logging=LoggingConfig(
            log_path=Path(logging.get("log_path", LoggingConfig.log_path)),
            file_name=str(logging.get("file_name", LoggingConfig.file_name)),
            level=str(logging.get("level", LoggingConfig.level)).upper(),
            console=bool(logging.get("console", LoggingConfig.console)),
            max_bytes=int(logging.get("max_bytes", LoggingConfig.max_bytes)),
            backup_count=int(logging.get("backup_count", LoggingConfig.backup_count)),
        ),
        processing=ProcessingConfig(
            mask_augmentation=MaskAugmentationProcessingConfig(
                enabled=bool(mask_augmentation.get("enabled", mask_defaults.enabled)),
                steps=_string_list(
                    mask_augmentation.get("steps", mask_defaults.steps)
                ),
                dilate_kernel_w=int(
                    mask_augmentation.get(
                        "dilate_kernel_w",
                        mask_defaults.dilate_kernel_w,
                    )
                ),
                dilate_kernel_h=int(
                    mask_augmentation.get(
                        "dilate_kernel_h",
                        mask_defaults.dilate_kernel_h,
                    )
                ),
                dilate_iterations=int(
                    mask_augmentation.get(
                        "dilate_iterations",
                        mask_defaults.dilate_iterations,
                    )
                ),
                erode_kernel_w=int(
                    mask_augmentation.get(
                        "erode_kernel_w",
                        mask_defaults.erode_kernel_w,
                    )
                ),
                erode_kernel_h=int(
                    mask_augmentation.get(
                        "erode_kernel_h",
                        mask_defaults.erode_kernel_h,
                    )
                ),
                erode_iterations=int(
                    mask_augmentation.get(
                        "erode_iterations",
                        mask_defaults.erode_iterations,
                    )
                ),
                open_kernel_w=int(
                    mask_augmentation.get(
                        "open_kernel_w",
                        mask_defaults.open_kernel_w,
                    )
                ),
                open_kernel_h=int(
                    mask_augmentation.get(
                        "open_kernel_h",
                        mask_defaults.open_kernel_h,
                    )
                ),
                open_iterations=int(
                    mask_augmentation.get(
                        "open_iterations",
                        mask_defaults.open_iterations,
                    )
                ),
                close_kernel_w=int(
                    mask_augmentation.get(
                        "close_kernel_w",
                        mask_defaults.close_kernel_w,
                    )
                ),
                close_kernel_h=int(
                    mask_augmentation.get(
                        "close_kernel_h",
                        mask_defaults.close_kernel_h,
                    )
                ),
                close_iterations=int(
                    mask_augmentation.get(
                        "close_iterations",
                        mask_defaults.close_iterations,
                    )
                ),
                fill_holes=bool(
                    mask_augmentation.get("fill_holes", mask_defaults.fill_holes)
                ),
                remove_small_components=bool(
                    mask_augmentation.get(
                        "remove_small_components",
                        mask_defaults.remove_small_components,
                    )
                ),
                min_component_area=float(
                    mask_augmentation.get(
                        "min_component_area",
                        mask_defaults.min_component_area,
                    )
                ),
                clear_border=bool(
                    mask_augmentation.get("clear_border", mask_defaults.clear_border)
                ),
            ),
            roi_assembly=RoiAssemblyProcessingConfig(
                method=str(roi_assembly.get("method", assembly_defaults.method)),
                connectivity=int(
                    roi_assembly.get("connectivity", assembly_defaults.connectivity)
                ),
            ),
            roi_filter=RoiFilterProcessingConfig(
                min_area=_optional_float(roi_filter.get("min_area", filter_defaults.min_area)),
                max_area=_optional_float(roi_filter.get("max_area", filter_defaults.max_area)),
                min_perimeter=float(
                    roi_filter.get("min_perimeter", filter_defaults.min_perimeter)
                ),
                max_perimeter=_optional_float(
                    roi_filter.get("max_perimeter", filter_defaults.max_perimeter)
                ),
                min_width=_optional_float(roi_filter.get("min_width", filter_defaults.min_width)),
                max_width=_optional_float(roi_filter.get("max_width", filter_defaults.max_width)),
                min_height=_optional_float(roi_filter.get("min_height", filter_defaults.min_height)),
                max_height=_optional_float(roi_filter.get("max_height", filter_defaults.max_height)),
                min_width_plus_height=_optional_float(
                    roi_filter.get(
                        "min_width_plus_height",
                        filter_defaults.min_width_plus_height,
                    )
                ),
                max_width_plus_height=_optional_float(
                    roi_filter.get(
                        "max_width_plus_height",
                        filter_defaults.max_width_plus_height,
                    )
                ),
            ),
            roi_recording=RoiRecordingProcessingConfig(
                padding=int(
                    roi_recording.get("padding", recording_defaults.padding)
                ),
                roi_encoding=str(
                    roi_recording.get("roi_encoding", recording_defaults.roi_encoding)
                ),
                zstd_min_bytes=int(
                    roi_recording.get("zstd_min_bytes", recording_defaults.zstd_min_bytes)
                ),
                always_store_mask=bool(
                    roi_recording.get(
                        "always_store_mask",
                        recording_defaults.always_store_mask,
                    )
                ),
                store_roi_payload_min_area=_optional_float(
                    roi_recording.get(
                        "store_roi_payload_min_area",
                        recording_defaults.store_roi_payload_min_area,
                    )
                ),
                store_roi_payload_min_width=_optional_float(
                    roi_recording.get(
                        "store_roi_payload_min_width",
                        recording_defaults.store_roi_payload_min_width,
                    )
                ),
                store_roi_payload_min_height=_optional_float(
                    roi_recording.get(
                        "store_roi_payload_min_height",
                        recording_defaults.store_roi_payload_min_height,
                    )
                ),
                store_roi_payload_min_width_plus_height=_optional_float(
                    roi_recording.get(
                        "store_roi_payload_min_width_plus_height",
                        recording_defaults.store_roi_payload_min_width_plus_height,
                    )
                ),
            ),
            video_ingest=VideoIngestProcessingConfig(
                n_tile=int(video_ingest.get("n_tile", VideoIngestProcessingConfig.n_tile)),
            ),
            flatfield=FlatfieldProcessingConfig(
                flatfield_correction=bool(
                    flatfield.get("flatfield_correction", FlatfieldProcessingConfig.flatfield_correction)
                ),
                flatfield_q=float(flatfield.get("flatfield_q", FlatfieldProcessingConfig.flatfield_q)),
                flatfield_axis=int(flatfield.get("flatfield_axis", FlatfieldProcessingConfig.flatfield_axis)),
            ),
            thresholding=ThresholdingProcessingConfig(
                method=str(thresholding.get("method", ThresholdingProcessingConfig.method)),
                manual_threshold=float(
                    thresholding.get("manual_threshold", ThresholdingProcessingConfig.manual_threshold)
                ),
                thresholding_maximum_value=float(
                    thresholding.get(
                        "thresholding_maximum_value",
                        ThresholdingProcessingConfig.thresholding_maximum_value,
                    )
                ),
                bounded_otsu_min_contrast=float(
                    thresholding.get(
                        "bounded_otsu_min_contrast",
                        ThresholdingProcessingConfig.bounded_otsu_min_contrast,
                    )
                ),
                bounded_otsu_max_foreground_fraction=float(
                    thresholding.get(
                        "bounded_otsu_max_foreground_fraction",
                        ThresholdingProcessingConfig.bounded_otsu_max_foreground_fraction,
                    )
                ),
                canny_enabled=bool(thresholding.get("canny_enabled", ThresholdingProcessingConfig.canny_enabled)),
                canny_low_threshold=float(
                    thresholding.get("canny_low_threshold", ThresholdingProcessingConfig.canny_low_threshold)
                ),
                canny_high_threshold=float(
                    thresholding.get("canny_high_threshold", ThresholdingProcessingConfig.canny_high_threshold)
                ),
                canny_blur_kernel=int(
                    thresholding.get("canny_blur_kernel", ThresholdingProcessingConfig.canny_blur_kernel)
                ),
                adaptive_block_size=int(
                    thresholding.get("adaptive_block_size", ThresholdingProcessingConfig.adaptive_block_size)
                ),
                adaptive_c=float(thresholding.get("adaptive_c", ThresholdingProcessingConfig.adaptive_c)),
                percentile_background_percentile=float(
                    thresholding.get(
                        "percentile_background_percentile",
                        ThresholdingProcessingConfig.percentile_background_percentile,
                    )
                ),
                percentile_min_contrast=float(
                    thresholding.get(
                        "percentile_min_contrast",
                        ThresholdingProcessingConfig.percentile_min_contrast,
                    )
                ),
                hysteresis_low_threshold=float(
                    thresholding.get(
                        "hysteresis_low_threshold",
                        ThresholdingProcessingConfig.hysteresis_low_threshold,
                    )
                ),
                hysteresis_high_threshold=float(
                    thresholding.get(
                        "hysteresis_high_threshold",
                        ThresholdingProcessingConfig.hysteresis_high_threshold,
                    )
                ),
                hysteresis_connectivity=int(
                    thresholding.get(
                        "hysteresis_connectivity",
                        ThresholdingProcessingConfig.hysteresis_connectivity,
                    )
                ),
                sobel_percentile=float(
                    thresholding.get("sobel_percentile", ThresholdingProcessingConfig.sobel_percentile)
                ),
                sobel_threshold=(
                    None
                    if thresholding.get("sobel_threshold", ThresholdingProcessingConfig.sobel_threshold) is None
                    else float(thresholding["sobel_threshold"])
                ),
                sobel_kernel_size=int(
                    thresholding.get("sobel_kernel_size", ThresholdingProcessingConfig.sobel_kernel_size)
                ),
            ),
            preprocessing=PreprocessingConfig(
                apply_mask=bool(preprocessing.get("apply_mask", PreprocessingConfig.apply_mask)),
                mask_path=preprocessing.get("mask_path", PreprocessingConfig.mask_path),
                adaptive_background_subtraction=bool(
                    preprocessing.get(
                        "adaptive_background_subtraction",
                        PreprocessingConfig.adaptive_background_subtraction,
                    )
                ),
                adaptive_background_period=int(
                    preprocessing.get(
                        "adaptive_background_period",
                        PreprocessingConfig.adaptive_background_period,
                    )
                ),
                background_correction=bool(
                    preprocessing.get("background_correction", PreprocessingConfig.background_correction)
                ),
                background_percentile=float(
                    preprocessing.get("background_percentile", PreprocessingConfig.background_percentile)
                ),
                invert_intensity=bool(
                    preprocessing.get("invert_intensity", PreprocessingConfig.invert_intensity)
                ),
                crop_enabled=bool(preprocessing.get("crop_enabled", PreprocessingConfig.crop_enabled)),
                crop_x=_optional_int(preprocessing.get("crop_x", PreprocessingConfig.crop_x)),
                crop_y=_optional_int(preprocessing.get("crop_y", PreprocessingConfig.crop_y)),
                crop_w=_optional_int(preprocessing.get("crop_w", PreprocessingConfig.crop_w)),
                crop_h=_optional_int(preprocessing.get("crop_h", PreprocessingConfig.crop_h)),
            ),
            frame_storage=FrameStorageProcessingConfig(
                image_encoding=frame_storage.get(
                    "image_encoding",
                    image_data_storage.get("encoding", FrameStorageProcessingConfig.image_encoding),
                ),
            ),
            thumbhash=ThumbhashProcessingConfig(
                max_dim=int(thumbhash.get("max_dim", ThumbhashProcessingConfig.max_dim)),
            ),
        ),
    )


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    raise ValueError("Config list value must be a string, list, or tuple.")
