import pytest

from Pelagia.config import CoreConfig, ImageDataStorageConfig, LoggingConfig


def test_image_data_storage_config_defaults_to_zstd():
    assert ImageDataStorageConfig().encoding == "zstd"


def test_image_data_storage_config_accepts_jpg():
    assert ImageDataStorageConfig(encoding="JPG").encoding == "jpg"


def test_image_data_storage_config_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        ImageDataStorageConfig(encoding="jpeg")


def test_core_config_load_applies_image_data_storage_encoding_env(monkeypatch):
    monkeypatch.setenv("PELAGIA_IMAGE_DATA_STORAGE_ENCODING", "ZSTD")

    config = CoreConfig.load(local_config_path=None)

    assert config.image_data_storage.encoding == "zstd"


def test_logging_config_defaults_to_core_file_logger():
    config = LoggingConfig()

    assert config.log_path.as_posix() == "logs"
    assert config.file_name == "pelagia.log"
    assert config.level == "INFO"
    assert config.console is True


def test_core_config_loads_packaged_defaults_without_local_config():
    config = CoreConfig.load(local_config_path=None, use_env=False)

    assert config.database.schema_name == "pelagia"
    assert config.kvstore.root_path.as_posix() == "data/kvstore"
    assert config.image_data_storage.encoding == "zstd"
    assert config.processing.segmentation.min_perimeter == 100.0
    assert config.processing.segmentation.padding == 100
    assert config.processing.segmentation.roi_encoding == "zstd"
    assert config.processing.video_ingest.n_tile == 2
    assert config.processing.video_ingest.flatfield_q == 0.9
    assert config.processing.video_ingest.adaptive_background_subtraction is False
    assert config.processing.video_ingest.adaptive_background_period == 50
    assert config.processing.video_ingest.frame_mask is False
    assert config.processing.video_ingest.frame_mask_path is None
    assert config.processing.thresholding.thresholding_maximum_value == 255.0
    assert config.processing.frame_storage.image_encoding == "jpg"
    assert config.processing.thumbhash.max_dim == 100
    assert config.logging.log_path.as_posix() == "logs"
    assert config.logging.file_name == "pelagia.log"
    assert config.logging.level == "INFO"


def test_core_config_loads_explicit_toml_overrides(tmp_path):
    config_path = tmp_path / "pelagia.config.toml"
    config_path.write_text(
        """
        [database]
        schema_name = "pelagia_custom"

        [queue]
        lease_seconds = 42

        [kvstore]
        root_path = "/tmp/pelagia-kv"

        [image_data_storage]
        encoding = "raw"

        [processing.segmentation]
        min_perimeter = 12.5
        padding = 3

        [processing.video_ingest]
        n_tile = 4
        adaptive_background_subtraction = true
        adaptive_background_period = 12
        frame_mask = true
        frame_mask_path = "/tmp/mask.png"

        [processing.thresholding]
        thresholding_maximum_value = 180

        [processing.frame_storage]
        image_encoding = "png"
        thumbhash_max_dim = 64
        """,
        encoding="utf-8",
    )

    config = CoreConfig.load(config_path=config_path, use_env=False)

    assert config.database.schema_name == "pelagia_custom"
    assert config.database.dsn == "postgresql://postgres:postgres@localhost:5432/pelagia"
    assert config.queue.lease_seconds == 42
    assert config.kvstore.root_path.as_posix() == "/tmp/pelagia-kv"
    assert config.image_data_storage.encoding == "raw"
    assert config.processing.segmentation.min_perimeter == 12.5
    assert config.processing.segmentation.padding == 3
    assert config.processing.video_ingest.n_tile == 4
    assert config.processing.video_ingest.adaptive_background_subtraction is True
    assert config.processing.video_ingest.adaptive_background_period == 12
    assert config.processing.video_ingest.frame_mask is True
    assert config.processing.video_ingest.frame_mask_path == "/tmp/mask.png"
    assert config.processing.thresholding.thresholding_maximum_value == 180.0
    assert config.processing.frame_storage.image_encoding == "png"
    assert config.processing.frame_storage.thumbhash_max_dim == 64


def test_core_config_env_overrides_toml(monkeypatch, tmp_path):
    config_path = tmp_path / "pelagia.config.toml"
    config_path.write_text(
        """
        [database]
        schema_name = "pelagia_from_toml"

        [image_data_storage]
        encoding = "png"
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("PELAGIA_DATABASE_SCHEMA", "pelagia_from_env")
    monkeypatch.setenv("PELAGIA_IMAGE_DATA_STORAGE_ENCODING", "zstd")
    monkeypatch.setenv("PELAGIA_SEGMENTATION_MIN_PERIMETER", "9")
    monkeypatch.setenv("PELAGIA_VIDEO_INGEST_N_TILE", "5")
    monkeypatch.setenv("PELAGIA_THRESHOLDING_MAXIMUM_VALUE", "170")
    monkeypatch.setenv("PELAGIA_VIDEO_INGEST_ADAPTIVE_BACKGROUND_SUBTRACTION", "true")
    monkeypatch.setenv("PELAGIA_VIDEO_INGEST_ADAPTIVE_BACKGROUND_PERIOD", "20")
    monkeypatch.setenv("PELAGIA_VIDEO_INGEST_FRAME_MASK", "true")
    monkeypatch.setenv("PELAGIA_VIDEO_INGEST_FRAME_MASK_PATH", "/tmp/env-mask.png")
    monkeypatch.setenv("PELAGIA_FRAME_STORAGE_IMAGE_ENCODING", "raw")
    monkeypatch.setenv("PELAGIA_LOG_PATH", "/tmp/env-pelagia-logs")
    monkeypatch.setenv("PELAGIA_LOG_LEVEL", "warning")
    monkeypatch.setenv("PELAGIA_LOG_CONSOLE", "false")

    config = CoreConfig.load(config_path=config_path)

    assert config.database.schema_name == "pelagia_from_env"
    assert config.image_data_storage.encoding == "zstd"
    assert config.processing.segmentation.min_perimeter == 9.0
    assert config.processing.video_ingest.n_tile == 5
    assert config.processing.video_ingest.adaptive_background_subtraction is True
    assert config.processing.video_ingest.adaptive_background_period == 20
    assert config.processing.video_ingest.frame_mask is True
    assert config.processing.video_ingest.frame_mask_path == "/tmp/env-mask.png"
    assert config.processing.thresholding.thresholding_maximum_value == 170.0
    assert config.processing.frame_storage.image_encoding == "raw"
    assert config.logging.log_path.as_posix() == "/tmp/env-pelagia-logs"
    assert config.logging.level == "WARNING"
    assert config.logging.console is False


def test_core_config_uses_pelagia_config_env(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [api]
        port = 9001

        [logging]
        log_path = "/tmp/pelagia-logs"
        level = "debug"
        console = false
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("PELAGIA_CONFIG", str(config_path))

    config = CoreConfig.load(use_env=False)

    assert config.api.port == 9001
    assert config.logging.log_path.as_posix() == "/tmp/pelagia-logs"
    assert config.logging.level == "DEBUG"
    assert config.logging.console is False
