import pytest

from Pelagia.config import CoreConfig, ImageDataStorageConfig


def test_image_data_storage_config_defaults_to_png():
    assert ImageDataStorageConfig().encoding == "png"


def test_image_data_storage_config_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        ImageDataStorageConfig(encoding="jpeg")


def test_core_config_reads_image_data_storage_encoding_from_env(monkeypatch):
    monkeypatch.setenv("PELAGIA_IMAGE_DATA_STORAGE_ENCODING", "ZSTD")

    config = CoreConfig.from_env()

    assert config.image_data_storage.encoding == "zstd"
