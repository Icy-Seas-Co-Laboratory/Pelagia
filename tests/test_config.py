import pytest

from Pelagia.config import CoreConfig, ImageDataStorageConfig


def test_image_data_storage_config_defaults_to_zstd():
    assert ImageDataStorageConfig().encoding == "zstd"


def test_image_data_storage_config_rejects_unknown_encoding():
    with pytest.raises(ValueError):
        ImageDataStorageConfig(encoding="jpeg")


def test_core_config_load_applies_image_data_storage_encoding_env(monkeypatch):
    monkeypatch.setenv("PELAGIA_IMAGE_DATA_STORAGE_ENCODING", "ZSTD")

    config = CoreConfig.load(local_config_path=None)

    assert config.image_data_storage.encoding == "zstd"


def test_core_config_loads_packaged_defaults_without_local_config():
    config = CoreConfig.load(local_config_path=None, use_env=False)

    assert config.database.schema_name == "pelagia"
    assert config.kvstore.root_path.as_posix() == "data/kvstore"
    assert config.image_data_storage.encoding == "zstd"


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
        """,
        encoding="utf-8",
    )

    config = CoreConfig.load(config_path=config_path, use_env=False)

    assert config.database.schema_name == "pelagia_custom"
    assert config.database.dsn == "postgresql://postgres:postgres@localhost:5432/pelagia"
    assert config.queue.lease_seconds == 42
    assert config.kvstore.root_path.as_posix() == "/tmp/pelagia-kv"
    assert config.image_data_storage.encoding == "raw"


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

    config = CoreConfig.load(config_path=config_path)

    assert config.database.schema_name == "pelagia_from_env"
    assert config.image_data_storage.encoding == "zstd"


def test_core_config_uses_pelagia_config_env(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        """
        [api]
        port = 9001
        """,
        encoding="utf-8",
    )
    monkeypatch.setenv("PELAGIA_CONFIG", str(config_path))

    config = CoreConfig.load(use_env=False)

    assert config.api.port == 9001
