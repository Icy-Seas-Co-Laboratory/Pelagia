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


def test_core_config_load_applies_auth_env(monkeypatch):
    monkeypatch.setenv("PELAGIA_AUTH_ENABLED", "false")
    monkeypatch.setenv("PELAGIA_AUTH_SESSION_TTL_SECONDS", "3600")
    monkeypatch.setenv("PELAGIA_AUTH_DEV_PROJECT_KEY", "sandbox")

    config = CoreConfig.load(local_config_path=None)

    assert config.auth.enabled is False
    assert config.auth.session_ttl_seconds == 3600
    assert config.auth.dev_project_key == "sandbox"


def test_core_config_load_applies_kvstore_backend_env(monkeypatch):
    monkeypatch.setenv("PELAGIA_KVSTORE_BACKEND", "kvstore2")
    monkeypatch.setenv("PELAGIA_KVSTORE_MAX_BLOB_BYTES", "123456")

    config = CoreConfig.load(local_config_path=None)

    assert config.kvstore.backend == "kvstore2"
    assert config.kvstore.max_blob_bytes == 123456


def test_logging_config_defaults_to_core_file_logger():
    config = LoggingConfig()

    assert config.log_path.as_posix() == "logs"
    assert config.file_name == "pelagia.log"
    assert config.level == "INFO"
    assert config.console is True


def test_core_config_loads_packaged_defaults_without_local_config():
    config = CoreConfig.load(local_config_path=None, use_env=False)

    assert config.database.schema_name == "pelagia"
    assert config.kvstore.backend == "kvstore"
    assert config.kvstore.root_path.as_posix() == "data/kvstore"
    assert config.kvstore.max_blob_bytes == 67108864
    assert config.file_browser.root_path_kvstore.as_posix() == "data/kvstore"
    assert config.file_browser.root_path_import_dir.as_posix() == "data/import"
    assert config.file_browser.allowed_root_paths == []
    assert config.image_data_storage.encoding == "zstd"
    assert config.auth.enabled is True
    assert config.auth.session_ttl_seconds == 604800
    assert config.auth.dev_project_key == "default"
    assert config.auth.bootstrap_admin_username is None
    assert config.auth.bootstrap_admin_password is None
    assert config.artifacts.local_root.as_posix() == ".pelagia"
    assert config.artifacts.models.builtin_enabled is True
    assert config.artifacts.models.local_path.as_posix() == ".pelagia/models"
    assert config.artifacts.models.metadata_filename == "metadata.toml"
    assert config.artifacts.plugins.builtin_enabled is True
    assert config.artifacts.plugins.local_path.as_posix() == ".pelagia/plugins"
    assert config.artifacts.plugins.metadata_filename == "metadata.toml"
    assert config.processing.video_ingest.n_tile == 4
    assert config.processing.flatfield.flatfield_correction is True
    assert config.processing.flatfield.flatfield_q == 0.5
    assert config.processing.flatfield.flatfield_axis == 0
    assert config.processing.flatfield.flatfield_min_field_value == 25.0
    assert config.processing.flatfield.flatfield_max_field_value == 255.0
    assert config.processing.thresholding.method == "manual"
    assert config.processing.thresholding.thresholding_maximum_value == 100.0
    assert config.processing.thresholding.bounded_otsu_min_contrast == 50.0
    assert config.processing.thresholding.canny_low_threshold == 60.0
    assert config.processing.thresholding.adaptive_block_size == 31
    assert config.processing.mask_augmentation.enabled is False
    assert config.processing.mask_augmentation.steps == ["dilate"]
    assert config.processing.roi_assembly.method == "connected_components"
    assert config.processing.roi_assembly.connectivity == 8
    assert config.processing.roi_filter.min_perimeter is None
    assert config.processing.roi_filter.min_area is None
    assert config.processing.roi_recording.padding == 50
    assert config.processing.roi_recording.roi_encoding == "zstd"
    assert config.processing.roi_recording.zstd_min_bytes == 8192
    assert config.processing.roi_recording.always_store_mask is True
    assert config.processing.preprocessing.apply_mask is False
    assert config.processing.preprocessing.mask_path is None
    assert config.processing.preprocessing.adaptive_background_subtraction is False
    assert config.processing.preprocessing.adaptive_background_period == 50
    assert config.processing.preprocessing.background_correction is False
    assert config.processing.preprocessing.background_min_field_value == 50.0
    assert config.processing.preprocessing.background_max_field_value == 255.0
    assert config.processing.preprocessing.invert_intensity is True
    assert config.processing.preprocessing.crop_enabled is False
    assert config.processing.preprocessing.crop_x is None
    assert config.processing.preprocessing.crop_y is None
    assert config.processing.preprocessing.crop_w is None
    assert config.processing.preprocessing.crop_h is None
    assert config.processing.frame_storage.image_encoding == "zstd"
    assert config.processing.thumbhash.max_dim == 100
    assert config.processing.roi_refinement.enabled is True
    assert config.processing.roi_refinement.model_kind == "keras_artifact"
    assert config.processing.roi_refinement.model_ref == "builtin:model/roi_refinement/example_model"
    assert config.processing.roi_refinement.model_run_dir is None
    assert config.processing.roi_refinement.model_artifact == "auto"
    assert config.processing.roi_refinement.tile_size == 256
    assert config.processing.roi_refinement.overlap_fraction == 0.25
    assert config.processing.roi_refinement.max_iterations == 5
    assert config.processing.roi_refinement.expansion_pixels == 256
    assert config.processing.roi_refinement.edge_touch_margin == 2
    assert config.processing.roi_refinement.output_threshold == 0.5
    assert config.processing.roi_refinement.batch_size is None
    assert config.processing.roi_refinement.encoding is None
    assert config.processing.roi_refinement.overlap_reconciliation_enabled is True
    assert config.processing.roi_refinement.overlap_iou_threshold == 0.5
    assert config.processing.roi_refinement.overlap_containment_threshold == 0.8
    assert config.processing.roi_refinement.residual_discovery_enabled is True
    assert config.processing.roi_refinement.residual_max_iterations == 1
    assert config.processing.roi_refinement.residual_roi_assembly_method is None
    assert config.processing.roi_refinement.residual_roi_assembly_connectivity == 8
    assert config.processing.roi_refinement.residual_min_area is None
    assert config.processing.roi_refinement.residual_padding is None
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
        backend = "kvstore2"
        root_path = "/tmp/pelagia-kv"
        max_blob_bytes = 12345

        [file_browser]
        root_path_kvstore = "/tmp/browser-kv"
        root_path_import_dir = "/tmp/raw-assets"
        allowed_root_paths = ["/tmp/shared", "/mnt/cruise"]

        [image_data_storage]
        encoding = "raw"

        [artifacts]
        local_root = "/tmp/pelagia-artifacts"

        [artifacts.models]
        builtin_enabled = false
        local_path = "/tmp/pelagia-artifacts/models"
        metadata_filename = "model.toml"

        [artifacts.plugins]
        builtin_enabled = false
        local_path = "/tmp/pelagia-artifacts/plugins"
        metadata_filename = "plugin.toml"

        [processing.video_ingest]
        n_tile = 4

        [processing.flatfield]
        flatfield_correction = false
        flatfield_q = 0.8
        flatfield_axis = 1
        flatfield_min_field_value = 2
        flatfield_max_field_value = 200

        [processing.thresholding]
        method = "bounded_otsu_canny"
        thresholding_maximum_value = 180
        bounded_otsu_min_contrast = 22
        canny_low_threshold = 12
        adaptive_block_size = 17

        [processing.mask_augmentation]
        steps = ["open", "fill_holes"]
        open_kernel_w = 5
        open_kernel_h = 5

        [processing.roi_assembly]
        method = "contours"
        connectivity = 4

        [processing.roi_filter]
        min_area = 5
        min_perimeter = 12.5
        min_width_plus_height = 9

        [processing.roi_recording]
        padding = 8
        roi_encoding = "png"
        store_roi_payload_min_area = 25

        [processing.preprocessing]
        apply_mask = false
        mask_path = "/tmp/mask.png"
        adaptive_background_subtraction = true
        adaptive_background_period = 12
        background_correction = true
        background_min_field_value = 3
        background_max_field_value = 220
        invert_intensity = true
        crop_enabled = true
        crop_x = 10
        crop_y = 20
        crop_w = 300
        crop_h = 400

        [processing.frame_storage]
        image_encoding = "png"

        [processing.thumbhash]
        max_dim = 64

        [processing.roi_refinement]
        enabled = true
        model_kind = "oracle_builder_unet"
        model_ref = "builtin:model/roi_refinement/example_model"
        model_run_dir = "../oracle-builder/runs/unet-test"
        model_artifact = "savedmodel"
        tile_size = 128
        overlap_fraction = 0.5
        max_iterations = 4
        expansion_pixels = 32
        edge_touch_margin = 2
        output_threshold = 0.6
        batch_size = 6
        encoding = "png"
        overlap_reconciliation_enabled = false
        overlap_iou_threshold = 0.3
        overlap_containment_threshold = 0.7
        residual_discovery_enabled = true
        residual_max_iterations = 2
        residual_roi_assembly_method = "contours"
        residual_roi_assembly_connectivity = 4
        residual_min_area = 3
        residual_min_width = 2
        residual_min_height = 2
        residual_min_width_plus_height = 5
        residual_padding = 1
        """,
        encoding="utf-8",
    )

    config = CoreConfig.load(config_path=config_path, use_env=False)

    assert config.database.schema_name == "pelagia_custom"
    assert config.database.dsn == "postgresql://postgres:postgres@localhost:5432/pelagia"
    assert config.queue.lease_seconds == 42
    assert config.kvstore.backend == "kvstore2"
    assert config.kvstore.root_path.as_posix() == "/tmp/pelagia-kv"
    assert config.kvstore.max_blob_bytes == 12345
    assert config.file_browser.root_path_kvstore.as_posix() == "/tmp/browser-kv"
    assert config.file_browser.root_path_import_dir.as_posix() == "/tmp/raw-assets"
    assert [path.as_posix() for path in config.file_browser.allowed_root_paths] == [
        "/tmp/shared",
        "/mnt/cruise",
    ]
    assert config.image_data_storage.encoding == "raw"
    assert config.artifacts.local_root.as_posix() == "/tmp/pelagia-artifacts"
    assert config.artifacts.models.builtin_enabled is False
    assert config.artifacts.models.local_path.as_posix() == "/tmp/pelagia-artifacts/models"
    assert config.artifacts.models.metadata_filename == "model.toml"
    assert config.artifacts.plugins.builtin_enabled is False
    assert config.artifacts.plugins.local_path.as_posix() == "/tmp/pelagia-artifacts/plugins"
    assert config.artifacts.plugins.metadata_filename == "plugin.toml"
    assert config.processing.video_ingest.n_tile == 4
    assert config.processing.flatfield.flatfield_correction is False
    assert config.processing.flatfield.flatfield_q == 0.8
    assert config.processing.flatfield.flatfield_axis == 1
    assert config.processing.flatfield.flatfield_min_field_value == 2.0
    assert config.processing.flatfield.flatfield_max_field_value == 200.0
    assert config.processing.thresholding.method == "bounded_otsu_canny"
    assert config.processing.thresholding.thresholding_maximum_value == 180.0
    assert config.processing.thresholding.bounded_otsu_min_contrast == 22.0
    assert config.processing.thresholding.canny_low_threshold == 12.0
    assert config.processing.thresholding.adaptive_block_size == 17
    assert config.processing.mask_augmentation.steps == ["open", "fill_holes"]
    assert config.processing.mask_augmentation.open_kernel_w == 5
    assert config.processing.mask_augmentation.open_kernel_h == 5
    assert config.processing.roi_assembly.method == "contours"
    assert config.processing.roi_assembly.connectivity == 4
    assert config.processing.roi_filter.min_area == 5.0
    assert config.processing.roi_filter.min_width_plus_height == 9.0
    assert config.processing.roi_filter.min_perimeter == 12.5
    assert config.processing.roi_recording.padding == 8
    assert config.processing.roi_recording.roi_encoding == "png"
    assert config.processing.roi_recording.store_roi_payload_min_area == 25.0
    assert config.processing.preprocessing.apply_mask is False
    assert config.processing.preprocessing.mask_path == "/tmp/mask.png"
    assert config.processing.preprocessing.adaptive_background_subtraction is True
    assert config.processing.preprocessing.adaptive_background_period == 12
    assert config.processing.preprocessing.background_correction is True
    assert config.processing.preprocessing.background_min_field_value == 3.0
    assert config.processing.preprocessing.background_max_field_value == 220.0
    assert config.processing.preprocessing.invert_intensity is True
    assert config.processing.preprocessing.crop_enabled is True
    assert config.processing.preprocessing.crop_x == 10
    assert config.processing.preprocessing.crop_y == 20
    assert config.processing.preprocessing.crop_w == 300
    assert config.processing.preprocessing.crop_h == 400
    assert config.processing.frame_storage.image_encoding == "png"
    assert config.processing.thumbhash.max_dim == 64
    assert config.processing.roi_refinement.enabled is True
    assert config.processing.roi_refinement.model_kind == "oracle_builder_unet"
    assert config.processing.roi_refinement.model_ref == "builtin:model/roi_refinement/example_model"
    assert config.processing.roi_refinement.model_run_dir == "../oracle-builder/runs/unet-test"
    assert config.processing.roi_refinement.model_artifact == "savedmodel"
    assert config.processing.roi_refinement.tile_size == 128
    assert config.processing.roi_refinement.overlap_fraction == 0.5
    assert config.processing.roi_refinement.max_iterations == 4
    assert config.processing.roi_refinement.expansion_pixels == 32
    assert config.processing.roi_refinement.edge_touch_margin == 2
    assert config.processing.roi_refinement.output_threshold == 0.6
    assert config.processing.roi_refinement.batch_size == 6
    assert config.processing.roi_refinement.encoding == "png"
    assert config.processing.roi_refinement.overlap_reconciliation_enabled is False
    assert config.processing.roi_refinement.overlap_iou_threshold == 0.3
    assert config.processing.roi_refinement.overlap_containment_threshold == 0.7
    assert config.processing.roi_refinement.residual_discovery_enabled is True
    assert config.processing.roi_refinement.residual_max_iterations == 2
    assert config.processing.roi_refinement.residual_roi_assembly_method == "contours"
    assert config.processing.roi_refinement.residual_roi_assembly_connectivity == 4
    assert config.processing.roi_refinement.residual_min_area == 3.0
    assert config.processing.roi_refinement.residual_min_width == 2.0
    assert config.processing.roi_refinement.residual_min_height == 2.0
    assert config.processing.roi_refinement.residual_min_width_plus_height == 5.0
    assert config.processing.roi_refinement.residual_padding == 1


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
    monkeypatch.setenv("PELAGIA_FILE_BROWSER_ROOT_PATH_KVSTORE", "/tmp/env-browser-kv")
    monkeypatch.setenv("PELAGIA_FILE_BROWSER_ROOT_PATH_IMPORT_DIR", "/tmp/env-import")
    monkeypatch.setenv("PELAGIA_FILE_BROWSER_ALLOWED_ROOT_PATHS", "/tmp/env-raw,/mnt/env-cruise")
    monkeypatch.setenv("PELAGIA_ARTIFACTS_LOCAL_ROOT", "/tmp/env-pelagia-artifacts")
    monkeypatch.setenv("PELAGIA_ARTIFACT_MODELS_BUILTIN_ENABLED", "false")
    monkeypatch.setenv("PELAGIA_ARTIFACT_MODELS_LOCAL_PATH", "/tmp/env-pelagia-artifacts/models")
    monkeypatch.setenv("PELAGIA_ARTIFACT_PLUGINS_BUILTIN_ENABLED", "false")
    monkeypatch.setenv("PELAGIA_ARTIFACT_PLUGINS_LOCAL_PATH", "/tmp/env-pelagia-artifacts/plugins")
    monkeypatch.setenv("PELAGIA_ROI_FILTER_MIN_PERIMETER", "9")
    monkeypatch.setenv("PELAGIA_VIDEO_INGEST_N_TILE", "5")
    monkeypatch.setenv("PELAGIA_FLATFIELD_CORRECTION", "false")
    monkeypatch.setenv("PELAGIA_FLATFIELD_Q", "0.7")
    monkeypatch.setenv("PELAGIA_FLATFIELD_AXIS", "1")
    monkeypatch.setenv("PELAGIA_FLATFIELD_MIN_FIELD_VALUE", "4")
    monkeypatch.setenv("PELAGIA_FLATFIELD_MAX_FIELD_VALUE", "210")
    monkeypatch.setenv("PELAGIA_THRESHOLDING_METHOD", "adaptive_mean")
    monkeypatch.setenv("PELAGIA_THRESHOLDING_MAXIMUM_VALUE", "170")
    monkeypatch.setenv("PELAGIA_THRESHOLDING_CANNY_LOW_THRESHOLD", "9")
    monkeypatch.setenv("PELAGIA_THRESHOLDING_ADAPTIVE_BLOCK_SIZE", "19")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_APPLY_MASK", "false")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_MASK_PATH", "/tmp/env-mask.png")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_ADAPTIVE_BACKGROUND_SUBTRACTION", "true")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_ADAPTIVE_BACKGROUND_PERIOD", "20")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_BACKGROUND_CORRECTION", "true")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_BACKGROUND_MIN_FIELD_VALUE", "5")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_BACKGROUND_MAX_FIELD_VALUE", "230")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_INVERT_INTENSITY", "true")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_CROP_ENABLED", "true")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_CROP_X", "1")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_CROP_Y", "2")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_CROP_W", "3")
    monkeypatch.setenv("PELAGIA_PREPROCESSING_CROP_H", "4")
    monkeypatch.setenv("PELAGIA_FRAME_STORAGE_IMAGE_ENCODING", "raw")
    monkeypatch.setenv("PELAGIA_LOG_PATH", "/tmp/env-pelagia-logs")
    monkeypatch.setenv("PELAGIA_LOG_LEVEL", "warning")
    monkeypatch.setenv("PELAGIA_LOG_CONSOLE", "false")

    config = CoreConfig.load(config_path=config_path)

    assert config.database.schema_name == "pelagia_from_env"
    assert config.image_data_storage.encoding == "zstd"
    assert config.file_browser.root_path_kvstore.as_posix() == "/tmp/env-browser-kv"
    assert config.file_browser.root_path_import_dir.as_posix() == "/tmp/env-import"
    assert [path.as_posix() for path in config.file_browser.allowed_root_paths] == [
        "/tmp/env-raw",
        "/mnt/env-cruise",
    ]
    assert config.artifacts.local_root.as_posix() == "/tmp/env-pelagia-artifacts"
    assert config.artifacts.models.builtin_enabled is False
    assert config.artifacts.models.local_path.as_posix() == "/tmp/env-pelagia-artifacts/models"
    assert config.artifacts.plugins.builtin_enabled is False
    assert config.artifacts.plugins.local_path.as_posix() == "/tmp/env-pelagia-artifacts/plugins"
    assert config.processing.roi_filter.min_perimeter == 9.0
    assert config.processing.video_ingest.n_tile == 5
    assert config.processing.flatfield.flatfield_correction is False
    assert config.processing.flatfield.flatfield_q == 0.7
    assert config.processing.flatfield.flatfield_axis == 1
    assert config.processing.flatfield.flatfield_min_field_value == 4.0
    assert config.processing.flatfield.flatfield_max_field_value == 210.0
    assert config.processing.thresholding.method == "adaptive_mean"
    assert config.processing.thresholding.thresholding_maximum_value == 170.0
    assert config.processing.thresholding.canny_low_threshold == 9.0
    assert config.processing.thresholding.adaptive_block_size == 19
    assert config.processing.preprocessing.apply_mask is False
    assert config.processing.preprocessing.mask_path == "/tmp/env-mask.png"
    assert config.processing.preprocessing.adaptive_background_subtraction is True
    assert config.processing.preprocessing.adaptive_background_period == 20
    assert config.processing.preprocessing.background_correction is True
    assert config.processing.preprocessing.background_min_field_value == 5.0
    assert config.processing.preprocessing.background_max_field_value == 230.0
    assert config.processing.preprocessing.invert_intensity is True
    assert config.processing.preprocessing.crop_enabled is True
    assert config.processing.preprocessing.crop_x == 1
    assert config.processing.preprocessing.crop_y == 2
    assert config.processing.preprocessing.crop_w == 3
    assert config.processing.preprocessing.crop_h == 4
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
