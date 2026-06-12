from pathlib import Path

from Pelagia.config import CoreConfig
from Pelagia.services.models import ArtifactRegistry, ModelService, load_artifact_metadata


def test_model_service_discovers_local_model_metadata(tmp_path):
    model_dir = tmp_path / "models" / "roi_refinement" / "demo_unet"
    model_dir.mkdir(parents=True)
    metadata_path = model_dir / "metadata.toml"
    metadata_path.write_text(
        """
        name = "demo_unet"
        kind = "roi_refinement"
        version = "0.1.0"
        description = "Demo refinement model."

        [artifact]
        framework = "keras"
        format = "keras"
        path = "model.keras"

        [io]
        input_shape = [256, 256, 2]
        output_shape = [256, 256, 1]
        """,
        encoding="utf-8",
    )
    config = CoreConfig.load(local_config_path=None, use_env=False)
    config.artifacts.models.builtin_enabled = False
    config.artifacts.models.local_path = tmp_path / "models"

    service = ModelService.from_config(config)
    artifacts = service.list_model_artifacts()

    assert len(artifacts) == 1
    artifact = artifacts[0]
    assert artifact["ref"] == "local:model/roi_refinement/demo_unet"
    assert artifact["name"] == "demo_unet"
    assert artifact["kind"] == "roi_refinement"
    assert artifact["artifact_path"] == str(model_dir / "model.keras")
    assert service.find_model_artifact("roi_refinement/demo_unet")["metadata"]["io"]["input_shape"] == [256, 256, 2]


def test_model_service_discovers_local_plugin_metadata_without_loading_plugin_code(tmp_path):
    plugin_dir = tmp_path / "plugins" / "example_plugin"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "metadata.toml").write_text(
        """
        name = "example_plugin"
        kind = "plugin"
        version = "0.1.0"
        description = "Manifest-only plugin placeholder."

        [plugin]
        entrypoint = "example_plugin:register"
        capabilities = ["export"]
        """,
        encoding="utf-8",
    )
    config = CoreConfig.load(local_config_path=None, use_env=False)
    config.artifacts.plugins.builtin_enabled = False
    config.artifacts.plugins.local_path = tmp_path / "plugins"

    service = ModelService.from_config(config)
    artifacts = service.list_plugin_artifacts()

    assert len(artifacts) == 1
    assert artifacts[0]["ref"] == "local:plugin/plugin/example_plugin"
    assert artifacts[0]["metadata"]["plugin"]["capabilities"] == ["export"]
    assert service.find_plugin_artifact("example_plugin")["description"] == "Manifest-only plugin placeholder."


def test_artifact_registry_ignores_missing_local_paths(tmp_path):
    config = CoreConfig.load(local_config_path=None, use_env=False)
    config.artifacts.models.builtin_enabled = False
    config.artifacts.plugins.builtin_enabled = False
    config.artifacts.models.local_path = tmp_path / "missing-models"
    config.artifacts.plugins.local_path = tmp_path / "missing-plugins"

    registry = ArtifactRegistry(config)

    assert registry.list_models() == []
    assert registry.list_plugins() == []


def test_load_artifact_metadata_reads_toml(tmp_path):
    metadata_path = tmp_path / "metadata.toml"
    metadata_path.write_text(
        """
        name = "small_model"
        kind = "classifier"
        """,
        encoding="utf-8",
    )

    metadata = load_artifact_metadata(metadata_path)

    assert metadata == {"name": "small_model", "kind": "classifier"}


def test_packaged_asset_roots_exist():
    config = CoreConfig.load(local_config_path=None, use_env=False)
    registry = ArtifactRegistry(config)

    assert Path(str(registry.builtin_model_root)).name == "models"
    assert Path(str(registry.builtin_plugin_root)).name == "plugins"
