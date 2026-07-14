from __future__ import annotations

from dataclasses import dataclass, field
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from ..config import CoreConfig
from ..domain import ModelRecord
from ..storage.postgres import PostgresRepository

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ArtifactSource = Literal["builtin", "local"]
ArtifactCategory = Literal["model", "plugin"]


@dataclass(frozen=True, slots=True)
class ArtifactManifest:
    """Metadata loaded from an artifact metadata.toml file."""

    category: ArtifactCategory
    source: ArtifactSource
    name: str
    kind: str
    version: str | None
    metadata: dict[str, Any]
    metadata_path: str
    root_path: str
    relative_path: str
    description: str | None = None

    @property
    def ref(self) -> str:
        """Stable user-facing reference for config/API selection."""
        return f"{self.source}:{self.category}/{self.kind}/{self.name}"

    def artifact_path(self) -> str | None:
        """Return the primary payload path from [artifact].path when present."""
        artifact = self.metadata.get("artifact")
        if not isinstance(artifact, dict):
            return None
        path = artifact.get("path")
        if not path:
            return None
        return str(Path(self.root_path) / str(path))

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref": self.ref,
            "category": self.category,
            "source": self.source,
            "name": self.name,
            "kind": self.kind,
            "version": self.version,
            "description": self.description,
            "metadata_path": self.metadata_path,
            "root_path": self.root_path,
            "relative_path": self.relative_path,
            "artifact_path": self.artifact_path(),
            "metadata": self.metadata,
        }


@dataclass(slots=True)
class ArtifactRegistry:
    """Filesystem/package discovery for Pelagia artifact manifests."""

    config: CoreConfig
    builtin_model_root: Any = field(default_factory=lambda: files("Pelagia").joinpath("assets", "models"))
    builtin_plugin_root: Any = field(default_factory=lambda: files("Pelagia").joinpath("assets", "plugins"))

    def list_models(self) -> list[ArtifactManifest]:
        manifests: list[ArtifactManifest] = []
        model_config = self.config.artifacts.models
        if model_config.builtin_enabled:
            manifests.extend(
                _discover_packaged_manifests(
                    self.builtin_model_root,
                    category="model",
                    source="builtin",
                    metadata_filename=model_config.metadata_filename,
                )
            )
        manifests.extend(
            _discover_local_manifests(
                model_config.local_path,
                category="model",
                source="local",
                metadata_filename=model_config.metadata_filename,
            )
        )
        return sorted(manifests, key=lambda item: (item.kind, item.name, item.source))

    def list_plugins(self) -> list[ArtifactManifest]:
        manifests: list[ArtifactManifest] = []
        plugin_config = self.config.artifacts.plugins
        if plugin_config.builtin_enabled:
            manifests.extend(
                _discover_packaged_manifests(
                    self.builtin_plugin_root,
                    category="plugin",
                    source="builtin",
                    metadata_filename=plugin_config.metadata_filename,
                )
            )
        manifests.extend(
            _discover_local_manifests(
                plugin_config.local_path,
                category="plugin",
                source="local",
                metadata_filename=plugin_config.metadata_filename,
            )
        )
        return sorted(manifests, key=lambda item: (item.kind, item.name, item.source))

    def find_model(self, ref_or_name: str) -> ArtifactManifest | None:
        return _find_manifest(self.list_models(), ref_or_name)

    def find_plugin(self, ref_or_name: str) -> ArtifactManifest | None:
        return _find_manifest(self.list_plugins(), ref_or_name)


class ModelService:
    """Coordinates model metadata registration and artifact discovery."""

    def __init__(
        self,
        repository: PostgresRepository | None = None,
        *,
        config: CoreConfig | None = None,
        artifact_registry: ArtifactRegistry | None = None,
    ):
        self.repository = repository
        self.catalog = None if repository is None else getattr(repository, "catalog", repository)
        self.config = config or CoreConfig.load()
        self.artifact_registry = artifact_registry or ArtifactRegistry(self.config)

    @classmethod
    def from_config(
        cls,
        config: CoreConfig | None = None,
        repository: PostgresRepository | None = None,
    ) -> "ModelService":
        """Create a model service for DB registration and artifact discovery."""
        return cls(repository=repository, config=config or CoreConfig.load())

    def register_model(self, model: ModelRecord, *, project_id: str) -> dict:
        """Register or update model metadata."""
        if self.repository is None:
            raise RuntimeError("A PostgresRepository is required to register model metadata.")
        return self.catalog.register_model(model, project_id=project_id)

    def list_model_artifacts(self) -> list[dict[str, Any]]:
        """List packaged and local model manifests."""
        return [manifest.to_dict() for manifest in self.artifact_registry.list_models()]

    def list_plugin_artifacts(self) -> list[dict[str, Any]]:
        """List packaged and local plugin manifests without loading plugin code."""
        return [manifest.to_dict() for manifest in self.artifact_registry.list_plugins()]

    def find_model_artifact(self, ref_or_name: str) -> dict[str, Any] | None:
        manifest = self.artifact_registry.find_model(ref_or_name)
        return None if manifest is None else manifest.to_dict()

    def find_plugin_artifact(self, ref_or_name: str) -> dict[str, Any] | None:
        manifest = self.artifact_registry.find_plugin(ref_or_name)
        return None if manifest is None else manifest.to_dict()


def load_artifact_metadata(metadata_path: str | Path) -> dict[str, Any]:
    """Load an artifact metadata.toml file from the filesystem."""
    path = Path(metadata_path)
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _discover_local_manifests(
    root: Path,
    *,
    category: ArtifactCategory,
    source: ArtifactSource,
    metadata_filename: str,
) -> list[ArtifactManifest]:
    root_path = Path(root).expanduser()
    if not root_path.exists():
        return []
    manifests = []
    for metadata_path in sorted(root_path.rglob(metadata_filename)):
        if metadata_path.is_file() and not _has_hidden_part(metadata_path.relative_to(root_path)):
            metadata = load_artifact_metadata(metadata_path)
            manifests.append(
                _manifest_from_metadata(
                    metadata,
                    category=category,
                    source=source,
                    metadata_path=str(metadata_path),
                    root_path=str(metadata_path.parent),
                    relative_path=metadata_path.parent.relative_to(root_path).as_posix(),
                )
            )
    return manifests


def _discover_packaged_manifests(
    root: Any,
    *,
    category: ArtifactCategory,
    source: ArtifactSource,
    metadata_filename: str,
) -> list[ArtifactManifest]:
    if not root.is_dir():
        return []
    manifests = []
    for metadata_resource in _walk_packaged_metadata(root, metadata_filename):
        metadata = tomllib.loads(metadata_resource.read_text(encoding="utf-8"))
        root_path = str(metadata_resource.parent)
        manifests.append(
            _manifest_from_metadata(
                metadata,
                category=category,
                source=source,
                metadata_path=str(metadata_resource),
                root_path=root_path,
                relative_path=_packaged_relative_path(root, metadata_resource.parent),
            )
        )
    return manifests


def _walk_packaged_metadata(root: Any, metadata_filename: str) -> list[Any]:
    matches = []
    for child in root.iterdir():
        if child.name.startswith("."):
            continue
        if child.is_dir():
            matches.extend(_walk_packaged_metadata(child, metadata_filename))
        elif child.name == metadata_filename:
            matches.append(child)
    return matches


def _has_hidden_part(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)


def _manifest_from_metadata(
    metadata: dict[str, Any],
    *,
    category: ArtifactCategory,
    source: ArtifactSource,
    metadata_path: str,
    root_path: str,
    relative_path: str,
) -> ArtifactManifest:
    name = metadata.get("name")
    kind = metadata.get("kind")
    if not name or not kind:
        raise ValueError(f"Artifact metadata must include name and kind: {metadata_path}")
    version = metadata.get("version")
    description = metadata.get("description")
    return ArtifactManifest(
        category=category,
        source=source,
        name=str(name),
        kind=str(kind),
        version=None if version is None else str(version),
        description=None if description is None else str(description),
        metadata=metadata,
        metadata_path=metadata_path,
        root_path=root_path,
        relative_path=relative_path,
    )


def _packaged_relative_path(root: Any, child: Any) -> str:
    try:
        return Path(str(child)).relative_to(Path(str(root))).as_posix()
    except ValueError:
        pass
    return child.name


def _find_manifest(manifests: list[ArtifactManifest], ref_or_name: str) -> ArtifactManifest | None:
    for manifest in manifests:
        if ref_or_name in {manifest.ref, manifest.name, f"{manifest.kind}/{manifest.name}"}:
            return manifest
    return None
