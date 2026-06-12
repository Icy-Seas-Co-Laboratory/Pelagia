from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .detection_refinement import IdentityRoiRefinementModel, RoiRefinementModel, RoiRefinementOptions
from ..services.models import ArtifactManifest, ModelService


SUPPORTED_ORACLE_ARTIFACTS = {"auto", "keras", "savedmodel"}


class OracleUnetRefinerError(RuntimeError):
    """Raised when an oracle-builder U-Net artifact cannot be used."""


@dataclass(frozen=True, slots=True)
class OracleBuilderUnetRun:
    """Validated metadata for an oracle-builder segmentation U-Net run."""

    run_dir: Path
    config: dict[str, Any]
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]
    final_keras_path: Path
    savedmodel_path: Path

    @property
    def tile_size(self) -> int:
        height, width, _ = self.input_shape
        if height != width:
            raise OracleUnetRefinerError(
                f"Pelagia ROI refinement expects square tiles; oracle-builder input_shape is {self.input_shape}."
            )
        return height


class _KerasPredictor:
    def __init__(self, model: Any):
        self.model = model

    def predict(self, batch: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(batch, verbose=0))


class _SavedModelPredictor:
    def __init__(self, saved_model: Any, tensorflow: Any):
        self.saved_model = saved_model
        self.tensorflow = tensorflow
        self.signature = saved_model.signatures.get("serving_default")
        if self.signature is None:
            raise OracleUnetRefinerError("SavedModel artifact does not include a serving_default signature.")

    def predict(self, batch: np.ndarray) -> np.ndarray:
        outputs = self.signature(self.tensorflow.convert_to_tensor(batch, dtype=self.tensorflow.float32))
        if not outputs:
            raise OracleUnetRefinerError("SavedModel prediction returned no outputs.")
        return np.asarray(next(iter(outputs.values())).numpy())


class OracleBuilderUnetRefinementModel:
    """
    Pelagia ROI refinement adapter for oracle-builder segmentation U-Net runs.

    The model accepts Pelagia's refinement batch shape KxHxWx2, normalizes it to
    oracle-builder's training convention, and returns a KxHxW probability batch.
    Binary thresholding remains owned by ``detection_refinement`` so all models
    share the same merge/threshold behavior.
    """

    def __init__(
        self,
        run_dir: str | Path,
        *,
        artifact: str = "auto",
        predictor: Any | None = None,
        run: OracleBuilderUnetRun | None = None,
    ):
        artifact = artifact.lower()
        if artifact not in SUPPORTED_ORACLE_ARTIFACTS:
            raise ValueError(
                f"artifact must be one of: {', '.join(sorted(SUPPORTED_ORACLE_ARTIFACTS))}."
            )
        self.run = run or load_oracle_builder_unet_run(run_dir)
        self.artifact = artifact
        self._predictor = predictor

    @property
    def method_name(self) -> str:
        return f"oracle_builder_unet:{self.run.run_dir.name}:{self.artifact}"

    def refinement_options(self, **overrides: Any) -> RoiRefinementOptions:
        """Return Pelagia tile options resolved from the oracle-builder run shape."""
        values = {"tile_size": self.run.tile_size}
        values.update({key: value for key, value in overrides.items() if value is not None})
        return RoiRefinementOptions(**values)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        expected_h, expected_w, expected_channels = self.run.input_shape
        array = normalize_oracle_unet_batch(batch)
        if array.shape[1:] != (expected_h, expected_w, expected_channels):
            raise ValueError(
                "Refinement batch shape does not match oracle-builder input_shape: "
                f"got {array.shape[1:]}, expected {(expected_h, expected_w, expected_channels)}."
            )
        prediction = self._predictor_or_load().predict(array)
        return oracle_unet_prediction_to_probability_mask(
            prediction,
            expected_batch_size=array.shape[0],
            output_shape=self.run.output_shape,
        )

    def _predictor_or_load(self) -> Any:
        if self._predictor is None:
            self._predictor = load_oracle_builder_unet_predictor(self.run, artifact=self.artifact)
        return self._predictor


@dataclass(frozen=True, slots=True)
class KerasArtifactRun:
    """Validated metadata for a model artifact manifest with Keras-like IO."""

    ref: str
    name: str
    artifact_path: Path
    metadata: dict[str, Any]
    input_shape: tuple[int, int, int]
    output_shape: tuple[int, int, int]

    @property
    def tile_size(self) -> int:
        height, width, _ = self.input_shape
        if height != width:
            raise OracleUnetRefinerError(
                f"Pelagia ROI refinement expects square tiles; model input_shape is {self.input_shape}."
            )
        return height


class KerasArtifactRoiRefinementModel:
    """ROI refinement adapter for packaged/local Keras model artifacts."""

    def __init__(
        self,
        manifest: ArtifactManifest | dict[str, Any],
        *,
        predictor: Any | None = None,
    ):
        self.run = load_keras_artifact_run(manifest)
        self._predictor = predictor

    @property
    def method_name(self) -> str:
        return f"keras_artifact:{self.run.ref}"

    def refinement_options(self, **overrides: Any) -> RoiRefinementOptions:
        values = {"tile_size": self.run.tile_size}
        values.update({key: value for key, value in overrides.items() if value is not None})
        return RoiRefinementOptions(**values)

    def predict(self, batch: np.ndarray) -> np.ndarray:
        expected_h, expected_w, expected_channels = self.run.input_shape
        array = normalize_oracle_unet_batch(batch)
        if array.shape[1:] != (expected_h, expected_w, expected_channels):
            raise ValueError(
                "Refinement batch shape does not match model input_shape: "
                f"got {array.shape[1:]}, expected {(expected_h, expected_w, expected_channels)}."
            )
        prediction = self._predictor_or_load().predict(array)
        return oracle_unet_prediction_to_probability_mask(
            prediction,
            expected_batch_size=array.shape[0],
            output_shape=self.run.output_shape,
        )

    def _predictor_or_load(self) -> Any:
        if self._predictor is None:
            self._predictor = _KerasPredictor(_load_keras_model(self.run.artifact_path))
        return self._predictor


def refinement_options_from_config(config_or_section: Any) -> RoiRefinementOptions:
    """Resolve ``RoiRefinementOptions`` from CoreConfig or its roi_refinement section."""
    section = _roi_refinement_section(config_or_section)
    return RoiRefinementOptions(
        tile_size=int(section.tile_size),
        overlap_fraction=float(section.overlap_fraction),
        max_iterations=int(section.max_iterations),
        expansion_pixels=section.expansion_pixels,
        edge_touch_margin=int(section.edge_touch_margin),
        output_threshold=float(section.output_threshold),
        batch_size=section.batch_size,
        encoding=section.encoding,
        overlap_reconciliation_enabled=section.overlap_reconciliation_enabled,
        overlap_iou_threshold=float(section.overlap_iou_threshold),
        overlap_containment_threshold=float(section.overlap_containment_threshold),
        residual_discovery_enabled=section.residual_discovery_enabled,
        residual_max_iterations=int(section.residual_max_iterations),
        residual_roi_assembly_method=section.residual_roi_assembly_method,
        residual_roi_assembly_connectivity=int(section.residual_roi_assembly_connectivity),
        residual_min_area=section.residual_min_area,
        residual_min_width=section.residual_min_width,
        residual_min_height=section.residual_min_height,
        residual_min_width_plus_height=section.residual_min_width_plus_height,
        residual_padding=section.residual_padding,
    )


def refinement_model_from_config(config_or_section: Any) -> RoiRefinementModel:
    """Return the configured ROI refinement model without leaking artifact details."""
    if hasattr(config_or_section, "processing"):
        return resolve_refinement_model(config_or_section)
    section = _roi_refinement_section(config_or_section)
    return resolve_refinement_model(
        model_kind=section.model_kind,
        model_ref=section.model_ref,
        model_run_dir=section.model_run_dir,
        model_artifact=section.model_artifact,
    )


def resolve_refinement_model(
    config: Any | None = None,
    *,
    model_kind: str | None = None,
    model_ref: str | None = None,
    model_run_dir: str | Path | None = None,
    model_artifact: str = "auto",
) -> RoiRefinementModel:
    """Resolve a configured/requested ROI refinement model."""
    resolved_kind = str(model_kind or "").lower()
    explicit_request = any(value is not None for value in (model_kind, model_ref, model_run_dir))
    if not resolved_kind and not model_ref and not model_run_dir and config is not None:
        section = _roi_refinement_section(config)
        if not getattr(section, "enabled", False):
            return IdentityRoiRefinementModel()
        resolved_kind = str(section.model_kind).lower()
        model_ref = section.model_ref
        model_run_dir = section.model_run_dir
        model_artifact = section.model_artifact
    elif config is not None and not explicit_request:
        section = _roi_refinement_section(config)
        if not getattr(section, "enabled", False):
            return IdentityRoiRefinementModel()

    if resolved_kind == "identity" or (not resolved_kind and not model_ref and not model_run_dir):
        return IdentityRoiRefinementModel()

    if model_ref:
        if config is None:
            raise ValueError("CoreConfig is required to resolve ROI refinement model_ref.")
        manifest = ModelService.from_config(config).find_model_artifact(model_ref)
        if manifest is None:
            raise ValueError(f"ROI refinement model_ref was not found: {model_ref!r}.")
        return KerasArtifactRoiRefinementModel(manifest)

    if resolved_kind not in {"oracle_builder_unet", "keras_artifact"}:
        raise ValueError("ROI refinement model_kind must be one of: identity, keras_artifact, oracle_builder_unet.")
    if resolved_kind == "keras_artifact":
        raise ValueError("ROI refinement model_ref is required for keras_artifact.")
    if not model_run_dir:
        raise ValueError("ROI refinement model_run_dir is required for oracle_builder_unet.")
    return OracleBuilderUnetRefinementModel(
        model_run_dir,
        artifact=model_artifact,
    )


def load_oracle_builder_unet_run(run_dir: str | Path) -> OracleBuilderUnetRun:
    """Load and validate oracle-builder run metadata for Pelagia refinement."""
    resolved_run_dir = Path(run_dir).expanduser().resolve()
    config_path = resolved_run_dir / "resolved_config.json"
    if not config_path.exists():
        raise OracleUnetRefinerError(f"Missing oracle-builder resolved config: {config_path}.")
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    run_config = config.get("run", {})
    if run_config.get("task") != "segmentation" or run_config.get("model") != "unet":
        raise OracleUnetRefinerError(
            "Oracle-builder run must have run.task='segmentation' and run.model='unet'."
        )

    input_shape = _shape_tuple(config.get("data", {}).get("input_shape"), "data.input_shape")
    output_shape = _shape_tuple(config.get("data", {}).get("output_shape"), "data.output_shape")
    if input_shape[-1] != 2:
        raise OracleUnetRefinerError(
            f"Oracle-builder U-Net input must have two channels; got input_shape={input_shape}."
        )
    if output_shape[-1] != 1:
        raise OracleUnetRefinerError(
            f"Oracle-builder U-Net output must have one channel; got output_shape={output_shape}."
        )
    if input_shape[:2] != output_shape[:2]:
        raise OracleUnetRefinerError(
            f"Oracle-builder input/output spatial shapes must match; got {input_shape} and {output_shape}."
        )

    model_dir = resolved_run_dir / "model"
    return OracleBuilderUnetRun(
        run_dir=resolved_run_dir,
        config=config,
        input_shape=input_shape,
        output_shape=output_shape,
        final_keras_path=model_dir / "final.keras",
        savedmodel_path=model_dir / "export_savedmodel",
    )


def load_keras_artifact_run(manifest: ArtifactManifest | dict[str, Any]) -> KerasArtifactRun:
    """Validate a discovered model manifest for Keras-style ROI refinement."""
    if isinstance(manifest, ArtifactManifest):
        metadata = manifest.metadata
        ref = manifest.ref
        name = manifest.name
        root_path = Path(manifest.root_path)
        artifact_path_value = manifest.artifact_path()
    else:
        metadata = dict(manifest.get("metadata") or {})
        ref = str(manifest.get("ref") or metadata.get("name") or "model")
        name = str(manifest.get("name") or metadata.get("name") or ref)
        root_path = Path(str(manifest.get("root_path") or "."))
        artifact_path_value = manifest.get("artifact_path")
    artifact = metadata.get("artifact")
    if not isinstance(artifact, dict):
        raise OracleUnetRefinerError(f"Model artifact {ref!r} must include an [artifact] table.")
    framework = str(artifact.get("framework") or artifact.get("format") or "").lower()
    if framework not in {"keras", "tensorflow", "tf"}:
        raise OracleUnetRefinerError(f"Model artifact {ref!r} is not a supported Keras/TensorFlow model.")
    if artifact_path_value:
        artifact_path = Path(str(artifact_path_value)).expanduser()
    else:
        payload_path = artifact.get("path")
        if not payload_path:
            raise OracleUnetRefinerError(f"Model artifact {ref!r} must include artifact.path.")
        artifact_path = root_path / str(payload_path)
    io_metadata = metadata.get("io")
    if not isinstance(io_metadata, dict):
        raise OracleUnetRefinerError(f"Model artifact {ref!r} must include an [io] table.")
    input_shape = _shape_tuple(io_metadata.get("input_shape"), "io.input_shape")
    output_shape = _shape_tuple(io_metadata.get("output_shape"), "io.output_shape")
    if input_shape[-1] != 2:
        raise OracleUnetRefinerError(f"Model artifact {ref!r} input_shape must have two channels.")
    if output_shape[-1] != 1:
        raise OracleUnetRefinerError(f"Model artifact {ref!r} output_shape must have one channel.")
    if input_shape[:2] != output_shape[:2]:
        raise OracleUnetRefinerError(
            f"Model artifact {ref!r} input/output spatial shapes must match."
        )
    return KerasArtifactRun(
        ref=ref,
        name=name,
        artifact_path=artifact_path,
        metadata=metadata,
        input_shape=input_shape,
        output_shape=output_shape,
    )


def load_oracle_builder_unet_predictor(
    run: OracleBuilderUnetRun | str | Path,
    *,
    artifact: str = "auto",
) -> Any:
    """Load the configured oracle-builder model artifact and return a predictor."""
    resolved_run = run if isinstance(run, OracleBuilderUnetRun) else load_oracle_builder_unet_run(run)
    artifact = artifact.lower()
    if artifact not in SUPPORTED_ORACLE_ARTIFACTS:
        raise ValueError(
            f"artifact must be one of: {', '.join(sorted(SUPPORTED_ORACLE_ARTIFACTS))}."
        )
    errors: list[str] = []
    if artifact in {"auto", "keras"}:
        if resolved_run.final_keras_path.exists():
            try:
                return _KerasPredictor(_load_keras_model(resolved_run.final_keras_path))
            except Exception as exc:  # pragma: no cover - depends on optional ML stack
                errors.append(f"final.keras: {exc}")
                if artifact == "keras":
                    raise OracleUnetRefinerError(
                        f"Failed to load Keras model at {resolved_run.final_keras_path}: {exc}"
                    ) from exc
        elif artifact == "keras":
            raise OracleUnetRefinerError(f"Missing Keras artifact: {resolved_run.final_keras_path}.")
    if artifact in {"auto", "savedmodel"}:
        if resolved_run.savedmodel_path.exists():
            try:
                tensorflow = _import_tensorflow()
                return _SavedModelPredictor(tensorflow.saved_model.load(str(resolved_run.savedmodel_path)), tensorflow)
            except Exception as exc:  # pragma: no cover - depends on optional ML stack
                errors.append(f"SavedModel: {exc}")
                if artifact == "savedmodel":
                    raise OracleUnetRefinerError(
                        f"Failed to load SavedModel at {resolved_run.savedmodel_path}: {exc}"
                    ) from exc
        elif artifact == "savedmodel":
            raise OracleUnetRefinerError(f"Missing SavedModel artifact: {resolved_run.savedmodel_path}.")
    detail = "; ".join(errors) if errors else "no supported model artifacts were found"
    raise OracleUnetRefinerError(f"Could not load oracle-builder U-Net model from {resolved_run.run_dir}: {detail}.")


def normalize_oracle_unet_batch(batch: np.ndarray) -> np.ndarray:
    """Normalize KxHxWx2 Pelagia tiles to oracle-builder's training convention."""
    array = np.asarray(batch)
    if array.ndim != 4 or array.shape[-1] != 2:
        raise ValueError("Oracle-builder U-Net input batch must have shape KxHxWx2.")
    normalized = np.zeros(array.shape, dtype=np.float32)
    image = array[..., 0].astype(np.float32, copy=False)
    if image.size and np.nanmax(image) > 1.0:
        image = image / 255.0
    normalized[..., 0] = np.nan_to_num(image, nan=0.0, posinf=1.0, neginf=0.0)
    normalized[..., 1] = (array[..., 1] > 0).astype(np.float32)
    return np.ascontiguousarray(normalized)


def oracle_unet_prediction_to_probability_mask(
    prediction: np.ndarray,
    *,
    expected_batch_size: int,
    output_shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a KxHxW float32 probability mask from common model output shapes."""
    array = np.asarray(prediction)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Oracle-builder U-Net prediction must have shape KxHxW or KxHxWx1; got {array.shape}.")
    expected_h, expected_w, _ = output_shape
    if array.shape != (expected_batch_size, expected_h, expected_w):
        raise ValueError(
            "Oracle-builder U-Net prediction shape mismatch: "
            f"got {array.shape}, expected {(expected_batch_size, expected_h, expected_w)}."
        )
    return np.ascontiguousarray(np.clip(array.astype(np.float32, copy=False), 0.0, 1.0))


def _shape_tuple(value: Any, key: str) -> tuple[int, int, int]:
    if not isinstance(value, list | tuple) or len(value) != 3:
        raise OracleUnetRefinerError(f"{key} must be a three-item shape.")
    return (int(value[0]), int(value[1]), int(value[2]))


def _roi_refinement_section(config_or_section: Any) -> Any:
    if hasattr(config_or_section, "processing"):
        return config_or_section.processing.roi_refinement
    return config_or_section


def _load_keras_model(path: Path) -> Any:
    try:
        import keras

        return keras.models.load_model(path)
    except ModuleNotFoundError:
        tensorflow = _import_tensorflow()
        return tensorflow.keras.models.load_model(path)


def _import_tensorflow() -> Any:
    try:
        import tensorflow

        return tensorflow
    except ModuleNotFoundError as exc:
        raise OracleUnetRefinerError(
            "TensorFlow is required to load oracle-builder U-Net artifacts. "
            "Install Pelagia with the optional ML dependencies, for example: pip install -e '.[ml]'."
        ) from exc
