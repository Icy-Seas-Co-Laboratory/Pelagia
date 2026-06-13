from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from ..config import CoreConfig, IMAGE_DATA_STORAGE_ENCODINGS, ProcessingConfig
from ..domain import AssetKind, JobStatus, PipelineStage
from .segmentation_options import (
    FRAME_PAYLOAD_KINDS,
    ROI_ENCODINGS,
    segmentation_capabilities,
)
from ..services.models import ModelService


def system_capabilities(config: CoreConfig) -> dict[str, Any]:
    """Return a GUI-facing capability map for the Pelagia backend."""
    processing = config.processing
    return {
        "name": "Pelagia",
        "version": "0.0.1",
        "api": {
            "endpoints": {
                "system": "/system",
                "system_config": "/system/config",
                "system_capabilities": "/system/capabilities",
                "preprocessing_options": "/preprocessing/options",
                "segmentation_options": "/segmentation/options",
                "roi_refinement_options": "/roi-refinement/options",
                "roi_refinement": "/roi-refinement",
                "queue_roi_refinement": "/roi-refinement/jobs",
                "live_segmentation": "/live/segmentation",
                "frame_original": "/frame/original",
                "frame_preprocessed": "/frame/preprocessed",
                "frame_context": "/frames/{frame_id}/context",
                "generate_background": "/frame/background",
                "queue_background": "/frame/background/jobs",
                "queue_preprocessing": "/frame/preprocess/jobs",
                "queue_segmentation": "/segmentation/jobs",
                "jobs": "/jobs",
                "jobs_summary": "/jobs/summary",
            },
        },
        "supported": {
            "asset_kinds": [kind.value for kind in AssetKind],
            "pipeline_stages": [stage.value for stage in PipelineStage],
            "job_statuses": [status.value for status in JobStatus],
            "image_encodings": sorted(IMAGE_DATA_STORAGE_ENCODINGS),
            "frame_payload_kinds": FRAME_PAYLOAD_KINDS,
            "roi_encoding_options": ROI_ENCODINGS,
        },
        "processing": {
            "groups": [
                "video_ingest",
                "preprocessing",
                "flatfield",
                "background",
                "frame_storage",
                "thumbhash",
                "thresholding",
                "mask_augmentation",
                "roi_assembly",
                "roi_filter",
                "roi_recording",
                "roi_refinement",
            ],
            "preprocessing": preprocessing_capabilities(processing),
            "segmentation": segmentation_capabilities(processing),
            "roi_refinement": roi_refinement_capabilities(config),
        },
        "jobs": {
            "stages": [
                {
                    "id": PipelineStage.EXTRACT_FRAMES.value,
                    "label": "Ingestion",
                    "unit": "frames",
                    "queue_endpoint": "/ingestion/videos",
                },
                {
                    "id": PipelineStage.PREPROCESS_FRAMES.value,
                    "label": "Preprocessing",
                    "unit": "frames",
                    "queue_endpoint": "/frame/preprocess/jobs",
                },
                {
                    "id": PipelineStage.SEGMENT.value,
                    "label": "Candidate ROIs",
                    "unit": "frames",
                    "secondary_unit": "detections",
                    "queue_endpoint": "/segmentation/jobs",
                },
                {
                    "id": PipelineStage.ROI_REFINEMENT.value,
                    "label": "ROI Refinement",
                    "unit": "rois",
                    "queue_endpoint": "/roi-refinement/jobs",
                },
            ],
            "queueable_stages": [
                PipelineStage.EXTRACT_FRAMES.value,
                PipelineStage.BACKGROUND_FRAMES.value,
                PipelineStage.PREPROCESS_FRAMES.value,
                PipelineStage.SEGMENT.value,
                PipelineStage.ROI_REFINEMENT.value,
            ],
            "worker_capabilities": [stage.value for stage in PipelineStage],
        },
        "storage": {
            "frame_storage": _dataclass_dict(processing.frame_storage),
            "image_data_storage": _dataclass_dict(config.image_data_storage),
            "kvstore": {
                "hash_algorithm_options": ["sha256", "blake3"],
                "configured_hash_algorithm": config.kvstore.hash_algorithm,
            },
        },
    }


def roi_refinement_capabilities(config: CoreConfig) -> dict[str, Any]:
    """Return GUI-facing ROI refinement defaults, valid options, and model refs."""
    models = ModelService.from_config(config).list_model_artifacts()
    roi_models = [model for model in models if model.get("kind") == "roi_refinement"]
    defaults = config.processing.roi_refinement
    return {
        "pipeline_stage_order": [
            "source",
            "model_selection",
            "tiling",
            "prediction",
            "expansion",
            "residual_discovery",
            "reconciliation",
            "recording",
        ],
        "supported": {
            "model_kinds": ["identity", "keras_artifact", "oracle_builder_unet"],
            "model_refs": [model["ref"] for model in roi_models],
            "model_artifacts": roi_models,
            "roi_encoding_options": ROI_ENCODINGS,
        },
        "defaults": {
            "roi_refinement": _dataclass_dict(defaults),
        },
        "fields": {
            "source": [
                _field("detection_ids", "Detection IDs", "string-list", request_field_name="detection_ids"),
            ],
            "model_selection": [
                _field("model_kind", "Model Kind", "enum", options=["identity", "keras_artifact", "oracle_builder_unet"], config_section="processing.roi_refinement"),
                _field("model_ref", "Model Reference", "enum", options=[model["ref"] for model in roi_models], config_section="processing.roi_refinement"),
                _field("model_run_dir", "Model Run Directory", "nullable-string", config_section="processing.roi_refinement"),
                _field("model_artifact", "Model Artifact", "enum", options=["auto", "keras", "savedmodel"], config_section="processing.roi_refinement"),
            ],
            "tiling": [
                _field("tile_size", "Tile Size", "integer", minimum=1, step=1, config_section="processing.roi_refinement"),
                _field("overlap_fraction", "Overlap Fraction", "number", minimum=0, maximum=0.99, step=0.01, config_section="processing.roi_refinement"),
                _field("batch_size", "Batch Size", "nullable-integer", minimum=1, step=1, config_section="processing.roi_refinement"),
            ],
            "prediction": [
                _field("output_threshold", "Output Threshold", "number", minimum=0, maximum=1, step=0.01, config_section="processing.roi_refinement"),
            ],
            "expansion": [
                _field("allow_frame_expansion", "Allow Frame Expansion", "boolean", default=True),
                _field("max_iterations", "Max Iterations", "integer", minimum=1, step=1, config_section="processing.roi_refinement"),
                _field("expansion_pixels", "Expansion Pixels", "nullable-integer", minimum=1, step=1, config_section="processing.roi_refinement"),
                _field("edge_touch_margin", "Edge Touch Margin", "integer", minimum=1, step=1, config_section="processing.roi_refinement"),
            ],
            "reconciliation": [
                _field("overlap_reconciliation_enabled", "Overlap Reconciliation", "boolean", config_section="processing.roi_refinement"),
                _field("overlap_iou_threshold", "Overlap IoU Threshold", "number", minimum=0, maximum=1, step=0.01, config_section="processing.roi_refinement"),
                _field("overlap_containment_threshold", "Overlap Containment Threshold", "number", minimum=0, maximum=1, step=0.01, config_section="processing.roi_refinement"),
            ],
            "residual_discovery": [
                _field("residual_discovery_enabled", "Residual Discovery", "boolean", config_section="processing.roi_refinement"),
                _field("residual_max_iterations", "Residual Max Iterations", "integer", minimum=1, step=1, config_section="processing.roi_refinement"),
                _field("residual_roi_assembly_method", "Residual Assembly Method", "enum", options=["connected_components", "contours"], config_section="processing.roi_refinement"),
                _field("residual_roi_assembly_connectivity", "Residual Assembly Connectivity", "enum", options=[4, 8], config_section="processing.roi_refinement"),
                _field("residual_min_area", "Residual Min Area", "nullable-number", minimum=0, step=1, config_section="processing.roi_refinement"),
                _field("residual_min_width", "Residual Min Width", "nullable-number", minimum=0, step=1, config_section="processing.roi_refinement"),
                _field("residual_min_height", "Residual Min Height", "nullable-number", minimum=0, step=1, config_section="processing.roi_refinement"),
                _field("residual_min_width_plus_height", "Residual Min Width + Height", "nullable-number", minimum=0, step=1, config_section="processing.roi_refinement"),
                _field("residual_padding", "Residual Padding", "nullable-integer", minimum=0, step=1, config_section="processing.roi_refinement"),
            ],
            "recording": [
                _field("store", "Store Refined Detections", "boolean", default=True),
                _field("encoding", "ROI Encoding", "enum", options=ROI_ENCODINGS, config_section="processing.roi_refinement"),
                _field("dry_run", "Dry Run", "boolean", default=False),
            ],
        },
    }


def preprocessing_capabilities(config: ProcessingConfig) -> dict[str, Any]:
    """Return GUI-facing preprocessing defaults, valid options, and field metadata."""
    return {
        "pipeline_stage_order": [
            "source",
            "crop",
            "mask",
            "flatfield",
            "background_correction",
            "inversion",
            "recording",
        ],
        "supported": {
            "image_encodings": sorted(IMAGE_DATA_STORAGE_ENCODINGS),
            "frame_payload_kinds": FRAME_PAYLOAD_KINDS,
            "flatfield_axes": [0, 1],
            "response_formats": ["metadata", "matrix"],
        },
        "defaults": {
            "preprocessing": _dataclass_dict(config.preprocessing),
            "flatfield": _dataclass_dict(config.flatfield),
            "frame_storage": _dataclass_dict(config.frame_storage),
            "thumbhash": _dataclass_dict(config.thumbhash),
        },
        "fields": {
            "source": [
                _field("frame_id", "Frame ID", "string", request_field_name="frame_id"),
                _field("frame_ids", "Frame IDs", "string-list", request_field_name="frame_ids"),
                _field("asset_id", "Asset ID", "string", request_field_name="asset_id"),
                _field("frame_num", "Frame Number", "nullable-integer", minimum=0),
                _field("start_frame", "Start Frame", "nullable-integer", minimum=0),
                _field("end_frame", "End Frame", "nullable-integer", minimum=0),
                _field("limit", "Limit", "nullable-integer", minimum=1),
            ],
            "crop": [
                _field("crop_enabled", "Crop", "boolean", config_section="processing.preprocessing"),
                _field("crop_x", "Crop X", "nullable-integer", minimum=0, step=1, config_section="processing.preprocessing"),
                _field("crop_y", "Crop Y", "nullable-integer", minimum=0, step=1, config_section="processing.preprocessing"),
                _field("crop_w", "Crop Width", "nullable-integer", minimum=1, step=1, config_section="processing.preprocessing"),
                _field("crop_h", "Crop Height", "nullable-integer", minimum=1, step=1, config_section="processing.preprocessing"),
            ],
            "mask": [
                _field("apply_mask", "Apply Mask", "boolean", config_section="processing.preprocessing"),
                _field("mask_path", "Mask Path", "nullable-string", config_section="processing.preprocessing"),
            ],
            "flatfield": [
                _field("flatfield_correction", "Flatfield Correction", "boolean", config_section="processing.flatfield"),
                _field("flatfield_q", "Flatfield Quantile", "number", minimum=0, maximum=1, step=0.01, config_section="processing.flatfield"),
                _field("flatfield_axis", "Flatfield Axis", "enum", options=[0, 1], config_section="processing.flatfield"),
                _field("flatfield_min_field_value", "Flatfield Min Field Value", "number", minimum=0, step=1, config_section="processing.flatfield"),
                _field("flatfield_max_field_value", "Flatfield Max Field Value", "nullable-number", minimum=0, step=1, config_section="processing.flatfield"),
            ],
            "background_correction": [
                _field("background_correction", "Background Correction", "boolean", config_section="processing.preprocessing"),
                _field("background_min_field_value", "Background Min Field Value", "number", minimum=0, step=1, config_section="processing.preprocessing"),
                _field("background_max_field_value", "Background Max Field Value", "nullable-number", minimum=0, step=1, config_section="processing.preprocessing"),
                _field("adaptive_background_subtraction", "Adaptive Background Subtraction", "boolean", config_section="processing.preprocessing"),
                _field("adaptive_background_period", "Adaptive Background Period", "integer", minimum=1, step=1, config_section="processing.preprocessing"),
            ],
            "inversion": [
                _field("invert_intensity", "Invert Intensity", "boolean", config_section="processing.preprocessing"),
            ],
            "recording": [
                _field("store", "Store Preprocessed Frame", "boolean", default=True),
                _field("encoding", "Storage Encoding", "enum", options=sorted(IMAGE_DATA_STORAGE_ENCODINGS), config_section="processing.frame_storage"),
                _field("response_format", "Response Format", "enum", options=["metadata", "matrix"]),
            ],
        },
    }


def _field(
    key: str,
    label: str,
    field_type: str,
    *,
    options: list[Any] | None = None,
    minimum: int | float | None = None,
    maximum: int | float | None = None,
    step: int | float | None = None,
    default: Any = None,
    config_section: str | None = None,
    request_field_name: str | None = None,
) -> dict[str, Any]:
    field: dict[str, Any] = {
        "key": key,
        "label": label,
        "type": field_type,
        "request_field_name": request_field_name or key,
    }
    if config_section is not None:
        field["config_section"] = config_section
    if options is not None:
        field["options"] = options
    if minimum is not None:
        field["min"] = minimum
    if maximum is not None:
        field["max"] = maximum
    if step is not None:
        field["step"] = step
    if default is not None:
        field["default"] = default
    return field


def _dataclass_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value)
