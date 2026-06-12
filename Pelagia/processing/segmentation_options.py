from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from ..config import ProcessingConfig


PIPELINE_STAGE_ORDER = [
    "source",
    "preprocessing",
    "thresholding",
    "mask_augmentation",
    "roi_assembly",
    "roi_filter",
    "roi_recording",
]

FRAME_PAYLOAD_KINDS = ["original", "raw", "preprocessed", "processed", "corrected"]
THRESHOLD_METHODS = [
    "manual",
    "otsu",
    "bounded_otsu",
    "bounded_otsu_canny",
    "canny",
    "adaptive_mean",
    "adaptive_gaussian",
    "percentile_background",
    "hysteresis",
    "sobel_edges",
    "auto",
]
MASK_AUGMENTATION_STEPS = [
    "none",
    "dilate",
    "erode",
    "open",
    "close",
    "fill_holes",
    "remove_small_components",
    "clear_border",
]
ROI_ASSEMBLY_METHODS = ["connected_components", "contours"]
ROI_ENCODINGS = ["zstd", "png", "raw", "auto"]


def resolve_segmentation_options(
    overrides: dict[str, Any] | None,
    config: ProcessingConfig,
) -> dict[str, dict[str, Any]]:
    """Resolve flat request/payload fields into grouped segmentation options."""
    values = {key: value for key, value in (overrides or {}).items() if value is not None}
    preprocessing = config.preprocessing
    flatfield = config.flatfield
    thresholding = config.thresholding
    mask = config.mask_augmentation
    assembly = config.roi_assembly
    roi_filter = config.roi_filter
    recording = config.roi_recording

    frame_payload_kind = _get(values, "frame_payload_kind", "original")
    apply_preprocessing = _get(
        values,
        "apply_preprocessing",
        str(frame_payload_kind).lower() in {"original", "raw"},
    )
    threshold = values.get("threshold")
    explicit_manual_threshold = threshold is not None and values.get("threshold_method") is None
    threshold_method = _normalize_method(
        _get(values, "threshold_method", "manual" if explicit_manual_threshold else thresholding.method),
        supported=THRESHOLD_METHODS,
        field_name="threshold_method",
    )
    mask_steps_default = [] if explicit_manual_threshold else mask.steps
    mask_steps = _normalize_steps(_get(values, "mask_augmentation_steps", mask_steps_default))

    resolved = {
        "source": {
            "frame_payload_kind": _normalize_method(
                frame_payload_kind,
                supported=FRAME_PAYLOAD_KINDS,
                field_name="frame_payload_kind",
            ),
            "apply_preprocessing": bool(apply_preprocessing),
        },
        "preprocessing": {
            "flatfield_correction": bool(
                _get(values, "flatfield_correction", flatfield.flatfield_correction)
            ),
            "flatfield_q": float(_get(values, "flatfield_q", flatfield.flatfield_q)),
            "flatfield_axis": int(_get(values, "flatfield_axis", flatfield.flatfield_axis)),
            "flatfield_min_field_value": float(
                _get(values, "flatfield_min_field_value", flatfield.flatfield_min_field_value)
            ),
            "flatfield_max_field_value": _get(
                values,
                "flatfield_max_field_value",
                flatfield.flatfield_max_field_value,
            ),
            "apply_mask": bool(_get(values, "apply_mask", preprocessing.apply_mask)),
            "crop_enabled": bool(_get(values, "crop_enabled", preprocessing.crop_enabled)),
            "crop_x": _get(values, "crop_x", preprocessing.crop_x),
            "crop_y": _get(values, "crop_y", preprocessing.crop_y),
            "crop_w": _get(values, "crop_w", preprocessing.crop_w),
            "crop_h": _get(values, "crop_h", preprocessing.crop_h),
            "background_correction": bool(
                _get(values, "background_correction", preprocessing.background_correction)
            ),
            "background_min_field_value": float(
                _get(
                    values,
                    "background_min_field_value",
                    preprocessing.background_min_field_value,
                )
            ),
            "background_max_field_value": _get(
                values,
                "background_max_field_value",
                preprocessing.background_max_field_value,
            ),
            "invert_intensity": bool(
                _get(values, "invert_intensity", preprocessing.invert_intensity)
            ),
        },
        "thresholding": {
            "threshold": threshold,
            "threshold_method": threshold_method,
            "manual_threshold": float(
                _get(values, "manual_threshold", thresholding.manual_threshold)
            ),
            "thresholding_maximum_value": float(
                _get(
                    values,
                    "thresholding_maximum_value",
                    thresholding.thresholding_maximum_value,
                )
            ),
            "bounded_otsu_min_contrast": float(
                _get(
                    values,
                    "bounded_otsu_min_contrast",
                    thresholding.bounded_otsu_min_contrast,
                )
            ),
            "bounded_otsu_max_foreground_fraction": float(
                _get(
                    values,
                    "bounded_otsu_max_foreground_fraction",
                    thresholding.bounded_otsu_max_foreground_fraction,
                )
            ),
            "canny_enabled": bool(_get(values, "canny_enabled", thresholding.canny_enabled)),
            "canny_low_threshold": float(
                _get(values, "canny_low_threshold", thresholding.canny_low_threshold)
            ),
            "canny_high_threshold": float(
                _get(values, "canny_high_threshold", thresholding.canny_high_threshold)
            ),
            "canny_blur_kernel": int(
                _get(values, "canny_blur_kernel", thresholding.canny_blur_kernel)
            ),
            "adaptive_block_size": int(
                _get(values, "adaptive_block_size", thresholding.adaptive_block_size)
            ),
            "adaptive_c": float(_get(values, "adaptive_c", thresholding.adaptive_c)),
            "percentile_background_percentile": float(
                _get(
                    values,
                    "percentile_background_percentile",
                    thresholding.percentile_background_percentile,
                )
            ),
            "percentile_min_contrast": float(
                _get(values, "percentile_min_contrast", thresholding.percentile_min_contrast)
            ),
            "hysteresis_low_threshold": float(
                _get(values, "hysteresis_low_threshold", thresholding.hysteresis_low_threshold)
            ),
            "hysteresis_high_threshold": float(
                _get(values, "hysteresis_high_threshold", thresholding.hysteresis_high_threshold)
            ),
            "hysteresis_connectivity": int(
                _get(values, "hysteresis_connectivity", thresholding.hysteresis_connectivity)
            ),
            "sobel_percentile": float(
                _get(values, "sobel_percentile", thresholding.sobel_percentile)
            ),
            "sobel_threshold": _get(values, "sobel_threshold", thresholding.sobel_threshold),
            "sobel_kernel_size": int(
                _get(values, "sobel_kernel_size", thresholding.sobel_kernel_size)
            ),
        },
        "mask_augmentation": {
            "mask_augmentation_enabled": bool(
                _get(values, "mask_augmentation_enabled", mask.enabled)
            ),
            "mask_augmentation_steps": mask_steps,
            "dilate_kernel_w": int(_get(values, "dilate_kernel_w", mask.dilate_kernel_w)),
            "dilate_kernel_h": int(_get(values, "dilate_kernel_h", mask.dilate_kernel_h)),
            "dilate_iterations": int(_get(values, "dilate_iterations", mask.dilate_iterations)),
            "erode_kernel_w": int(_get(values, "erode_kernel_w", mask.erode_kernel_w)),
            "erode_kernel_h": int(_get(values, "erode_kernel_h", mask.erode_kernel_h)),
            "erode_iterations": int(_get(values, "erode_iterations", mask.erode_iterations)),
            "open_kernel_w": int(_get(values, "open_kernel_w", mask.open_kernel_w)),
            "open_kernel_h": int(_get(values, "open_kernel_h", mask.open_kernel_h)),
            "open_iterations": int(_get(values, "open_iterations", mask.open_iterations)),
            "close_kernel_w": int(_get(values, "close_kernel_w", mask.close_kernel_w)),
            "close_kernel_h": int(_get(values, "close_kernel_h", mask.close_kernel_h)),
            "close_iterations": int(_get(values, "close_iterations", mask.close_iterations)),
            "fill_holes": bool(_get(values, "fill_holes", mask.fill_holes)),
            "remove_small_components": bool(
                _get(values, "remove_small_components", mask.remove_small_components)
            ),
            "min_component_area": float(_get(values, "min_component_area", mask.min_component_area)),
            "clear_border": bool(_get(values, "clear_border", mask.clear_border)),
        },
        "roi_assembly": {
            "roi_assembly_method": _normalize_method(
                _get(values, "roi_assembly_method", assembly.method),
                supported=ROI_ASSEMBLY_METHODS,
                field_name="roi_assembly_method",
            ),
            "roi_assembly_connectivity": int(
                _get(values, "roi_assembly_connectivity", assembly.connectivity)
            ),
        },
        "roi_filter": {
            "min_area": _get(values, "min_area", roi_filter.min_area),
            "max_area": _get(values, "max_area", roi_filter.max_area),
            "min_perimeter": _get(values, "min_perimeter", roi_filter.min_perimeter),
            "max_perimeter": _get(values, "max_perimeter", roi_filter.max_perimeter),
            "min_width": _get(values, "min_width", roi_filter.min_width),
            "max_width": _get(values, "max_width", roi_filter.max_width),
            "min_height": _get(values, "min_height", roi_filter.min_height),
            "max_height": _get(values, "max_height", roi_filter.max_height),
            "min_width_plus_height": _get(
                values,
                "min_width_plus_height",
                roi_filter.min_width_plus_height,
            ),
            "max_width_plus_height": _get(
                values,
                "max_width_plus_height",
                roi_filter.max_width_plus_height,
            ),
        },
        "roi_recording": {
            "padding": int(_get(values, "padding", recording.padding)),
            "roi_encoding": _normalize_method(
                _get(values, "roi_encoding", recording.roi_encoding),
                supported=ROI_ENCODINGS,
                field_name="roi_encoding",
            ),
            "zstd_min_bytes": int(_get(values, "zstd_min_bytes", recording.zstd_min_bytes)),
            "store_roi_payload_min_area": _get(
                values,
                "store_roi_payload_min_area",
                recording.store_roi_payload_min_area,
            ),
            "store_roi_payload_min_width": _get(
                values,
                "store_roi_payload_min_width",
                recording.store_roi_payload_min_width,
            ),
            "store_roi_payload_min_height": _get(
                values,
                "store_roi_payload_min_height",
                recording.store_roi_payload_min_height,
            ),
            "store_roi_payload_min_width_plus_height": _get(
                values,
                "store_roi_payload_min_width_plus_height",
                recording.store_roi_payload_min_width_plus_height,
            ),
            "always_store_mask": bool(_get(values, "always_store_mask", recording.always_store_mask)),
        },
    }
    _validate_positive_ints(resolved)
    return resolved


def flatten_segmentation_options(resolved: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Flatten grouped options into segment_frame keyword arguments."""
    flat: dict[str, Any] = {}
    for group in PIPELINE_STAGE_ORDER:
        flat.update(resolved.get(group, {}))
    return flat


def segment_frame_kwargs(resolved: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return kwargs accepted directly by segment_frame."""
    kwargs = flatten_segmentation_options(resolved)
    kwargs.pop("frame_payload_kind", None)
    return kwargs


def segmentation_capabilities(config: ProcessingConfig) -> dict[str, Any]:
    """Return GUI-facing segmentation capabilities and effective defaults."""
    defaults = resolve_segmentation_options({}, config)
    return {
        "pipeline_stage_order": PIPELINE_STAGE_ORDER,
        "supported": {
            "frame_payload_kinds": FRAME_PAYLOAD_KINDS,
            "threshold_methods": THRESHOLD_METHODS,
            "mask_augmentation_steps": MASK_AUGMENTATION_STEPS,
            "roi_assembly_methods": ROI_ASSEMBLY_METHODS,
            "roi_encoding_options": ROI_ENCODINGS,
        },
        "defaults": defaults,
        "fields": _field_groups(),
        "config_defaults": {
            "preprocessing": _dataclass_dict(config.preprocessing),
            "flatfield": _dataclass_dict(config.flatfield),
            "thresholding": _dataclass_dict(config.thresholding),
            "mask_augmentation": _dataclass_dict(config.mask_augmentation),
            "roi_assembly": _dataclass_dict(config.roi_assembly),
            "roi_filter": _dataclass_dict(config.roi_filter),
            "roi_recording": _dataclass_dict(config.roi_recording),
        },
    }


def _get(values: dict[str, Any], key: str, default: Any) -> Any:
    return default if key not in values or values[key] is None else values[key]


def _normalize_method(value: Any, *, supported: list[str], field_name: str) -> str:
    normalized = str(value).replace("-", "_").lower()
    if normalized not in supported:
        raise ValueError(
            f"Unsupported {field_name} {value!r}; expected one of: {', '.join(supported)}."
        )
    return normalized


def _normalize_steps(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_steps = [value]
    else:
        raw_steps = list(value)
    steps = [str(step).replace("-", "_").lower() for step in raw_steps]
    unknown = [step for step in steps if step not in MASK_AUGMENTATION_STEPS]
    if unknown:
        raise ValueError(
            "Unsupported mask augmentation step(s) "
            f"{unknown!r}; expected one of: {', '.join(MASK_AUGMENTATION_STEPS)}."
        )
    return [] if steps == ["none"] else steps


def _validate_positive_ints(resolved: dict[str, dict[str, Any]]) -> None:
    for group, keys in {
        "thresholding": ["canny_blur_kernel", "adaptive_block_size", "hysteresis_connectivity", "sobel_kernel_size"],
        "mask_augmentation": [
            "dilate_kernel_w",
            "dilate_kernel_h",
            "dilate_iterations",
            "erode_kernel_w",
            "erode_kernel_h",
            "erode_iterations",
            "open_kernel_w",
            "open_kernel_h",
            "open_iterations",
            "close_kernel_w",
            "close_kernel_h",
            "close_iterations",
        ],
        "roi_assembly": ["roi_assembly_connectivity"],
        "roi_recording": ["zstd_min_bytes"],
    }.items():
        for key in keys:
            value = resolved[group][key]
            if value is not None and int(value) < 1:
                raise ValueError(f"{key} must be >= 1.")


def _field_groups() -> dict[str, list[dict[str, Any]]]:
    return {
        "source": [
            _field("frame_payload_kind", "Frame Payload", "enum", options=FRAME_PAYLOAD_KINDS),
            _field("apply_preprocessing", "Apply Preprocessing", "boolean"),
        ],
        "preprocessing": [
            _field("flatfield_correction", "Flatfield Correction", "boolean"),
            _field("flatfield_q", "Flatfield Quantile", "number", minimum=0, maximum=1, step=0.01),
            _field("flatfield_axis", "Flatfield Axis", "integer", minimum=0, maximum=1, step=1),
            _field("flatfield_min_field_value", "Flatfield Min Field Value", "number", minimum=0, step=1),
            _field("flatfield_max_field_value", "Flatfield Max Field Value", "nullable-number", minimum=0, step=1),
            _field("apply_mask", "Apply Mask", "boolean"),
            _field("crop_enabled", "Crop", "boolean"),
            _field("crop_x", "Crop X", "nullable-number", minimum=0, step=1),
            _field("crop_y", "Crop Y", "nullable-number", minimum=0, step=1),
            _field("crop_w", "Crop Width", "nullable-number", minimum=1, step=1),
            _field("crop_h", "Crop Height", "nullable-number", minimum=1, step=1),
            _field("background_correction", "Background Correction", "boolean"),
            _field("background_min_field_value", "Background Min Field Value", "number", minimum=0, step=1),
            _field("background_max_field_value", "Background Max Field Value", "nullable-number", minimum=0, step=1),
            _field("invert_intensity", "Invert Intensity", "boolean"),
        ],
        "thresholding": [
            _field("threshold_method", "Threshold Method", "enum", options=THRESHOLD_METHODS),
            _field("threshold", "Manual Threshold", "nullable-number", minimum=0, maximum=255, step=1, methods=["manual", "auto"]),
            _field("manual_threshold", "Manual Threshold Default", "number", minimum=0, maximum=255, step=1, methods=["manual"]),
            _field("thresholding_maximum_value", "Maximum Threshold", "number", minimum=0, maximum=255, step=1, methods=["otsu", "bounded_otsu", "bounded_otsu_canny"]),
            _field("bounded_otsu_min_contrast", "Bounded Otsu Min Contrast", "number", minimum=0, step=1, methods=["bounded_otsu", "bounded_otsu_canny"]),
            _field("bounded_otsu_max_foreground_fraction", "Max Foreground Fraction", "number", minimum=0, maximum=1, step=0.01, methods=["bounded_otsu", "bounded_otsu_canny"]),
            _field("canny_enabled", "Use Canny", "boolean", methods=["bounded_otsu_canny"]),
            _field("canny_low_threshold", "Canny Low", "number", minimum=0, step=1, methods=["canny", "bounded_otsu_canny"]),
            _field("canny_high_threshold", "Canny High", "number", minimum=0, step=1, methods=["canny", "bounded_otsu_canny"]),
            _field("canny_blur_kernel", "Canny Blur Kernel", "integer", minimum=1, step=2, methods=["canny", "bounded_otsu_canny"]),
            _field("adaptive_block_size", "Adaptive Block Size", "integer", minimum=3, step=2, methods=["adaptive_mean", "adaptive_gaussian"]),
            _field("adaptive_c", "Adaptive C", "number", step=0.5, methods=["adaptive_mean", "adaptive_gaussian"]),
            _field("percentile_background_percentile", "Background Percentile", "number", minimum=0, maximum=100, step=1, methods=["percentile_background"]),
            _field("percentile_min_contrast", "Percentile Min Contrast", "number", minimum=0, step=1, methods=["percentile_background"]),
            _field("hysteresis_low_threshold", "Hysteresis Low", "number", minimum=0, step=1, methods=["hysteresis"]),
            _field("hysteresis_high_threshold", "Hysteresis High", "number", minimum=0, step=1, methods=["hysteresis"]),
            _field("hysteresis_connectivity", "Hysteresis Connectivity", "integer", options=[4, 8], methods=["hysteresis"]),
            _field("sobel_percentile", "Sobel Percentile", "number", minimum=0, maximum=100, step=1, methods=["sobel_edges"]),
            _field("sobel_threshold", "Sobel Threshold", "nullable-number", minimum=0, step=1, methods=["sobel_edges"]),
            _field("sobel_kernel_size", "Sobel Kernel Size", "integer", minimum=1, step=2, methods=["sobel_edges"]),
        ],
        "mask_augmentation": [
            _field("mask_augmentation_enabled", "Enable Mask Augmentation", "boolean"),
            _field("mask_augmentation_steps", "Mask Steps", "multi-enum", options=MASK_AUGMENTATION_STEPS),
            _field("dilate_kernel_w", "Dilate Kernel Width", "integer", minimum=1, step=1),
            _field("dilate_kernel_h", "Dilate Kernel Height", "integer", minimum=1, step=1),
            _field("dilate_iterations", "Dilate Iterations", "integer", minimum=1, step=1),
            _field("erode_kernel_w", "Erode Kernel Width", "integer", minimum=1, step=1),
            _field("erode_kernel_h", "Erode Kernel Height", "integer", minimum=1, step=1),
            _field("erode_iterations", "Erode Iterations", "integer", minimum=1, step=1),
            _field("open_kernel_w", "Open Kernel Width", "integer", minimum=1, step=1),
            _field("open_kernel_h", "Open Kernel Height", "integer", minimum=1, step=1),
            _field("open_iterations", "Open Iterations", "integer", minimum=1, step=1),
            _field("close_kernel_w", "Close Kernel Width", "integer", minimum=1, step=1),
            _field("close_kernel_h", "Close Kernel Height", "integer", minimum=1, step=1),
            _field("close_iterations", "Close Iterations", "integer", minimum=1, step=1),
            _field("fill_holes", "Fill Holes", "boolean"),
            _field("remove_small_components", "Remove Small Components", "boolean"),
            _field("min_component_area", "Minimum Component Area", "number", minimum=0, step=1),
            _field("clear_border", "Clear Border Components", "boolean"),
        ],
        "roi_assembly": [
            _field("roi_assembly_method", "ROI Assembly Method", "enum", options=ROI_ASSEMBLY_METHODS),
            _field("roi_assembly_connectivity", "ROI Connectivity", "integer", options=[4, 8]),
        ],
        "roi_filter": [
            _field("min_area", "Min Area", "nullable-number", minimum=0, step=1),
            _field("max_area", "Max Area", "nullable-number", minimum=0, step=1),
            _field("min_perimeter", "Min BBox Perimeter", "number", minimum=0, step=1),
            _field("max_perimeter", "Max BBox Perimeter", "nullable-number", minimum=0, step=1),
            _field("min_width", "Min Width", "nullable-number", minimum=0, step=1),
            _field("max_width", "Max Width", "nullable-number", minimum=0, step=1),
            _field("min_height", "Min Height", "nullable-number", minimum=0, step=1),
            _field("max_height", "Max Height", "nullable-number", minimum=0, step=1),
            _field("min_width_plus_height", "Min Width + Height", "nullable-number", minimum=0, step=1),
            _field("max_width_plus_height", "Max Width + Height", "nullable-number", minimum=0, step=1),
        ],
        "roi_recording": [
            _field("padding", "ROI Padding", "integer", minimum=0, step=1),
            _field("roi_encoding", "ROI Encoding", "enum", options=ROI_ENCODINGS),
            _field("zstd_min_bytes", "Zstd Minimum Bytes", "integer", minimum=1, step=1),
            _field("store_roi_payload_min_area", "Store Payload Min Area", "nullable-number", minimum=0, step=1),
            _field("store_roi_payload_min_width", "Store Payload Min Width", "nullable-number", minimum=0, step=1),
            _field("store_roi_payload_min_height", "Store Payload Min Height", "nullable-number", minimum=0, step=1),
            _field("store_roi_payload_min_width_plus_height", "Store Payload Min Width + Height", "nullable-number", minimum=0, step=1),
            _field("always_store_mask", "Always Store Mask", "boolean"),
        ],
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
    methods: list[str] | None = None,
) -> dict[str, Any]:
    group = _group_for_key(key)
    field: dict[str, Any] = {
        "key": key,
        "label": label,
        "type": field_type,
        "config_section": "processing.flatfield" if key.startswith("flatfield_") else f"processing.{group}",
        "request_field_name": key,
    }
    if options is not None:
        field["options"] = options
    if minimum is not None:
        field["min"] = minimum
    if maximum is not None:
        field["max"] = maximum
    if step is not None:
        field["step"] = step
    if methods is not None:
        field["threshold_methods"] = methods
    return field


def _group_for_key(key: str) -> str:
    for group, fields in {
        "source": ["frame_payload_kind", "apply_preprocessing"],
        "preprocessing": [
            "flatfield_correction",
            "flatfield_q",
            "flatfield_axis",
            "flatfield_min_field_value",
            "flatfield_max_field_value",
            "apply_mask",
            "crop_enabled",
            "crop_x",
            "crop_y",
            "crop_w",
            "crop_h",
            "background_correction",
            "background_min_field_value",
            "background_max_field_value",
            "invert_intensity",
        ],
        "thresholding": [
            "threshold",
            "threshold_method",
            "manual_threshold",
            "thresholding_maximum_value",
            "bounded_otsu_min_contrast",
            "bounded_otsu_max_foreground_fraction",
            "canny_enabled",
            "canny_low_threshold",
            "canny_high_threshold",
            "canny_blur_kernel",
            "adaptive_block_size",
            "adaptive_c",
            "percentile_background_percentile",
            "percentile_min_contrast",
            "hysteresis_low_threshold",
            "hysteresis_high_threshold",
            "hysteresis_connectivity",
            "sobel_percentile",
            "sobel_threshold",
            "sobel_kernel_size",
        ],
        "mask_augmentation": [
            "mask_augmentation_enabled",
            "mask_augmentation_steps",
            "dilate_kernel_w",
            "dilate_kernel_h",
            "dilate_iterations",
            "erode_kernel_w",
            "erode_kernel_h",
            "erode_iterations",
            "open_kernel_w",
            "open_kernel_h",
            "open_iterations",
            "close_kernel_w",
            "close_kernel_h",
            "close_iterations",
            "fill_holes",
            "remove_small_components",
            "min_component_area",
            "clear_border",
        ],
        "roi_assembly": ["roi_assembly_method", "roi_assembly_connectivity"],
        "roi_filter": [
            "min_area",
            "max_area",
            "min_perimeter",
            "max_perimeter",
            "min_width",
            "max_width",
            "min_height",
            "max_height",
            "min_width_plus_height",
            "max_width_plus_height",
        ],
        "roi_recording": [
            "padding",
            "roi_encoding",
            "zstd_min_bytes",
            "store_roi_payload_min_area",
            "store_roi_payload_min_width",
            "store_roi_payload_min_height",
            "store_roi_payload_min_width_plus_height",
            "always_store_mask",
        ],
    }.items():
        if key in fields:
            return group
    return "unknown"


def _dataclass_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    return dict(value)
