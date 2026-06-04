from typing import Any

import numpy as np

from .defaults import default_processing_config


def metadata_bool(metadata: dict[str, Any], key: str, default: bool = False) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def flatfield_correction_for_framedata(
    data: np.ndarray,
    q: float | None = None,
    min_field_value: float | None = None,
) -> np.ndarray:
    defaults = default_processing_config().video_ingest
    q = defaults.flatfield_q if q is None else q
    min_field_value = 0.0 if min_field_value is None else min_field_value
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1.")

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(
            "flatfield correction currently expects a 2D grayscale frame.")

    field = np.maximum(np.quantile(array, q=q, axis=0), min_field_value)
    corrected = array / field.T * 255.0
    return np.ascontiguousarray(np.clip(corrected, 0, 255).astype(np.uint8))


def flatfield_global_correction_for_framedata(
    data: np.ndarray,
    q: float | None = None,
    axis: int | None = None,
) -> np.ndarray:
    defaults = default_processing_config().video_ingest
    q = defaults.flatfield_q if q is None else q
    axis = defaults.flatfield_axis if axis is None else axis
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1.")

    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("flatfield correction requires at least two dimensions.")

    if axis == 0:
        field = np.quantile(array, q=q, axis=0).astype(np.float32)
    elif axis == 1:
        if array.ndim == 2:
            field = np.quantile(array, q=q, axis=1).astype(np.float32)[:, None]
        else:
            field = np.quantile(array, q=q, axis=1).astype(np.float32)[:, None, :]
    else:
        raise ValueError("flatfield axis must be 0 or 1.")

    scale = float(np.mean(field[field > 0])) if np.any(field > 0) else 1.0
    corrected = array.astype(np.float32) / np.maximum(field, 1.0) * scale

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(array.dtype)

    return np.ascontiguousarray(corrected)
