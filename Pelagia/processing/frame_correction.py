from typing import Any

import numpy as np


def metadata_bool(metadata: dict[str, Any], key: str, default: bool = False) -> bool:
    value = metadata.get(key, default)
    if isinstance(value, str):
        return value.lower() in {"1", "true", "yes", "on"}
    return bool(value)


def flatfield_correction_for_framedata(
    data: np.ndarray,
    q: float = 0.9,
    output_max: float = 255.0,
    min_field_value: float = 0.0,
) -> np.ndarray:
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1.")

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError(
            "flatfield correction currently expects a 2D grayscale frame.")

    field = np.quantile(array, q=q, axis=0)
    corrected = array / field.T * output_max
    return np.ascontiguousarray(np.clip(corrected, 0, 255).astype(np.uint8))


def flatfield_global_correction_for_framedata(
    data: np.ndarray,
    q: float = 0.95,
    axis: int = 0,
) -> np.ndarray:
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


_metadata_bool = metadata_bool
_flatfield_correction_for_framedata = flatfield_correction_for_framedata
