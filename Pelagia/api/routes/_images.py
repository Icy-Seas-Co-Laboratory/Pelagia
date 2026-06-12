from __future__ import annotations

try:
    from fastapi import HTTPException
except ImportError:  # pragma: no cover
    HTTPException = None  # type: ignore

import cv2
import numpy as np


def pad_image_to_square(array, *, fill_value: int | float = 0) -> np.ndarray:
    """Center an image on a square canvas using the longest image dimension."""
    image = np.asarray(array)
    if image.ndim < 2:
        raise HTTPException(status_code=422, detail="Square padding requires at least 2D image data.")

    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise HTTPException(status_code=422, detail="Square padding requires non-empty image data.")
    side = max(height, width)
    if height == side and width == side:
        return np.ascontiguousarray(image)

    canvas_shape = (side, side, *image.shape[2:])
    canvas = np.full(canvas_shape, fill_value, dtype=image.dtype)
    y0 = (side - height) // 2
    x0 = (side - width) // 2
    canvas[y0:y0 + height, x0:x0 + width, ...] = image
    return np.ascontiguousarray(canvas)


def invert_image(array) -> np.ndarray:
    """Invert image intensity while preserving dtype."""
    image = np.asarray(array)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        inverted = info.max - image
    elif np.issubdtype(image.dtype, np.floating):
        max_value = 1.0 if image.size == 0 else float(np.nanmax(image))
        inverted = max_value - image
    else:
        raise HTTPException(status_code=422, detail="Image inversion requires numeric image data.")
    return np.ascontiguousarray(inverted.astype(image.dtype, copy=False))


def add_scale_bar(
    array,
    *,
    length_px: int | None = None,
    height_px: int = 4,
    margin_px: int = 8,
    color: str = "white",
) -> np.ndarray:
    """Draw a simple lower-left pixel scale bar onto an image."""
    image = np.array(array, copy=True)
    if image.ndim < 2:
        raise HTTPException(status_code=422, detail="Scale bar requires at least 2D image data.")

    image_height, image_width = image.shape[:2]
    if image_height < 1 or image_width < 1:
        raise HTTPException(status_code=422, detail="Scale bar requires non-empty image data.")
    if height_px < 1:
        raise HTTPException(status_code=422, detail="scale_bar_height_px must be >= 1.")
    if margin_px < 0:
        raise HTTPException(status_code=422, detail="scale_bar_margin_px must be >= 0.")

    available_width = image_width - (2 * int(margin_px))
    if available_width < 1:
        raise HTTPException(status_code=422, detail="Image is too small for the requested scale bar margin.")
    resolved_length = int(length_px) if length_px is not None else max(1, min(50, available_width // 4))
    if resolved_length < 1:
        raise HTTPException(status_code=422, detail="scale_bar_length_px must be >= 1.")
    if resolved_length > available_width:
        raise HTTPException(status_code=422, detail="scale_bar_length_px exceeds available image width.")

    resolved_height = min(int(height_px), image_height)
    x0 = int(margin_px)
    y1 = image_height - int(margin_px)
    if y1 <= 0:
        y1 = image_height
    y0 = max(0, y1 - resolved_height)
    x1 = min(image_width, x0 + resolved_length)
    image[y0:y1, x0:x1, ...] = _scale_bar_value(image, color)
    return np.ascontiguousarray(image)


def _scale_bar_value(image: np.ndarray, color: str):
    normalized = str(color).strip().lower()
    if normalized not in {"white", "black"}:
        raise HTTPException(status_code=422, detail="scale_bar_color must be one of: white, black.")
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        value = info.max if normalized == "white" else info.min
    elif np.issubdtype(image.dtype, np.floating):
        value = 1.0 if normalized == "white" else 0.0
    else:
        raise HTTPException(status_code=422, detail="Scale bar requires numeric image data.")
    if image.ndim == 2:
        return value
    channels = image.shape[2]
    return np.array([value] * channels, dtype=image.dtype)


def preview_image(array, max_dim: int) -> np.ndarray:
    if max_dim < 1:
        raise HTTPException(status_code=422, detail="preview_max_dim must be >= 1.")

    image = np.asarray(array)
    if image.ndim < 2:
        raise HTTPException(status_code=422, detail="Image preview requires at least 2D image data.")

    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise HTTPException(status_code=422, detail="Image preview requires non-empty image data.")

    scale = min(float(max_dim) / float(width), float(max_dim) / float(height), 1.0)
    if scale >= 1.0:
        return np.ascontiguousarray(image)

    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    return cv2.resize(
        np.ascontiguousarray(image),
        (preview_width, preview_height),
        interpolation=cv2.INTER_AREA,
    )


def scale_image(array, scale: float) -> np.ndarray:
    if not np.isfinite(scale) or scale <= 0.0 or scale > 1.0:
        raise HTTPException(status_code=422, detail="scale must be > 0 and <= 1.")

    image = np.asarray(array)
    if image.ndim < 2:
        raise HTTPException(status_code=422, detail="Image scaling requires at least 2D image data.")

    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise HTTPException(status_code=422, detail="Image scaling requires non-empty image data.")

    if scale == 1.0:
        return np.ascontiguousarray(image)

    scaled_width = max(1, int(round(width * scale)))
    scaled_height = max(1, int(round(height * scale)))
    return cv2.resize(
        np.ascontiguousarray(image),
        (scaled_width, scaled_height),
        interpolation=cv2.INTER_AREA,
    )


def resize_image_to_dimension(
    array,
    *,
    width: int | None = None,
    height: int | None = None,
) -> np.ndarray:
    if width is None and height is None:
        return np.ascontiguousarray(array)
    if width is not None and height is not None:
        raise HTTPException(status_code=422, detail="Provide width or height, not both.")
    if width is not None and width < 1:
        raise HTTPException(status_code=422, detail="width must be >= 1.")
    if height is not None and height < 1:
        raise HTTPException(status_code=422, detail="height must be >= 1.")

    image = np.asarray(array)
    if image.ndim < 2:
        raise HTTPException(status_code=422, detail="Image resizing requires at least 2D image data.")

    source_height, source_width = image.shape[:2]
    if source_height < 1 or source_width < 1:
        raise HTTPException(status_code=422, detail="Image resizing requires non-empty image data.")

    if width is not None:
        scale = float(width) / float(source_width)
        target_width = int(width)
        target_height = max(1, int(round(source_height * scale)))
    else:
        scale = float(height) / float(source_height)
        target_height = int(height)
        target_width = max(1, int(round(source_width * scale)))

    if target_width == source_width and target_height == source_height:
        return np.ascontiguousarray(image)

    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        np.ascontiguousarray(image),
        (target_width, target_height),
        interpolation=interpolation,
    )


def encode_image(array, fmt: str) -> tuple[bytes, str]:
    requested = fmt.lower()
    if requested == "jpg":
        requested = "jpeg"
    image = _prepare_image_for_encoding(array, requested)
    if requested == "png":
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        media_type = "image/png"
    elif requested == "jpeg":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        media_type = "image/jpeg"
    else:
        raise HTTPException(
            status_code=422,
            detail="Image data format must be one of: png, jpg, jpeg, matrix.",
        )
    if not ok:
        raise HTTPException(
            status_code=500,
            detail=f"Image data could not be encoded as {requested}.",
        )
    return encoded.tobytes(), media_type


def _prepare_image_for_encoding(array, fmt: str) -> np.ndarray:
    """Convert scientific frame arrays into browser-encodable image arrays."""
    image = np.asarray(array)
    if image.ndim < 2:
        raise HTTPException(status_code=422, detail="Image encoding requires at least 2D image data.")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise HTTPException(status_code=422, detail="Image encoding requires non-empty image data.")

    image = _normalize_image_channels(image)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if fmt == "png" and image.dtype == np.uint16:
        return np.ascontiguousarray(image)
    return _normalize_to_uint8(image)


def _normalize_image_channels(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.ndim == 3 and image.shape[2] in {1, 3, 4}:
        return image[:, :, 0] if image.shape[2] == 1 else image
    squeezed = np.squeeze(image)
    if squeezed.ndim == 2:
        return squeezed
    if squeezed.ndim == 3 and squeezed.shape[2] in {1, 3, 4}:
        return squeezed[:, :, 0] if squeezed.shape[2] == 1 else squeezed
    raise HTTPException(
        status_code=422,
        detail="Image encoding supports grayscale, RGB, or RGBA image arrays.",
    )


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.bool_:
        return np.ascontiguousarray(image.astype(np.uint8) * 255)

    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        clipped = np.clip(image, info.min, info.max).astype(np.float32)
        if info.min == 0 and info.max <= 255:
            return np.ascontiguousarray(clipped.astype(np.uint8))
        return _scale_numeric_to_uint8(clipped)

    if np.issubdtype(image.dtype, np.floating):
        return _scale_numeric_to_uint8(image.astype(np.float32, copy=False))

    raise HTTPException(status_code=422, detail="Image encoding requires numeric image data.")


def _scale_numeric_to_uint8(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)

    finite_values = image[finite]
    min_value = float(np.min(finite_values))
    max_value = float(np.max(finite_values))
    safe = np.nan_to_num(image, nan=min_value, neginf=min_value, posinf=max_value)

    if min_value >= 0.0 and max_value <= 1.0:
        scaled = safe * 255.0
    elif max_value > min_value:
        scaled = (safe - min_value) * (255.0 / (max_value - min_value))
    else:
        scaled = np.zeros(image.shape, dtype=np.float32)
    return np.ascontiguousarray(np.clip(scaled, 0, 255).astype(np.uint8))
