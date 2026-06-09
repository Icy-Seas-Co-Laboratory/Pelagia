from __future__ import annotations

try:
    from fastapi import HTTPException
except ImportError:  # pragma: no cover
    HTTPException = None  # type: ignore

import cv2
import numpy as np


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
    image = np.ascontiguousarray(array)
    requested = fmt.lower()
    if requested == "jpg":
        requested = "jpeg"
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
