from __future__ import annotations

import base64

import cv2
import numpy as np
from thumbhash import rgba_to_thumb_hash


def compute_thumbhash(array: np.ndarray, *, max_dim: int = 100) -> bytes:
    """
    Compute a standard ThumbHash payload for a frame-like image array.

    Color arrays are treated as OpenCV-style BGR/BGRA inputs because Pelagia's
    image ingestion stack is cv2-based. Grayscale arrays are expanded to RGBA.
    """
    image = np.asarray(array)
    if image.ndim < 2:
        raise ValueError("ThumbHash generation requires at least 2D image data.")

    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if image.ndim == 2:
        preview_source = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
    elif image.ndim == 3 and image.shape[2] in {1, 3, 4}:
        channels = int(image.shape[2])
        if channels == 1:
            preview_source = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGBA)
        elif channels == 3:
            preview_source = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
        else:
            preview_source = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
    else:
        raise ValueError(f"Unsupported image shape for ThumbHash generation: {image.shape}.")

    height, width = preview_source.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("ThumbHash generation requires non-empty image data.")

    scale = min(float(max_dim) / float(width), float(max_dim) / float(height), 1.0)
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    if scale < 1.0:
        preview = cv2.resize(
            np.ascontiguousarray(preview_source),
            (preview_width, preview_height),
            interpolation=cv2.INTER_AREA,
        )
    else:
        preview = np.ascontiguousarray(preview_source)

    rgba = np.ascontiguousarray(preview).reshape(-1).tolist()
    return bytes(rgba_to_thumb_hash(preview_width, preview_height, rgba))


def thumbhash_to_base64(payload: bytes) -> str:
    """Encode ThumbHash bytes for JSON APIs and browser clients."""
    return base64.b64encode(payload).decode("ascii")
