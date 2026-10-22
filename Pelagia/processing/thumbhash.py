"""Generate compact frame previews using the ThumbHash format."""

from __future__ import annotations

import base64
from numbers import Integral

import cv2
import numpy as np
from thumbhash import rgba_to_thumb_hash

from .defaults import default_processing_config


def compute_thumbhash(array: np.ndarray, *, max_dim: int | None = None) -> bytes:
    """
    Compute a standard ThumbHash payload for a frame-like image array.

    Color arrays are treated as OpenCV-style BGR/BGRA inputs because Pelagia's
    image ingestion stack is cv2-based. Resize in the source channel layout
    before expanding to RGBA, bounding the expanded buffer to max_dim squared
    pixels. Non-uint8 data retains full-source min/max normalization.
    """
    resolved_max_dim = default_processing_config().thumbhash.max_dim if max_dim is None else max_dim
    if (
        isinstance(resolved_max_dim, bool)
        or not isinstance(resolved_max_dim, Integral)
        or not 1 <= resolved_max_dim <= 100
    ):
        raise ValueError("ThumbHash max_dim must be an integer between 1 and 100.")
    image = np.asarray(array)
    if image.ndim < 2:
        raise ValueError("ThumbHash generation requires at least 2D image data.")
    if image.ndim != 2 and not (image.ndim == 3 and image.shape[2] in {1, 3, 4}):
        raise ValueError(f"Unsupported image shape for ThumbHash generation: {image.shape}.")
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("ThumbHash generation requires non-empty image data.")

    # Keep full-source normalization and uint8 quantization before resampling:
    # moving these after resize would change contrast and rounding for scientific inputs.
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    channels = 1 if image.ndim == 2 else image.shape[2]
    if image.ndim == 3 and channels == 1:
        image = image[:, :, 0]

    scale = min(float(resolved_max_dim) / float(width), float(resolved_max_dim) / float(height), 1.0)
    preview_width = max(1, int(round(width * scale)))
    preview_height = max(1, int(round(height * scale)))
    if scale < 1.0:
        # This display placeholder favors throughput over antialiased downsampling.
        image = cv2.resize(
            image,
            (preview_width, preview_height),
            interpolation=cv2.INTER_LINEAR,
        )
    conversion = {
        1: cv2.COLOR_GRAY2RGBA,
        3: cv2.COLOR_BGR2RGBA,
        4: cv2.COLOR_BGRA2RGBA,
    }[channels]
    preview = cv2.cvtColor(image, conversion)
    # The native binding borrows immutable bytes and releases the GIL while encoding.
    return rgba_to_thumb_hash(preview_width, preview_height, preview.tobytes())


def thumbhash_to_base64(payload: bytes) -> str:
    """Encode ThumbHash bytes for JSON APIs and browser clients."""
    return base64.b64encode(payload).decode("ascii")
