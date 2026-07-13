from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

@dataclass(frozen=True, slots=True)
class ImageCodec:
    name: str
    aliases: frozenset[str]
    extension: str
    media_type: str
    storage_supported: bool = True
    response_supported: bool = True


IMAGE_CODECS = (
    ImageCodec("png", frozenset({"png", "image/png"}), "png", "image/png"),
    ImageCodec("jpg", frozenset({"jpg", "jpeg", "image/jpeg"}), "jpg", "image/jpeg"),
    ImageCodec("jxl", frozenset({"jxl", "jpegxl", "jpeg-xl", "jpeg_xl", "image/jxl"}), "jxl", "image/jxl"),
    ImageCodec("jxs", frozenset({"jxs", "jpegxs", "jpeg-xs", "jpeg_xs", "image/jxs"}), "jxs", "image/jxs"),
    ImageCodec("raw", frozenset({"raw", "raw_ndarray_c_order"}), "raw", "application/octet-stream", response_supported=False),
    ImageCodec("zstd", frozenset({"zstd", "zstandard", "zstd_ndarray_c_order"}), "zst", "application/zstd", response_supported=False),
)
_CODECS_BY_ALIAS = {alias: codec for codec in IMAGE_CODECS for alias in codec.aliases}
STORAGE_ENCODINGS = frozenset(codec.name for codec in IMAGE_CODECS)
RESPONSE_ENCODINGS = frozenset(codec.name for codec in IMAGE_CODECS if codec.response_supported)


def normalize_image_encoding(value: object, *, response: bool = False) -> str:
    normalized = str(value).strip().lower()
    codec = _CODECS_BY_ALIAS.get(normalized)
    if codec is None or (response and not codec.response_supported):
        supported = "png, jpg, jpeg, jxl, jxs" if response else "png, jpg, jxl, jxs, raw, zstd"
        raise ValueError(f"Image encoding must be one of: {supported}.")
    return codec.name


def image_codec(value: object, *, response: bool = False) -> ImageCodec:
    return _CODECS_BY_ALIAS[normalize_image_encoding(value, response=response)]


def image_extension(value: object, *, response: bool = True) -> str:
    return image_codec(value, response=response).extension


def image_media_type(value: object, *, response: bool = True) -> str:
    return image_codec(value, response=response).media_type


def image_codec_available(value: object) -> bool:
    name = normalize_image_encoding(value)
    if name in {"png", "jpg", "raw", "zstd"}:
        return True
    try:
        import imagecodecs
    except ImportError:
        return False
    attribute = "JPEGXL" if name == "jxl" else "JPEGXS"
    codec = getattr(imagecodecs, attribute, None)
    return bool(codec is not None and getattr(codec, "available", False))


def encode_image_response(array, value: object, *, quality: int = 95) -> tuple[bytes, str]:
    """Encode an array for an HTTP image response without API-layer dependencies."""
    name = normalize_image_encoding(value, response=True)
    image = prepare_image_for_response(array, name)
    if name == "png":
        ok, encoded = cv2.imencode(".png", image, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if not ok:
            raise RuntimeError("Image data could not be encoded as png.")
        return encoded.tobytes(), image_media_type(name)
    if name == "jpg":
        ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError("Image data could not be encoded as jpg.")
        return encoded.tobytes(), image_media_type(name)
    from .frame_codec import encode_array_payload

    payload, _, _ = encode_array_payload(image, name, quality=90 if name == "jxl" else None)
    return payload, image_media_type(name)


def prepare_image_for_response(array, encoding: str) -> np.ndarray:
    image = np.asarray(array)
    if image.ndim < 2:
        raise ValueError("Image encoding requires at least 2D image data.")
    if image.shape[0] < 1 or image.shape[1] < 1:
        raise ValueError("Image encoding requires non-empty image data.")
    image = _normalize_image_channels(image)
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)
    if encoding == "png" and image.dtype == np.uint16:
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
    raise ValueError("Image encoding supports grayscale, RGB, or RGBA image arrays.")


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
    raise ValueError("Image encoding requires numeric image data.")


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
