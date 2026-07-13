from typing import Any

import cv2
import numpy as np

from .codec_registry import normalize_image_encoding


DEFAULT_LOSSY_IMAGE_QUALITY = 90
DEFAULT_JXL_EFFORT = 1


def encode_array_payload(array: np.ndarray, encoding: object, *, quality: int | None = None) -> tuple[bytes, str, str]:
    try:
        requested = normalize_image_encoding(encoding or "zstd")
    except ValueError:
        requested = str(encoding or "zstd").lower()
    if requested in {"raw", "raw_ndarray_c_order"}:
        return array.tobytes(order="C"), "raw", "raw_ndarray_c_order"

    if requested in {"png", "image/png"}:
        ok, encoded = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if not ok:
            raise ValueError(
                f"Frame array with dtype {array.dtype} and shape {array.shape} cannot be encoded as PNG."
            )
        return encoded.tobytes(), "png", "png"

    if requested in {"jpg", "jpeg", "image/jpeg"}:
        resolved_quality = _normalize_lossy_quality(quality)
        ok, encoded = cv2.imencode(
            ".jpg",
            array,
            [cv2.IMWRITE_JPEG_QUALITY, resolved_quality],
        )
        if not ok:
            raise ValueError(
                f"Frame array with dtype {array.dtype} and shape {array.shape} cannot be encoded as JPG."
            )
        return encoded.tobytes(), "jpg", "jpg"

    if requested in {"jxl", "jpegxl", "jpeg-xl", "jpeg_xl", "image/jxl"}:
        return _encode_jxl_payload(
            array,
            quality=_normalize_lossy_quality(quality),
        ), "jxl", "jxl"

    if requested in {"jxs", "jpegxs", "jpeg-xs", "jpeg_xs", "image/jxs"}:
        return _encode_jxs_payload(array), "jxs", "jxs"

    if requested in {"zstd", "zstandard", "zstd_ndarray_c_order"}:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required to encode frame arrays as zstd.") from exc

        return (
            zstd.ZstdCompressor(level=2, threads=1).compress(array.tobytes(order="C")),
            "zstd",
            "zstd_ndarray_c_order",
        )

    raise ValueError("Frame array encoding must be one of: raw, png, jpg, jxl, jxs, zstd.")


def decode_array_payload(payload: bytes, metadata: dict[str, Any]) -> np.ndarray:
    encoding = str(
        metadata.get(
            "kvstore_encoding",
            metadata.get("array_encoding", metadata.get("kvstore_format", "png")),
        )
    ).lower()
    try:
        encoding = normalize_image_encoding(encoding)
    except ValueError:
        pass

    if encoding in {"png", "image/png"}:
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Stored frame payload could not be decoded as PNG.")
        return np.ascontiguousarray(decoded)

    if encoding in {"jpg", "jpeg", "image/jpeg"}:
        decoded = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if decoded is None:
            raise ValueError("Stored frame payload could not be decoded as JPG.")
        return np.ascontiguousarray(decoded)

    if encoding in {"jxl", "jpegxl", "jpeg-xl", "jpeg_xl", "image/jxl"}:
        return _decode_jxl_payload(payload)

    if encoding in {"jxs", "jpegxs", "jpeg-xs", "jpeg_xs", "image/jxs"}:
        return _decode_jxs_payload(payload, expected_shape=metadata.get("shape"))

    if encoding in {"zstd", "zstandard", "zstd_ndarray_c_order"}:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required to decode zstd frame arrays.") from exc
        payload = zstd.ZstdDecompressor().decompress(payload)

    elif encoding not in {"raw", "raw_ndarray_c_order"}:
        raise ValueError("Stored frame array encoding must be one of: raw, png, jpg, jxl, jxs, zstd.")

    dtype = metadata.get("dtype")
    shape = metadata.get("shape")
    if dtype is None or shape is None:
        raise ValueError("Raw and zstd frame payloads require dtype and shape metadata.")
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(tuple(shape)).copy(order="C")


def _normalize_lossy_quality(value: int | None) -> int:
    quality = DEFAULT_LOSSY_IMAGE_QUALITY if value is None else int(value)
    if quality < 0 or quality > 100:
        raise ValueError("Image quality must be between 0 and 100.")
    return quality


def _encode_jxl_payload(array: np.ndarray, *, quality: int) -> bytes:
    imagecodecs = _imagecodecs()
    image = np.ascontiguousarray(array)
    try:
        return bytes(
            imagecodecs.jpegxl_encode(
                image,
                level=int(quality),
                effort=DEFAULT_JXL_EFFORT,
                numthreads=4,
            )
        )
    except Exception as exc:
        raise RuntimeError(f"JPEG XL encoding failed: {exc}") from exc


def _decode_jxl_payload(payload: bytes) -> np.ndarray:
    imagecodecs = _imagecodecs()
    try:
        return np.ascontiguousarray(imagecodecs.jpegxl_decode(payload))
    except Exception as exc:
        raise RuntimeError(f"JPEG XL decoding failed: {exc}") from exc


def _encode_jxs_payload(array: np.ndarray) -> bytes:
    imagecodecs = _imagecodecs()
    _require_imagecodec(imagecodecs, "JPEGXS", "JPEG XS")
    image = np.ascontiguousarray(array)
    if image.ndim == 2:
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
    elif image.ndim == 3 and image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    try:
        return bytes(imagecodecs.jpegxs_encode(image))
    except Exception as exc:
        raise RuntimeError(f"JPEG XS encoding failed: {exc}") from exc


def _decode_jxs_payload(payload: bytes, *, expected_shape: object = None) -> np.ndarray:
    imagecodecs = _imagecodecs()
    _require_imagecodec(imagecodecs, "JPEGXS", "JPEG XS")
    try:
        decoded = np.ascontiguousarray(imagecodecs.jpegxs_decode(payload))
    except Exception as exc:
        raise RuntimeError(f"JPEG XS decoding failed: {exc}") from exc

    shape = tuple(expected_shape) if isinstance(expected_shape, (list, tuple)) else None
    if decoded.ndim == 3 and decoded.shape[2] == 3:
        if shape is not None and len(shape) == 2:
            return np.ascontiguousarray(decoded[:, :, 0])
        if shape is not None and len(shape) == 3 and shape[2] == 1:
            return np.ascontiguousarray(decoded[:, :, :1])
    return decoded


def _require_imagecodec(imagecodecs, attribute: str, label: str) -> None:
    codec = getattr(imagecodecs, attribute, None)
    if codec is None or not getattr(codec, "available", False):
        raise RuntimeError(
            f"{label} support is not available in this imagecodecs build. "
            "Install imagecodecs from source with the codec's native library enabled."
        )


def _imagecodecs():
    try:
        import imagecodecs
    except ImportError as exc:
        raise RuntimeError(
            "JPEG XL and JPEG XS frame storage require the 'imagecodecs' Python package. "
            "Install Pelagia with its core dependencies or choose image_encoding='zstd', 'png', 'jpg', or 'raw'."
        ) from exc
    return imagecodecs
