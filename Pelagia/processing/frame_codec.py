from typing import Any

import cv2
import numpy as np


def encode_array_payload(array: np.ndarray, encoding: object) -> tuple[bytes, str, str]:
    requested = str(encoding or "png").lower()
    if requested in {"raw", "raw_ndarray_c_order"}:
        return array.tobytes(order="C"), "raw", "raw_ndarray_c_order"

    if requested in {"png", "image/png"}:
        ok, encoded = cv2.imencode(".png", array, [cv2.IMWRITE_PNG_COMPRESSION, 4])
        if not ok:
            raise ValueError(
                f"Frame array with dtype {array.dtype} and shape {array.shape} cannot be encoded as PNG."
            )
        return encoded.tobytes(), "png", "png"

    if requested in {"jpg", "image/jpeg"}:
        ok, encoded = cv2.imencode(".jpg", array, [cv2.IMWRITE_JPEG_QUALITY, 96])
        if not ok:
            raise ValueError(
                f"Frame array with dtype {array.dtype} and shape {array.shape} cannot be encoded as JPG."
            )
        return encoded.tobytes(), "jpg", "jpg"

    if requested in {"zstd", "zstandard", "zstd_ndarray_c_order"}:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required to encode frame arrays as zstd.") from exc

        return (
            zstd.ZstdCompressor(level=3).compress(array.tobytes(order="C")),
            "zstd",
            "zstd_ndarray_c_order",
        )

    raise ValueError("Frame array encoding must be one of: raw, png, jpg, zstd.")


def decode_array_payload(payload: bytes, metadata: dict[str, Any]) -> np.ndarray:
    encoding = str(
        metadata.get(
            "kvstore_encoding",
            metadata.get("array_encoding", metadata.get("kvstore_format", "png")),
        )
    ).lower()

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

    if encoding in {"zstd", "zstandard", "zstd_ndarray_c_order"}:
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise RuntimeError("zstandard is required to decode zstd frame arrays.") from exc
        payload = zstd.ZstdDecompressor().decompress(payload)

    elif encoding not in {"raw", "raw_ndarray_c_order"}:
        raise ValueError("Stored frame array encoding must be one of: raw, png, jpg, zstd.")

    dtype = metadata.get("dtype")
    shape = metadata.get("shape")
    if dtype is None or shape is None:
        raise ValueError("Raw and zstd frame payloads require dtype and shape metadata.")
    return np.frombuffer(payload, dtype=np.dtype(dtype)).reshape(tuple(shape)).copy(order="C")
