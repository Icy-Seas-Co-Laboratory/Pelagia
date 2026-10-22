from __future__ import annotations

import base64
from importlib import metadata
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from thumbhash import rgba_to_thumb_hash, thumb_hash_to_rgba

from Pelagia.processing import thumbhash
from Pelagia.processing.thumbhash import compute_thumbhash, thumbhash_to_base64


# Generated with rgbaToThumbHash from evanw/thumbhash/js/thumbhash.js at commit
# a652ce6ed691242f459f468f0a8756cda3b90a82, using the RGBA pattern below.
_CANONICAL_FIXTURES = (
    ("color", 9, 7, "1cf80915940447a847879683797979785f317070ca"),
    ("portrait", 7, 9, "1d180a1d0a2094477947864789887686da0a1070d4"),
    ("alpha", 9, 7, "dc07860c8a1707485998865afa941306963d83776756301708"),
    ("transparent", 9, 7, "00088204800000000000000000000000000000000000000000"),
)


def _fixture_bgra(kind: str, width: int, height: int) -> np.ndarray:
    """Return the BGRA form of an RGBA pattern hashed by the canonical JS code."""
    y, x = np.indices((height, width))
    rgba = np.empty((height, width, 4), dtype=np.uint8)
    rgba[:, :, 0] = (x * 37 + y * 11) % 256
    rgba[:, :, 1] = (x * 17 + y * 41) % 256
    rgba[:, :, 2] = (x * 71 + y * 3) % 256
    if kind == "alpha":
        rgba[:, :, 3] = (x * 29 + y * 53) % 256
    else:
        rgba[:, :, 3] = 0 if kind == "transparent" else 255
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def _decoded_rgba(payload: bytes) -> tuple[tuple[int, int], np.ndarray]:
    width, height, rgba = thumb_hash_to_rgba(payload)
    return (width, height), np.frombuffer(rgba, dtype=np.uint8).reshape(height, width, 4)


def test_fast_thumbhash_encoder_version_is_pinned():
    """Distributions sharing this import namespace must not coexist after sync."""
    assert metadata.version("fast-thumbhash-rs") == "0.1.0"
    for conflicting in ("thumbhash", "thumbhash-rs"):
        with pytest.raises(metadata.PackageNotFoundError):
            metadata.version(conflicting)


def test_compute_thumbhash_uses_standard_thumbhash_encoder_for_grayscale():
    image = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    rgba = [
        0, 0, 0, 255,
        128, 128, 128, 255,
        255, 255, 255, 255,
        64, 64, 64, 255,
    ]

    payload = compute_thumbhash(image)

    assert payload == rgba_to_thumb_hash(2, 2, bytes(rgba))
    assert not payload.startswith(b"PTH1")


def test_thumbhash_to_base64_encodes_api_transport_string():
    payload = b"\x01\x02\x03"

    assert thumbhash_to_base64(payload) == base64.b64encode(payload).decode("ascii")


@pytest.mark.parametrize(("kind", "width", "height", "canonical_hex"), _CANONICAL_FIXTURES)
def test_native_encoder_remains_compatible_with_canonical_thumbhash_fixtures(
    kind, width, height, canonical_hex
):
    """Compare decoded previews because optimized DCT arithmetic is not byte-stable."""
    image = _fixture_bgra(kind, width, height)
    original = image.copy()

    generated = compute_thumbhash(image)
    canonical = bytes.fromhex(canonical_hex)
    generated_shape, generated_rgba = _decoded_rgba(generated)
    canonical_shape, canonical_rgba = _decoded_rgba(canonical)

    assert generated_shape == canonical_shape
    assert generated_rgba.shape == canonical_rgba.shape
    assert np.abs(generated_rgba.astype(np.int16) - canonical_rgba.astype(np.int16)).mean() <= 3
    if kind == "transparent":
        assert np.all(generated_rgba[:, :, 3] == 0)
    elif kind == "alpha":
        assert generated[2] & 0x80
        assert 0 < generated_rgba[:, :, 3].max() < 255
    else:
        assert not generated[2] & 0x80
        assert np.all(generated_rgba[:, :, 3] == 255)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("alpha", [1, 128, 254])
def test_uniform_non_opaque_alpha_is_preserved_as_thumbhash_alpha(alpha):
    rgba = np.full((9, 7, 4), (20, 100, 220, alpha), dtype=np.uint8)
    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    original = image.copy()

    payload = compute_thumbhash(image)
    _, decoded = _decoded_rgba(payload)

    assert payload[2] & 0x80
    # The 4-bit alpha DC coefficient can quantize an almost-opaque input to 255.
    if alpha < 254:
        assert decoded[:, :, 3].min() < 255
    else:
        assert np.all(decoded[:, :, 3] == 255)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("shape", [(1, 1), (1, 100), (100, 1)])
def test_tiny_and_extreme_aspect_inputs_encode_to_decodable_hashes(shape):
    rgba = np.full((*shape, 4), (20, 100, 220, 255), dtype=np.uint8)
    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    original = image.copy()

    payload = compute_thumbhash(image)
    (width, height), decoded = _decoded_rgba(payload)

    assert isinstance(payload, bytes)
    assert payload
    assert width > 0 and height > 0
    assert decoded.shape == (height, width, 4)
    assert np.all(decoded[:, :, 3] == 255)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("alpha", [0, 255])
def test_constant_images_retain_transparency_semantics(alpha):
    rgba = np.full((17, 23, 4), (30, 150, 200, alpha), dtype=np.uint8)
    image = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)
    original = image.copy()

    payload = compute_thumbhash(image)
    _, decoded = _decoded_rgba(payload)

    assert bool(payload[2] & 0x80) is (alpha == 0)
    assert np.all(decoded[:, :, 3] == alpha)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("channels", [None, 1, 3, 4])
@pytest.mark.parametrize("shape", [(240, 360), (360, 240), (1, 300), (300, 1), (7, 9)])
def test_resize_precedes_rgba_expansion(channels, shape):
    input_shape = shape if channels is None else (*shape, channels)
    image = np.random.default_rng(42).integers(0, 256, input_shape, dtype=np.uint8)
    original = image.copy()
    scale = min(64 / max(shape), 1)
    width, height = max(1, round(shape[1] * scale)), max(1, round(shape[0] * scale))
    with (
        patch.object(thumbhash.cv2, "cvtColor", wraps=cv2.cvtColor) as convert,
        patch.object(thumbhash.cv2, "resize", wraps=cv2.resize) as resize,
        patch.object(thumbhash, "rgba_to_thumb_hash", return_value=b"\x01\x02\x03") as encode,
    ):
        assert compute_thumbhash(image, max_dim=64) == b"\x01\x02\x03"

    assert convert.call_count == 1
    assert convert.call_args.args[0].shape[:2] == (height, width)
    if scale < 1:
        assert resize.call_count == 1
        assert resize.call_args.args[0].shape == (shape if channels in {None, 1} else image.shape)
        assert resize.call_args.kwargs["interpolation"] == cv2.INTER_LINEAR
    else:
        resize.assert_not_called()
    encoded_width, encoded_height, rgba = encode.call_args.args
    assert (encoded_width, encoded_height) == (width, height)
    assert isinstance(rgba, bytes)
    assert len(rgba) == width * height * 4
    conversion = {None: cv2.COLOR_GRAY2RGBA, 1: cv2.COLOR_GRAY2RGBA, 3: cv2.COLOR_BGR2RGBA, 4: cv2.COLOR_BGRA2RGBA}[channels]
    expected = original
    if scale < 1:
        expected = cv2.resize(expected, (width, height), interpolation=cv2.INTER_LINEAR)
    expected = cv2.cvtColor(expected, conversion)
    assert rgba == expected.tobytes()
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("channels", [1, 3, 4])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("shape", [(120, 180), (121, 179)])
def test_hash_uses_bilinear_resize_after_full_source_normalization(channels, dtype, shape):
    image = np.random.default_rng(123).integers(0, 256, (*shape, channels), dtype=np.uint8).astype(dtype)
    if dtype != np.uint8:
        image *= 2
    if channels == 4:
        image[:, :, 3] = 255 if dtype == np.uint8 else 510
    # Exercise strided input and retain the full-source normalization contract.
    image = image[:, ::-1]
    normalized = image if dtype == np.uint8 else cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    conversion = {1: cv2.COLOR_GRAY2RGBA, 3: cv2.COLOR_BGR2RGBA, 4: cv2.COLOR_BGRA2RGBA}[channels]
    height = round(shape[0] * 30 / shape[1])
    # OpenCV's channel-specific resize kernels can round differently, so the
    # reference must also interpolate the original channels before RGBA expansion.
    preview = cv2.resize(normalized, (30, height), interpolation=cv2.INTER_LINEAR)
    preview = cv2.cvtColor(preview, conversion)
    expected = rgba_to_thumb_hash(30, height, preview.tobytes())

    assert compute_thumbhash(image, max_dim=30) == expected


@pytest.mark.parametrize("max_dim", [0, -10, 101, 1.5, True])
def test_invalid_max_dim_rejected(max_dim):
    with pytest.raises(ValueError, match="max_dim"):
        compute_thumbhash(np.zeros((2, 2), dtype=np.uint8), max_dim=max_dim)


@pytest.mark.parametrize("max_dim", [1, np.int64(100)])
def test_valid_max_dim_boundaries(max_dim):
    assert compute_thumbhash(np.zeros((2, 2), dtype=np.uint8), max_dim=max_dim)


@pytest.mark.parametrize("shape", [(0, 3), (3, 0, 4), (0, 3, 1)])
def test_empty_images_rejected_before_opencv(shape):
    with pytest.raises(ValueError, match="non-empty"):
        compute_thumbhash(np.empty(shape, dtype=np.uint16))


@pytest.mark.parametrize("shape", [(3,), (2, 2, 2), (2, 2, 3, 1)])
def test_unsupported_shapes_rejected(shape):
    with pytest.raises(ValueError, match="2D|Unsupported image shape"):
        compute_thumbhash(np.zeros(shape, dtype=np.uint8))
