from __future__ import annotations

import base64

import numpy as np
from thumbhash import rgba_to_thumb_hash

from Pelagia.processing.thumbhash import compute_thumbhash, thumbhash_to_base64


def test_compute_thumbhash_uses_standard_thumbhash_encoder_for_grayscale():
    image = np.array([[0, 128], [255, 64]], dtype=np.uint8)
    rgba = [
        0, 0, 0, 255,
        128, 128, 128, 255,
        255, 255, 255, 255,
        64, 64, 64, 255,
    ]

    payload = compute_thumbhash(image)

    assert payload == bytes(rgba_to_thumb_hash(2, 2, rgba))
    assert not payload.startswith(b"PTH1")


def test_thumbhash_to_base64_encodes_api_transport_string():
    payload = b"\x01\x02\x03"

    assert thumbhash_to_base64(payload) == base64.b64encode(payload).decode("ascii")
