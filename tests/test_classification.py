from __future__ import annotations

import numpy as np
import pytest

from Pelagia.processing.classification import refined_bbox_classification_crop


def test_refined_bbox_classification_crop_rejects_bbox_outside_payload():
    with pytest.raises(ValueError, match="falls outside stored crop_bbox"):
        refined_bbox_classification_crop(
            np.zeros((4, 5), dtype="uint8"),
            {
                "id": "refined-1",
                "bbox_x": 9,
                "bbox_y": 20,
                "bbox_w": 2,
                "bbox_h": 2,
                "crop_bbox_x": 10,
                "crop_bbox_y": 20,
                "crop_bbox_w": 5,
                "crop_bbox_h": 4,
            },
        )


def test_refined_bbox_classification_crop_preserves_channels():
    image = np.arange(4 * 5 * 3, dtype="uint8").reshape(4, 5, 3)
    result = refined_bbox_classification_crop(
        image,
        {
            "id": "refined-1",
            "bbox_x": 11,
            "bbox_y": 22,
            "bbox_w": 3,
            "bbox_h": 2,
            "crop_bbox_x": 10,
            "crop_bbox_y": 20,
            "crop_bbox_w": 5,
            "crop_bbox_h": 4,
        },
    )

    np.testing.assert_array_equal(result.image, image[2:4, 1:4, :])
    assert result.image.flags.c_contiguous
    assert result.metadata["input_shape"] == [2, 3, 3]
