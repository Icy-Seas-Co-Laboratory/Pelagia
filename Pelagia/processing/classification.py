"""Classification input preparation owned by Pelagia."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


CLASSIFICATION_CROP_POLICY = "refined_detection_bbox_v1"


@dataclass(frozen=True, slots=True)
class ClassificationCrop:
    image: np.ndarray
    metadata: dict[str, Any]


def refined_bbox_classification_crop(
    image: np.ndarray,
    detection: Mapping[str, Any],
) -> ClassificationCrop:
    """Crop a stored, padded refined ROI to its frame-coordinate object bbox."""

    detection_id = str(detection.get("id") or "unknown")
    array = np.asarray(image)
    if array.ndim not in {2, 3}:
        raise ValueError(
            f"Refined detection {detection_id} has unsupported ROI shape {array.shape}."
        )

    names = (
        "bbox_x", "bbox_y", "bbox_w", "bbox_h",
        "crop_bbox_x", "crop_bbox_y", "crop_bbox_w", "crop_bbox_h",
    )
    try:
        geometry = {name: int(detection[name]) for name in names}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Refined detection {detection_id} requires complete bbox and crop_bbox geometry."
        ) from exc

    bbox_x, bbox_y, bbox_w, bbox_h = (geometry[name] for name in names[:4])
    crop_x, crop_y, crop_w, crop_h = (geometry[name] for name in names[4:])
    if bbox_w <= 0 or bbox_h <= 0 or crop_w <= 0 or crop_h <= 0:
        raise ValueError(f"Refined detection {detection_id} has non-positive bbox geometry.")
    if tuple(array.shape[:2]) != (crop_h, crop_w):
        raise ValueError(
            f"Refined detection {detection_id} ROI shape {array.shape[:2]} does not match "
            f"stored crop_bbox dimensions {(crop_h, crop_w)}."
        )

    local_x = bbox_x - crop_x
    local_y = bbox_y - crop_y
    x1 = local_x + bbox_w
    y1 = local_y + bbox_h
    if local_x < 0 or local_y < 0 or x1 > crop_w or y1 > crop_h:
        raise ValueError(
            f"Refined detection {detection_id} bbox {(bbox_x, bbox_y, bbox_w, bbox_h)} "
            f"falls outside stored crop_bbox {(crop_x, crop_y, crop_w, crop_h)}."
        )

    cropped = np.ascontiguousarray(array[local_y:y1, local_x:x1, ...])
    return ClassificationCrop(
        image=cropped,
        metadata={
            "policy": CLASSIFICATION_CROP_POLICY,
            "frame_bbox": [bbox_x, bbox_y, bbox_w, bbox_h],
            "stored_crop_bbox": [crop_x, crop_y, crop_w, crop_h],
            "stored_shape": list(array.shape),
            "input_shape": list(cropped.shape),
        },
    )
