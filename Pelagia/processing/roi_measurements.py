"""Canonical geometric measurements for ROI masks."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .mask_augmentation import as_binary_mask


def external_mask_contours(mask: np.ndarray) -> list[np.ndarray]:
    """Return external OpenCV contours for a two-dimensional binary mask.

    Inner hole boundaries are excluded; hole areas are not subtracted.
    This matches Pelagia's historical stored
    contour-perimeter convention and gives one stable definition for persisted
    refined ROI geometry and read-time compatibility measurements.
    """
    binary = as_binary_mask(mask)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [np.ascontiguousarray(contour) for contour in contours if contour.size]


def measure_contours(contours: Sequence[np.ndarray]) -> tuple[float, float]:
    """Return summed OpenCV area and closed arc length for external contours."""
    area = float(sum(cv2.contourArea(contour) for contour in contours))
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    return area, perimeter


def measure_mask_contours(mask: np.ndarray) -> tuple[float, float]:
    """Return external-contour area (px²) and perimeter (px) for a mask."""
    return measure_contours(external_mask_contours(mask))
