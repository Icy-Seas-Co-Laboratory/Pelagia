import cv2
import numpy as np
import pytest

from Pelagia.processing.detection_recording import (
    MajorAxisOrientation,
    component_major_axis_orientation,
)
from Pelagia.processing.mask_augmentation import as_binary_mask


def _coordinate_reference(mask: np.ndarray) -> MajorAxisOrientation:
    """Pre-optimization coordinate-array implementation used as an oracle."""
    binary = as_binary_mask(mask)
    rows, columns = np.nonzero(binary)
    if columns.size == 0:
        raise ValueError("ROI orientation requires at least one foreground pixel.")

    points = np.column_stack((columns, rows)).astype(np.float64, copy=False)
    centroid = np.mean(points, axis=0)
    if len(points) < 2:
        return MajorAxisOrientation(float(centroid[0]), float(centroid[1]), None, None, None, None)

    covariance = np.cov(points, rowvar=False, bias=True)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major_variance = float(eigenvalues[-1])
    minor_variance = float(eigenvalues[0])
    if major_variance <= 0 or np.isclose(major_variance, minor_variance, rtol=1e-6, atol=1e-12):
        return MajorAxisOrientation(float(centroid[0]), float(centroid[1]), None, None, None, None)

    vector = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    if vector[0] < 0 or (np.isclose(vector[0], 0.0) and vector[1] < 0):
        vector *= -1
    orientation = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
    projections = points @ vector
    axis_center = vector * ((float(np.min(projections)) + float(np.max(projections))) / 2.0)
    centroid_offset = float(np.dot(centroid - axis_center, vector))
    direction = (
        None
        if np.isclose(centroid_offset, 0.0, atol=1e-9)
        else float((orientation + (0.0 if centroid_offset > 0 else 180.0)) % 360.0)
    )
    return MajorAxisOrientation(
        centroid_x=float(centroid[0]),
        centroid_y=float(centroid[1]),
        major_axis_center_x=float(axis_center[0]),
        major_axis_center_y=float(axis_center[1]),
        orientation_degrees=orientation,
        direction_degrees=direction,
    )


def _representative_masks() -> list[np.ndarray]:
    asymmetric = np.zeros((11, 31), dtype=np.uint8)
    cv2.ellipse(asymmetric, (13, 5), (11, 3), 17, 0, 360, 255, -1)
    asymmetric[2:5, 23:27] = 255

    disconnected = np.zeros((80, 120), dtype=np.uint8)
    cv2.ellipse(disconnected, (28, 28), (22, 6), 32, 0, 360, 255, -1)
    disconnected[65:69, 100:109] = 255

    nested_island = np.zeros((101, 101), dtype=np.uint8)
    cv2.ellipse(nested_island, (50, 50), (43, 24), 21, 0, 360, 255, 4)
    cv2.ellipse(nested_island, (55, 49), (7, 3), 21, 0, 360, 255, -1)

    diagonal = np.zeros((64, 64), dtype=np.uint8)
    cv2.line(diagonal, (3, 9), (58, 51), 255, 3)
    diagonal[48:55, 16:21] = 255

    steep_negative = np.zeros((97, 31), dtype=np.uint8)
    cv2.line(steep_negative, (22, 3), (8, 92), 255, 2)
    steep_negative[70:75, 21:24] = 255
    return [asymmetric, disconnected, nested_island, diagonal, steep_negative]


@pytest.mark.parametrize("mask", _representative_masks())
def test_moment_orientation_matches_coordinate_reference(mask: np.ndarray):
    expected = _coordinate_reference(mask)
    actual = component_major_axis_orientation(mask)

    assert actual.centroid_x == pytest.approx(expected.centroid_x, rel=0, abs=1e-10)
    assert actual.centroid_y == pytest.approx(expected.centroid_y, rel=0, abs=1e-10)
    assert actual.major_axis_center_x == pytest.approx(
        expected.major_axis_center_x, rel=0, abs=1e-9
    )
    assert actual.major_axis_center_y == pytest.approx(
        expected.major_axis_center_y, rel=0, abs=1e-9
    )
    assert actual.orientation_degrees == pytest.approx(
        expected.orientation_degrees, rel=0, abs=1e-10
    )
    assert actual.direction_degrees == pytest.approx(
        expected.direction_degrees, rel=0, abs=1e-10
    )


def test_moment_orientation_preserves_degenerate_mask_behavior():
    with pytest.raises(ValueError, match="at least one foreground pixel"):
        component_major_axis_orientation(np.zeros((3, 4), dtype=np.uint8))

    singleton = component_major_axis_orientation(np.array([[0, 255]], dtype=np.uint8))
    assert singleton == MajorAxisOrientation(1.0, 0.0, None, None, None, None)

    isotropic = component_major_axis_orientation(np.ones((3, 3), dtype=np.uint8))
    assert isotropic == MajorAxisOrientation(1.0, 1.0, None, None, None, None)
