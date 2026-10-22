#!/usr/bin/env python3
"""Compare the optimized ROI-orientation measurement with its prior implementation."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pelagia.processing.detection_recording import (
    MajorAxisOrientation,
    component_major_axis_orientation,
)
from Pelagia.processing.mask_augmentation import as_binary_mask


def coordinate_array_orientation(mask: np.ndarray) -> MajorAxisOrientation:
    """Run the coordinate-array operations replaced by the optimized implementation."""
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
    if eigenvalues[-1] <= 0 or np.isclose(eigenvalues[-1], eigenvalues[0], rtol=1e-6, atol=1e-12):
        return MajorAxisOrientation(float(centroid[0]), float(centroid[1]), None, None, None, None)
    vector = np.asarray(eigenvectors[:, -1], dtype=np.float64)
    if vector[0] < 0 or (np.isclose(vector[0], 0.0) and vector[1] < 0):
        vector *= -1
    projections = points @ vector
    center = vector * ((float(np.min(projections)) + float(np.max(projections))) / 2.0)
    orientation = float(np.degrees(np.arctan2(vector[1], vector[0])) % 180.0)
    centroid_offset = float(np.dot(centroid - center, vector))
    direction = (
        None
        if np.isclose(centroid_offset, 0.0, atol=1e-9)
        else float((orientation + (0.0 if centroid_offset > 0 else 180.0)) % 360.0)
    )
    return MajorAxisOrientation(
        float(centroid[0]),
        float(centroid[1]),
        float(center[0]),
        float(center[1]),
        orientation,
        direction,
    )


def masks() -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    small = np.zeros((32, 48), dtype=np.uint8)
    cv2.ellipse(small, (23, 15), (15, 4), 27, 0, 360, 255, -1)
    result["small_candidate_bbox"] = small

    filled = np.zeros((768, 1024), dtype=np.uint8)
    cv2.ellipse(filled, (510, 380), (430, 180), 24, 0, 360, 255, -1)
    filled[140:190, 780:890] = 255
    result["large_filled"] = filled

    sparse = np.zeros_like(filled)
    for offset in range(0, 700, 35):
        cv2.ellipse(sparse, (80 + offset, 90 + offset // 2), (18, 4), 31, 0, 360, 255, -1)
    result["sparse_disconnected"] = sparse

    noisy = np.zeros_like(filled)
    grid_y, grid_x = np.mgrid[2:766:8, 2:1022:8]
    noisy[grid_y.ravel()[:6550], grid_x.ravel()[:6550]] = 255
    result["noisy_6550_components"] = noisy

    ring = np.zeros_like(filled)
    cv2.ellipse(ring, (510, 380), (430, 180), 24, 0, 360, 255, 5)
    cv2.ellipse(ring, (530, 380), (60, 20), 24, 0, 360, 255, -1)
    result["ring_with_island"] = ring
    return result


def median_ms(function, mask: np.ndarray, iterations: int) -> float:
    for _ in range(5):
        function(mask)
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        function(mask)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    return statistics.median(samples)


def peak_python_bytes(function, mask: np.ndarray) -> int:
    tracemalloc.start()
    function(mask)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    cv2.setNumThreads(1)
    report = {}
    for name, mask in masks().items():
        before_ms = median_ms(coordinate_array_orientation, mask, args.iterations)
        after_ms = median_ms(component_major_axis_orientation, mask, args.iterations)
        report[name] = {
            "shape": list(mask.shape),
            "foreground_pixels": int(np.count_nonzero(mask)),
            "coordinate_array_median_ms": round(before_ms, 4),
            "moments_row_extrema_median_ms": round(after_ms, 4),
            "speedup": round(before_ms / after_ms, 2),
            "coordinate_array_peak_bytes": peak_python_bytes(coordinate_array_orientation, mask),
            "moments_row_extrema_peak_bytes": peak_python_bytes(
                component_major_axis_orientation, mask
            ),
        }
    print(
        json.dumps(
            {
                "environment": {
                    "python": platform.python_version(),
                    "numpy": np.__version__,
                    "opencv": cv2.__version__,
                    "opencv_threads": 1,
                },
                "iterations": args.iterations,
                "scenarios": report,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
