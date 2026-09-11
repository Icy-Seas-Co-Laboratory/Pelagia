"""Deterministic, edge-constrained ROI refinement for line-scan imagery.

This module deliberately has no model-serving dependency.  It retains the
candidate mask as a seed and grows it only through locally similar pixels.  A
strong, oblique image gradient is treated as a boundary barrier; gradients
whose edge orientation is horizontal or vertical are excluded because those
orientations commonly originate from line-scan acquisition artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from .detection_refinement import RoiRefinementInput, RoiRefinementPrediction
from .frame_preprocess import as_binary_mask, as_grayscale_array


HEURISTIC_EDGE_METHOD_NAME = "heuristic_edge_v1"


@dataclass(frozen=True, slots=True)
class HeuristicEdgeRefinementParameters:
    """Versioned controls for :class:`HeuristicEdgeRefinementBackend`."""

    gradient_percentile: float = 90.0
    axis_exclusion_degrees: float = 12.0
    max_growth_pixels: int = 32
    intensity_mad_multiplier: float = 3.0
    minimum_intensity_tolerance: float = 12.0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.gradient_percentile) <= 100.0:
            raise ValueError("gradient_percentile must be between 0 and 100.")
        if not 0.0 <= float(self.axis_exclusion_degrees) < 45.0:
            raise ValueError("axis_exclusion_degrees must be >= 0 and < 45.")
        if int(self.max_growth_pixels) < 0:
            raise ValueError("max_growth_pixels must be >= 0.")
        if float(self.intensity_mad_multiplier) < 0:
            raise ValueError("intensity_mad_multiplier must be >= 0.")
        if float(self.minimum_intensity_tolerance) < 0:
            raise ValueError("minimum_intensity_tolerance must be >= 0.")


class HeuristicEdgeRefinementBackend:
    """CPU-only whole-crop refiner compatible with ``RoiRefinementBackend``."""

    method_name = HEURISTIC_EDGE_METHOD_NAME

    def __init__(self, parameters: HeuristicEdgeRefinementParameters | None = None) -> None:
        self.parameters = parameters or HeuristicEdgeRefinementParameters()

    def refine_batch(self, inputs: list[RoiRefinementInput]) -> list[RoiRefinementPrediction]:
        return [self.refine(item) for item in inputs]

    def refine(self, item: RoiRefinementInput) -> RoiRefinementPrediction:
        mask, audit = heuristic_edge_refine(
            item.image,
            item.candidate_mask,
            parameters=self.parameters,
        )
        return RoiRefinementPrediction(
            mask=mask,
            metadata={
                "inference_backend": "pelagia_builtin",
                "refinement_algorithm": HEURISTIC_EDGE_METHOD_NAME,
                "refinement_algorithm_version": 1,
                "refinement_parameters": asdict(self.parameters),
                "heuristic_edge_audit": audit,
                "dominant_edge_angle_degrees": audit["dominant_edge_angle_degrees"],
                "edge_strength": audit["edge_strength"],
            },
        )


def heuristic_edge_refine(
    image: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    parameters: HeuristicEdgeRefinementParameters | None = None,
) -> tuple[np.ndarray, dict[str, float | int | None]]:
    """Return a binary mask grown from a candidate seed without crossing oblique edges.

    Axis-aligned image edges are *not* barriers.  This is intentional: they
    are commonly produced by line-scan seams and should not determine a ROI
    boundary.  The returned audit contains only resolved numeric evidence so
    job metadata remains compact and serializable.
    """
    resolved = parameters or HeuristicEdgeRefinementParameters()
    gray = as_grayscale_array(image).astype(np.float32, copy=False)
    seed = as_binary_mask(candidate_mask) > 0
    if gray.shape[:2] != seed.shape:
        raise ValueError("image and candidate_mask must have matching spatial shapes.")
    if not np.any(seed) or resolved.max_growth_pixels == 0:
        return np.ascontiguousarray(seed.astype(np.uint8) * 255), {
            "seed_foreground_pixels": int(np.count_nonzero(seed)),
            "refined_foreground_pixels": int(np.count_nonzero(seed)),
            "barrier_pixels": 0,
            "strong_gradient_threshold": 0.0,
            "dominant_edge_angle_degrees": None,
            "edge_strength": 0.0,
        }

    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    grad_x = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    strong_threshold = float(np.percentile(magnitude, resolved.gradient_percentile))
    # Gradient direction is normal to the edge.  Near either axis is equivalent
    # for a 90-degree rotation, so the same exclusion protects both literal
    # horizontal and literal vertical edges.
    orientation = np.mod(np.degrees(np.arctan2(grad_y, grad_x)), 90.0)
    axis_distance = np.minimum(orientation, 90.0 - orientation)
    oblique = axis_distance >= resolved.axis_exclusion_degrees
    barriers = (magnitude >= strong_threshold) & oblique & (magnitude > 0)

    seed_values = gray[seed]
    seed_median = float(np.median(seed_values))
    mad = float(np.median(np.abs(seed_values - seed_median)))
    tolerance = max(
        float(resolved.minimum_intensity_tolerance),
        float(resolved.intensity_mad_multiplier) * 1.4826 * mad,
    )
    similar_intensity = np.abs(gray - seed_median) <= tolerance
    kernel_size = 2 * int(resolved.max_growth_pixels) + 1
    permitted_extent = cv2.dilate(
        seed.astype(np.uint8),
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
    ).astype(bool)
    traversable = permitted_extent & similar_intensity & ~barriers
    traversable |= seed
    component_count, labels = cv2.connectedComponents(traversable.astype(np.uint8), connectivity=8)
    seed_labels = np.unique(labels[seed])
    keep_labels = seed_labels[seed_labels > 0]
    refined = np.isin(labels, keep_labels) if component_count > 1 else seed
    refined |= seed
    barrier_orientations = np.mod(np.degrees(np.arctan2(grad_y[barriers], grad_x[barriers])), 180.0)
    return np.ascontiguousarray(refined.astype(np.uint8) * 255), {
        "seed_foreground_pixels": int(np.count_nonzero(seed)),
        "refined_foreground_pixels": int(np.count_nonzero(refined)),
        "barrier_pixels": int(np.count_nonzero(barriers)),
        "strong_gradient_threshold": strong_threshold,
        "seed_median_intensity": seed_median,
        "intensity_tolerance": float(tolerance),
        "dominant_edge_angle_degrees": (
            None if barrier_orientations.size == 0 else float(np.median(barrier_orientations))
        ),
        "edge_strength": (
            0.0 if not np.any(barriers) else float(np.mean(magnitude[barriers]))
        ),
    }
