"""Seed-constrained gradient graph cut for CPU ROI refinement."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import breadth_first_order, maximum_flow

from .detection_refinement import RoiRefinementInput, RoiRefinementPrediction
from .frame_preprocess import as_binary_mask, as_grayscale_array


HEURISTIC_V3_METHOD_NAME = "heuristic_v3"
_CAPACITY_SCALE = 256
_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))


@dataclass(frozen=True, slots=True)
class HeuristicV3RefinementParameters:
    max_growth_pixels: int = 32
    foreground_fraction: float = 0.10
    intensity_weight: float = 2.0
    boundary_weight: float = 3.0
    growth_penalty: float = 0.20
    gradient_sigma: float | None = None
    minimum_gradient_sigma: float = 8.0
    max_graph_pixels: int = 100_000

    def __post_init__(self) -> None:
        if self.max_growth_pixels < 0:
            raise ValueError("max_growth_pixels must be >= 0.")
        if not 0 < self.foreground_fraction < 1:
            raise ValueError("foreground_fraction must be between 0 and 1.")
        for name in ("intensity_weight", "boundary_weight", "growth_penalty"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0.")
        if self.gradient_sigma is not None and self.gradient_sigma <= 0:
            raise ValueError("gradient_sigma must be > 0 when set.")
        if self.minimum_gradient_sigma <= 0:
            raise ValueError("minimum_gradient_sigma must be > 0.")
        if self.max_graph_pixels < 1:
            raise ValueError("max_graph_pixels must be >= 1.")


class HeuristicV3RefinementBackend:
    method_name = HEURISTIC_V3_METHOD_NAME
    protects_candidate_identity = True

    def __init__(self, parameters: HeuristicV3RefinementParameters | None = None) -> None:
        self.parameters = parameters or HeuristicV3RefinementParameters()

    def refine_batch(self, inputs: list[RoiRefinementInput]) -> list[RoiRefinementPrediction]:
        return [self.refine(item) for item in inputs]

    def refine(self, item: RoiRefinementInput) -> RoiRefinementPrediction:
        mask, audit = heuristic_v3_refine(
            item.image,
            item.candidate_mask,
            competing_seed_mask=item.competing_seed_mask,
            parameters=self.parameters,
        )
        return RoiRefinementPrediction(
            mask=mask,
            metadata={
                "inference_backend": "pelagia_builtin",
                "refinement_algorithm": HEURISTIC_V3_METHOD_NAME,
                "refinement_algorithm_version": 3,
                "refinement_parameters": asdict(self.parameters),
                "heuristic_v3_audit": audit,
            },
        )


def heuristic_v3_refine(
    image: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    competing_seed_mask: np.ndarray | None = None,
    parameters: HeuristicV3RefinementParameters | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str]]:
    """Minimize a binary unary-plus-Potts energy around immutable candidate seeds."""
    params = parameters or HeuristicV3RefinementParameters()
    gray = as_grayscale_array(image).astype(np.float32, copy=False)
    seed = as_binary_mask(candidate_mask) > 0
    if gray.shape != seed.shape:
        raise ValueError("image and candidate_mask must have matching spatial shapes.")
    competing = (
        np.zeros_like(seed) if competing_seed_mask is None
        else as_binary_mask(competing_seed_mask) > 0
    )
    if competing.shape != seed.shape:
        raise ValueError("competing_seed_mask and candidate_mask must have matching shapes.")
    if np.any(seed & competing):
        raise ValueError("candidate_mask and competing_seed_mask overlap; both seed constraints cannot hold.")
    seed_pixels = int(np.count_nonzero(seed))
    if not seed_pixels:
        return np.zeros_like(seed, dtype=np.uint8), {
            "seed_foreground_pixels": 0, "refined_foreground_pixels": 0,
            "expanded_foreground_pixels": 0, "graph_pixels": 0,
            "solver": "scipy_maximum_flow_dinic_v1",
        }
    if params.max_growth_pixels == 0:
        return (seed.astype(np.uint8) * 255), {
            "seed_foreground_pixels": seed_pixels,
            "refined_foreground_pixels": seed_pixels,
            "expanded_foreground_pixels": 0,
            "graph_pixels": seed_pixels,
            "solver": "scipy_maximum_flow_dinic_v1",
        }

    finite = np.isfinite(gray)
    if not np.all(finite[seed]):
        raise ValueError("Candidate seed intensities must be finite.")
    seed_distance = cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_C, 3)
    permitted = (seed_distance <= params.max_growth_pixels) & finite & ~competing
    permitted |= seed
    graph_pixels = int(np.count_nonzero(permitted))
    if graph_pixels > params.max_graph_pixels:
        raise ValueError(
            f"Graph cut requires {graph_pixels} pixels, exceeding max_graph_pixels={params.max_graph_pixels}."
        )
    index = np.full(seed.shape, -1, dtype=np.int32)
    index[permitted] = np.arange(graph_pixels, dtype=np.int32)
    values = gray[permitted].astype(np.float64)
    seed_values = gray[seed].astype(np.float64)
    background_values = gray[finite & ~permitted & ~competing]
    if not background_values.size:
        background_values = gray[finite & ~seed & ~competing]
    background = float(np.median(background_values)) if background_values.size else 0.0
    foreground = float(np.median(seed_values))
    contrast = max(foreground - background, 1.0)
    midpoint = background + params.foreground_fraction * contrast

    # Signed unary evidence is continuous: no intensity is a hard cutoff.
    normalized = (values - midpoint) / contrast
    source_capacity = params.intensity_weight * np.maximum(normalized, 0.0)
    sink_capacity = params.intensity_weight * np.maximum(-normalized, 0.0)
    sink_capacity += params.growth_penalty * (~seed[permitted])

    sigma = params.gradient_sigma
    if sigma is None:
        local_diff = np.abs(np.diff(gray, axis=0)).ravel()
        local_diff = local_diff[np.isfinite(local_diff)]
        sigma = max(
            params.minimum_gradient_sigma,
            float(np.percentile(local_diff, 75)) if local_diff.size else 0.0,
        )
    edge_rows: list[np.ndarray] = []
    edge_cols: list[np.ndarray] = []
    edge_caps: list[np.ndarray] = []
    boundary_penalty = np.zeros(graph_pixels, dtype=np.float64)
    height, width = seed.shape
    for dy, dx in _OFFSETS:
        y0 = max(0, -dy)
        y1 = min(height, height - dy)
        x0 = max(0, -dx)
        x1 = min(width, width - dx)
        left_ids = index[y0:y1, x0:x1].ravel()
        right_ids = index[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        left_values = gray[y0:y1, x0:x1].ravel()
        right_values = gray[y0 + dy:y1 + dy, x0 + dx:x1 + dx].ravel()
        valid = np.isfinite(left_values) & np.isfinite(right_values)
        weights = params.boundary_weight * np.exp(
            -np.square(left_values.astype(np.float64) - right_values.astype(np.float64))
            / (2.0 * sigma * sigma)
        )
        if dy and dx:
            weights /= sqrt(2.0)
        both = valid & (left_ids >= 0) & (right_ids >= 0)
        a = left_ids[both]
        b = right_ids[both]
        caps = np.rint(weights[both] * _CAPACITY_SCALE).astype(np.int64)
        keep = caps > 0
        edge_rows.extend((a[keep], b[keep]))
        edge_cols.extend((b[keep], a[keep]))
        edge_caps.extend((caps[keep], caps[keep]))
        left_only = valid & (left_ids >= 0) & (right_ids < 0)
        right_only = valid & (right_ids >= 0) & (left_ids < 0)
        np.add.at(boundary_penalty, left_ids[left_only], weights[left_only])
        np.add.at(boundary_penalty, right_ids[right_only], weights[right_only])
    sink_capacity += boundary_penalty

    source = graph_pixels
    sink = source + 1
    finite_total = (
        float(np.sum(source_capacity + sink_capacity))
        + sum(float(np.sum(caps)) / _CAPACITY_SCALE for caps in edge_caps)
    )
    hard_capacity = int(np.ceil(finite_total * _CAPACITY_SCALE)) + 1
    source_cap = np.rint(source_capacity * _CAPACITY_SCALE).astype(np.int64)
    sink_cap = np.rint(sink_capacity * _CAPACITY_SCALE).astype(np.int64)
    source_cap[seed[permitted]] = hard_capacity
    rows = edge_rows + [np.full(graph_pixels, source), np.arange(graph_pixels)]
    cols = edge_cols + [np.arange(graph_pixels), np.full(graph_pixels, sink)]
    caps = edge_caps + [source_cap, sink_cap]
    capacity = csr_matrix(
        (np.concatenate(caps), (np.concatenate(rows), np.concatenate(cols))),
        shape=(graph_pixels + 2, graph_pixels + 2), dtype=np.int64,
    )
    flow = maximum_flow(capacity, source, sink)
    residual = capacity - flow.flow
    residual.data = (residual.data > 0).astype(np.int8)
    residual.eliminate_zeros()
    reachable = breadth_first_order(residual, source, directed=True, return_predecessors=False)
    selected = np.zeros(graph_pixels + 2, dtype=bool)
    selected[reachable] = True
    output = np.zeros(seed.shape, dtype=np.uint8)
    output[permitted] = selected[:graph_pixels].astype(np.uint8) * 255

    # A minimum cut may select an isolated bright island; only seed-connected
    # components represent an expansion of this candidate.
    component_count, labels = cv2.connectedComponents(output, connectivity=8)
    seed_labels = np.unique(labels[seed])
    retained = np.isin(labels, seed_labels[seed_labels > 0]) if component_count > 1 else seed
    retained |= seed
    retained &= ~competing
    refined_pixels = int(np.count_nonzero(retained))
    audit: dict[str, float | int | str] = {
        "seed_foreground_pixels": seed_pixels,
        "refined_foreground_pixels": refined_pixels,
        "expanded_foreground_pixels": refined_pixels - seed_pixels,
        "competing_seed_pixels": int(np.count_nonzero(competing)),
        "graph_pixels": graph_pixels,
        "background_level": background,
        "seed_median_intensity": foreground,
        "foreground_midpoint": midpoint,
        "gradient_sigma": float(sigma),
        "cut_capacity": int(flow.flow_value),
        "solver": "scipy_maximum_flow_dinic_v1",
    }
    return retained.astype(np.uint8) * 255, audit
