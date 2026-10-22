"""Gradient-aware seeded expansion for candidate ROI masks.

The v2 heuristic evaluates a nested sequence of intensity-connected masks.
Gradients are evidence for selecting one of those masks; they are deliberately
not barriers to growth.  This lets a candidate cross an internal contrast edge
when a more defensible outer boundary exists at a lower intensity threshold.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import cv2
import numpy as np

from .detection_refinement import RoiRefinementInput, RoiRefinementPrediction
from .frame_preprocess import as_binary_mask, as_grayscale_array


HEURISTIC_V2_METHOD_NAME = "heuristic_v2"


@dataclass(frozen=True, slots=True)
class HeuristicV2RefinementParameters:
    """Versioned controls for gradient-aware seeded region expansion."""

    max_growth_pixels: int = 32
    threshold_step: float = 4.0
    max_threshold_steps: int = 64
    inferred_seed_percentile: float = 10.0
    background_percentile: float = 50.0
    noise_floor_mad_multiplier: float = 3.0
    minimum_background_margin: float = 2.0
    minimum_foreground_intensity: float | None = None
    minimum_boundary_drop: float = 1.0
    minimum_boundary_coverage: float = 0.55
    stable_area_growth_ratio: float = 0.35
    boundary_weight: float = 1.0
    stability_weight: float = 0.25
    recovery_weight: float = 0.20
    instability_penalty_weight: float = 0.50
    diffuse_penalty_weight: float = 0.35
    minimum_score_improvement: float = 0.02
    comparable_score_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if int(self.max_growth_pixels) < 0:
            raise ValueError("max_growth_pixels must be >= 0.")
        if float(self.threshold_step) <= 0:
            raise ValueError("threshold_step must be > 0.")
        if int(self.max_threshold_steps) < 1:
            raise ValueError("max_threshold_steps must be >= 1.")
        for name in ("inferred_seed_percentile", "background_percentile"):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 100.0:
                raise ValueError(f"{name} must be between 0 and 100.")
        for name in (
            "noise_floor_mad_multiplier",
            "minimum_background_margin",
            "minimum_boundary_drop",
            "stable_area_growth_ratio",
            "boundary_weight",
            "stability_weight",
            "recovery_weight",
            "instability_penalty_weight",
            "diffuse_penalty_weight",
            "minimum_score_improvement",
            "comparable_score_tolerance",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0.")
        if not 0.0 <= float(self.minimum_boundary_coverage) <= 1.0:
            raise ValueError("minimum_boundary_coverage must be between 0 and 1.")


class HeuristicV2RefinementBackend:
    """CPU-only whole-crop backend for the v2 heuristic."""

    method_name = HEURISTIC_V2_METHOD_NAME
    protects_candidate_identity = True

    def __init__(self, parameters: HeuristicV2RefinementParameters | None = None) -> None:
        self.parameters = parameters or HeuristicV2RefinementParameters()

    def refine_batch(self, inputs: list[RoiRefinementInput]) -> list[RoiRefinementPrediction]:
        return [self.refine(item) for item in inputs]

    def refine(self, item: RoiRefinementInput) -> RoiRefinementPrediction:
        mask, audit = heuristic_v2_refine(
            item.image,
            item.candidate_mask,
            competing_seed_mask=item.competing_seed_mask,
            candidate_generation_threshold=item.candidate_generation_threshold,
            parameters=self.parameters,
        )
        return RoiRefinementPrediction(
            mask=mask,
            metadata={
                "inference_backend": "pelagia_builtin",
                "refinement_algorithm": HEURISTIC_V2_METHOD_NAME,
                "refinement_algorithm_version": 2,
                "refinement_parameters": asdict(self.parameters),
                "heuristic_v2_audit": audit,
                "dominant_edge_angle_degrees": audit["dominant_edge_angle_degrees"],
                "edge_strength": audit["selected_boundary_median_drop"],
            },
        )


def heuristic_v2_refine(
    image: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    competing_seed_mask: np.ndarray | None = None,
    candidate_generation_threshold: float | None = None,
    parameters: HeuristicV2RefinementParameters | None = None,
) -> tuple[np.ndarray, dict[str, float | int | str | None]]:
    """Expand ``candidate_mask`` to the best supported nested threshold mask.

    The seed is immutable.  Growth uses eight-connectivity, stays within
    ``max_growth_pixels`` of the seed, and is restricted to the seed's nearest
    territory when competing seeds are supplied.  A no-expansion baseline is
    evaluated alongside every expansion, so weak evidence retains the input.
    """
    resolved = parameters or HeuristicV2RefinementParameters()
    source_gray = as_grayscale_array(image)
    discrete_intensity = np.issubdtype(source_gray.dtype, np.integer)
    gray = source_gray.astype(np.float32, copy=False)
    seed = as_binary_mask(candidate_mask) > 0
    if gray.shape[:2] != seed.shape:
        raise ValueError("image and candidate_mask must have matching spatial shapes.")

    if competing_seed_mask is None:
        competing = np.zeros(seed.shape, dtype=bool)
    else:
        competing = as_binary_mask(competing_seed_mask) > 0
        if competing.shape != seed.shape:
            raise ValueError(
                "competing_seed_mask and candidate_mask must have matching spatial shapes."
            )
    overlap_pixels = int(np.count_nonzero(seed & competing))
    if overlap_pixels:
        raise ValueError(
            "candidate_mask and competing_seed_mask overlap; immutable seed retention "
            "and candidate identity protection cannot both be satisfied."
        )

    seed_pixels = int(np.count_nonzero(seed))
    empty_audit = _empty_audit(seed_pixels=seed_pixels, overlap_pixels=overlap_pixels)
    if seed_pixels == 0:
        return np.zeros(seed.shape, dtype=np.uint8), empty_audit

    finite = np.isfinite(gray)
    if candidate_generation_threshold is None:
        finite_seed_values = gray[seed & finite]
        if finite_seed_values.size:
            starting_threshold = float(
                np.percentile(finite_seed_values, resolved.inferred_seed_percentile)
            )
        else:
            starting_threshold = 0.0
        threshold_source = "seed_percentile"
    else:
        starting_threshold = float(candidate_generation_threshold)
        if not np.isfinite(starting_threshold):
            raise ValueError("candidate_generation_threshold must be finite when provided.")
        threshold_source = "candidate_generation"

    if int(resolved.max_growth_pixels) == 0:
        audit = dict(empty_audit)
        audit.update(
            {
                "starting_threshold": starting_threshold,
                "threshold_source": threshold_source,
                "selected_threshold": starting_threshold,
            }
        )
        return _mask_uint8(seed), audit

    seed_distance = cv2.distanceTransform((~seed).astype(np.uint8), cv2.DIST_C, 3)
    permitted = seed_distance <= float(int(resolved.max_growth_pixels))
    permitted &= finite
    permitted &= ~competing
    permitted |= seed

    competition_limited_pixels = 0
    if np.any(competing):
        competing_distance = cv2.distanceTransform(
            (~competing).astype(np.uint8), cv2.DIST_C, 3
        )
        own_territory = seed_distance < competing_distance
        own_territory |= seed
        before = int(np.count_nonzero(permitted))
        permitted &= own_territory
        permitted |= seed
        competition_limited_pixels = before - int(np.count_nonzero(permitted))

    background_level, noise_sigma, minimum_threshold = _estimate_noise_floor(
        gray,
        permitted=permitted,
        seed=seed,
        competing=competing,
        starting_threshold=starting_threshold,
        parameters=resolved,
    )
    thresholds = _threshold_sequence(
        starting_threshold,
        minimum_threshold,
        step=resolved.threshold_step,
        maximum_steps=int(resolved.max_threshold_steps),
        observed_values=gray[permitted & finite] if discrete_intensity else None,
    )
    contrast_scale = max(
        float(noise_sigma),
        float(resolved.minimum_background_margin),
        float(resolved.minimum_boundary_drop),
        np.finfo(np.float32).eps,
    )

    baseline_metrics = _boundary_metrics(
        gray,
        seed,
        minimum_drop=max(resolved.minimum_boundary_drop, 0.5 * noise_sigma),
        contrast_scale=contrast_scale,
    )
    # The no-growth baseline has no observed threshold-to-threshold stability,
    # so it receives boundary evidence only.  Otherwise its synthetic perfect
    # stability would systematically prevent recovery of a dim outer shell.
    baseline_score = resolved.boundary_weight * baseline_metrics["score"]
    selected = seed.copy()
    selected_area = seed_pixels
    selected_threshold = starting_threshold
    selected_score = float(baseline_score)
    best_supported_score = float(baseline_score)
    selected_metrics = baseline_metrics
    selected_step_growth = 0.0
    selected_stability = 1.0
    selected_recovered = 0
    selected_diffuse_fraction = 0.0

    previous_area = seed_pixels
    thresholds_evaluated = 0
    valid_expansions = 0
    minimum_drop = max(resolved.minimum_boundary_drop, 0.5 * noise_sigma)
    diffuse_cutoff = background_level + max(
        resolved.minimum_background_margin, 2.0 * noise_sigma
    )

    for threshold in thresholds:
        expanded = _seed_connected_mask(
            seed,
            permitted=permitted,
            eligible=finite & (gray >= threshold),
        )
        area = int(np.count_nonzero(expanded))
        thresholds_evaluated += 1
        if area <= seed_pixels:
            previous_area = area
            continue

        metrics = _boundary_metrics(
            gray,
            expanded,
            minimum_drop=minimum_drop,
            contrast_scale=contrast_scale,
        )
        step_growth = max(0.0, float(area - previous_area) / float(max(previous_area, 1)))
        stability = 1.0 / (
            1.0 + step_growth / max(resolved.stable_area_growth_ratio, np.finfo(float).eps)
        )
        instability = max(
            0.0,
            step_growth - resolved.stable_area_growth_ratio,
        ) / max(resolved.stable_area_growth_ratio, np.finfo(float).eps)
        instability = min(1.0, instability)

        added = expanded & ~seed
        recovered = area - seed_pixels
        recovered_below_start = int(np.count_nonzero(added & (gray < starting_threshold)))
        recovery_extent = min(1.0, float(recovered) / float(max(seed_pixels, 1)))
        lower_intensity_fraction = float(recovered_below_start) / float(max(recovered, 1))
        recovery = recovery_extent * (0.5 + 0.5 * lower_intensity_fraction)
        diffuse_fraction = float(np.mean(gray[added] <= diffuse_cutoff))
        diffuse_penalty = diffuse_fraction * recovery_extent
        # A coherent outer boundary can justify a large single threshold step;
        # the same jump into weakly bounded background receives the full
        # instability penalty.
        weak_boundary_fraction = 1.0 - min(1.0, float(metrics["score"]))
        score = (
            resolved.boundary_weight * metrics["score"]
            + resolved.stability_weight * stability
            + resolved.recovery_weight * recovery
            - resolved.instability_penalty_weight * instability * weak_boundary_fraction
            - resolved.diffuse_penalty_weight * diffuse_penalty
        )

        boundary_is_valid = (
            metrics["coverage"] >= resolved.minimum_boundary_coverage
            and metrics["median_drop"] >= minimum_drop
        )
        if boundary_is_valid:
            valid_expansions += 1
            improves_baseline = score >= baseline_score + resolved.minimum_score_improvement
            is_new_best = score > best_supported_score
            comparable_to_best = (
                score >= best_supported_score - resolved.comparable_score_tolerance
            )
            if improves_baseline and (is_new_best or (comparable_to_best and area > selected_area)):
                selected = expanded
                selected_area = area
                selected_threshold = float(threshold)
                selected_score = float(score)
                selected_metrics = metrics
                selected_step_growth = step_growth
                selected_stability = stability
                selected_recovered = recovered_below_start
                selected_diffuse_fraction = diffuse_fraction
            if improves_baseline:
                best_supported_score = max(best_supported_score, float(score))
        previous_area = area

    # Restate both invariants after selection so future scoring changes cannot
    # accidentally contract the seed or take another candidate's pixels.
    selected |= seed
    selected &= ~competing
    selected |= seed
    refined_pixels = int(np.count_nonzero(selected))
    return _mask_uint8(selected), {
        "seed_foreground_pixels": seed_pixels,
        "refined_foreground_pixels": refined_pixels,
        "expanded_foreground_pixels": refined_pixels - seed_pixels,
        "recovered_below_start_pixels": selected_recovered,
        "starting_threshold": starting_threshold,
        "threshold_source": threshold_source,
        "minimum_threshold": float(minimum_threshold),
        "selected_threshold": float(selected_threshold),
        "thresholds_evaluated": thresholds_evaluated,
        "valid_expansions": valid_expansions,
        "baseline_score": float(baseline_score),
        "selected_score": float(selected_score),
        "selected_boundary_median_drop": float(selected_metrics["median_drop"]),
        "selected_boundary_lower_quartile_drop": float(selected_metrics["lower_quartile_drop"]),
        "selected_boundary_coverage": float(selected_metrics["coverage"]),
        "selected_step_growth_ratio": float(selected_step_growth),
        "selected_stability": float(selected_stability),
        "selected_diffuse_fraction": float(selected_diffuse_fraction),
        "background_level": float(background_level),
        "background_noise_sigma": float(noise_sigma),
        "competing_seed_pixels": int(np.count_nonzero(competing)),
        "competing_seed_overlap_pixels": overlap_pixels,
        "competition_limited_pixels": competition_limited_pixels,
        "dominant_edge_angle_degrees": selected_metrics["dominant_angle"],
    }


def _estimate_noise_floor(
    gray: np.ndarray,
    *,
    permitted: np.ndarray,
    seed: np.ndarray,
    competing: np.ndarray,
    starting_threshold: float,
    parameters: HeuristicV2RefinementParameters,
) -> tuple[float, float, float]:
    sample = gray[permitted & ~seed & ~competing & np.isfinite(gray)]
    if sample.size == 0:
        sample = gray[np.isfinite(gray) & ~seed]
    if sample.size == 0:
        background_level = float(starting_threshold)
        noise_sigma = 0.0
    else:
        upper = float(np.percentile(sample, parameters.background_percentile))
        lower_sample = sample[sample <= upper]
        if lower_sample.size == 0:
            lower_sample = sample
        background_level = float(np.median(lower_sample))
        mad = float(np.median(np.abs(lower_sample - background_level)))
        noise_sigma = 1.4826 * mad

    if parameters.minimum_foreground_intensity is None:
        floor = background_level + max(
            parameters.minimum_background_margin,
            parameters.noise_floor_mad_multiplier * noise_sigma,
        )
    else:
        floor = float(parameters.minimum_foreground_intensity)
        if not np.isfinite(floor):
            raise ValueError("minimum_foreground_intensity must be finite when provided.")
    return background_level, noise_sigma, min(float(starting_threshold), float(floor))


def _threshold_sequence(
    start: float,
    stop: float,
    *,
    step: float,
    maximum_steps: int,
    observed_values: np.ndarray | None = None,
) -> list[float]:
    if stop >= start:
        return [float(start)]
    if observed_values is not None:
        transitions = np.unique(observed_values.astype(np.float64, copy=False))
        transitions = transitions[(transitions < start) & (transitions >= stop)]
        if transitions.size + 1 <= maximum_steps:
            # A threshold only changes the reachable mask when it crosses an
            # observed intensity.  Skipping empty intervals is a substantial
            # saving for integer imagery and leaves the evaluated masks exact.
            return [float(start), *[float(value) for value in transitions[::-1]]]
    span = start - stop
    count = min(maximum_steps, max(2, int(np.ceil(span / step)) + 1))
    # linspace reaches the noise floor even when a fixed step would exceed the
    # configured evaluation budget.  Values remain strictly descending.
    values = np.linspace(start, stop, count, dtype=np.float64)
    return [float(value) for value in values]


def _seed_connected_mask(
    seed: np.ndarray,
    *,
    permitted: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    traversable = permitted & eligible
    traversable |= seed
    component_count, labels = cv2.connectedComponents(
        traversable.astype(np.uint8), connectivity=8
    )
    if component_count <= 1:
        return seed.copy()
    seed_labels = np.unique(labels[seed])
    seed_labels = seed_labels[seed_labels > 0]
    if seed_labels.size == 0:
        return seed.copy()
    expanded = np.isin(labels, seed_labels)
    expanded |= seed
    return expanded


def _boundary_metrics(
    gray: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_drop: float,
    contrast_scale: float,
) -> dict[str, float | None]:
    drops: list[np.ndarray] = []
    angles: list[np.ndarray] = []
    height, width = mask.shape
    for dy, dx in (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ):
        inside_y = slice(max(0, -dy), min(height, height - dy))
        inside_x = slice(max(0, -dx), min(width, width - dx))
        outside_y = slice(max(0, dy), min(height, height + dy))
        outside_x = slice(max(0, dx), min(width, width + dx))
        inside_mask = mask[inside_y, inside_x]
        outside_mask = mask[outside_y, outside_x]
        pairs = inside_mask & ~outside_mask
        if not np.any(pairs):
            continue
        pair_drops = gray[inside_y, inside_x][pairs] - gray[outside_y, outside_x][pairs]
        pair_drops = pair_drops[np.isfinite(pair_drops)]
        if pair_drops.size == 0:
            continue
        drops.append(pair_drops.astype(np.float32, copy=False))
        # Edge orientation is perpendicular to the inside-to-outside normal.
        edge_angle = float((np.degrees(np.arctan2(dy, dx)) + 90.0) % 180.0)
        angles.append(np.full(pair_drops.shape, edge_angle, dtype=np.float32))

    if not drops:
        return {
            "median_drop": 0.0,
            "lower_quartile_drop": 0.0,
            "coverage": 0.0,
            "score": 0.0,
            "dominant_angle": None,
        }
    all_drops = np.concatenate(drops)
    all_angles = np.concatenate(angles)
    median_drop = float(np.median(all_drops))
    lower_quartile = float(np.percentile(all_drops, 25.0))
    positive = all_drops >= float(minimum_drop)
    coverage = float(np.mean(positive))
    robust_drop = 0.75 * max(0.0, median_drop) + 0.25 * max(0.0, lower_quartile)
    score = coverage * robust_drop / (robust_drop + contrast_scale)
    dominant_angle = _axial_mean_angle(all_angles[positive], all_drops[positive])
    return {
        "median_drop": median_drop,
        "lower_quartile_drop": lower_quartile,
        "coverage": coverage,
        "score": score,
        "dominant_angle": dominant_angle,
    }


def _axial_mean_angle(angles: np.ndarray, weights: np.ndarray) -> float | None:
    if angles.size == 0:
        return None
    positive_weights = np.maximum(weights.astype(np.float64, copy=False), 0.0)
    weight_sum = float(np.sum(positive_weights))
    if weight_sum <= 0:
        return None
    doubled = np.deg2rad(2.0 * angles.astype(np.float64, copy=False))
    vector = np.sum(positive_weights * np.exp(1j * doubled)) / weight_sum
    if abs(vector) < 0.1:
        return None
    return float((np.degrees(np.angle(vector)) / 2.0) % 180.0)


def _mask_uint8(mask: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mask.astype(np.uint8) * 255)


def _empty_audit(
    *, seed_pixels: int, overlap_pixels: int
) -> dict[str, float | int | str | None]:
    return {
        "seed_foreground_pixels": seed_pixels,
        "refined_foreground_pixels": seed_pixels,
        "expanded_foreground_pixels": 0,
        "recovered_below_start_pixels": 0,
        "starting_threshold": 0.0,
        "threshold_source": "unavailable",
        "minimum_threshold": 0.0,
        "selected_threshold": 0.0,
        "thresholds_evaluated": 0,
        "valid_expansions": 0,
        "baseline_score": 0.0,
        "selected_score": 0.0,
        "selected_boundary_median_drop": 0.0,
        "selected_boundary_lower_quartile_drop": 0.0,
        "selected_boundary_coverage": 0.0,
        "selected_step_growth_ratio": 0.0,
        "selected_stability": 1.0,
        "selected_diffuse_fraction": 0.0,
        "background_level": 0.0,
        "background_noise_sigma": 0.0,
        "competing_seed_pixels": 0,
        "competing_seed_overlap_pixels": overlap_pixels,
        "competition_limited_pixels": 0,
        "dominant_edge_angle_degrees": None,
    }
