import numpy as np
import pytest

from Pelagia.processing.detection_refinement import RoiRefinementInput
from Pelagia.processing.heuristic_refinement import (
    HEURISTIC_EDGE_METHOD_NAME,
    HeuristicEdgeRefinementBackend,
    HeuristicEdgeRefinementParameters,
    heuristic_edge_refine,
)


def test_heuristic_refiner_grows_a_similar_seed_on_a_uniform_crop():
    image = np.full((21, 21), 100, dtype=np.uint8)
    seed = np.zeros_like(image)
    seed[9:12, 9:12] = 255

    mask, audit = heuristic_edge_refine(
        image,
        seed,
        parameters=HeuristicEdgeRefinementParameters(max_growth_pixels=3),
    )

    assert np.count_nonzero(mask) > np.count_nonzero(seed)
    assert np.all(mask[6:15, 6:15] == 255)
    assert audit["seed_foreground_pixels"] == 9
    assert audit["refined_foreground_pixels"] == 81


def test_heuristic_refiner_ignores_a_strong_vertical_line_scan_edge():
    image = np.full((21, 21), 100, dtype=np.uint8)
    image[:, 11:] = 160  # Strong vertical edge: deliberately not a barrier.
    seed = np.zeros_like(image)
    seed[9:12, 8:11] = 255

    mask, audit = heuristic_edge_refine(
        image,
        seed,
        parameters=HeuristicEdgeRefinementParameters(
            max_growth_pixels=3,
            minimum_intensity_tolerance=100,
            gradient_percentile=80,
            axis_exclusion_degrees=10,
        ),
    )

    assert np.any(mask[:, 12:] > 0)
    assert audit["barrier_pixels"] == 0


def test_heuristic_refiner_does_not_cross_a_strong_diagonal_edge():
    yy, xx = np.indices((31, 31))
    image = np.where(xx > yy, 160, 100).astype(np.uint8)
    seed = np.zeros_like(image)
    seed[18:21, 7:10] = 255

    mask, audit = heuristic_edge_refine(
        image,
        seed,
        parameters=HeuristicEdgeRefinementParameters(
            max_growth_pixels=8,
            minimum_intensity_tolerance=100,
            gradient_percentile=75,
            axis_exclusion_degrees=10,
        ),
    )

    assert audit["barrier_pixels"] > 0
    # The seed is below the diagonal; no retained foreground crosses to x > y.
    assert not np.any(mask[(xx > yy) & (yy >= 10) & (yy <= 25)] > 0)


def test_backend_reports_versioned_parameters_and_audit_metadata():
    backend = HeuristicEdgeRefinementBackend(
        HeuristicEdgeRefinementParameters(max_growth_pixels=2, axis_exclusion_degrees=8)
    )
    image = np.full((9, 9), 80, dtype=np.uint8)
    seed = np.zeros_like(image)
    seed[4, 4] = 255

    prediction = backend.refine_batch(
        [RoiRefinementInput(detection_id="d-1", image=image, candidate_mask=seed)]
    )[0]

    assert backend.method_name == HEURISTIC_EDGE_METHOD_NAME
    assert prediction.metadata["inference_backend"] == "pelagia_builtin"
    assert prediction.metadata["refinement_algorithm"] == HEURISTIC_EDGE_METHOD_NAME
    assert prediction.metadata["refinement_parameters"]["max_growth_pixels"] == 2
    assert prediction.metadata["heuristic_edge_audit"]["seed_foreground_pixels"] == 1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"axis_exclusion_degrees": 45}, "axis_exclusion_degrees"),
        ({"max_growth_pixels": -1}, "max_growth_pixels"),
    ],
)
def test_heuristic_parameters_reject_invalid_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        HeuristicEdgeRefinementParameters(**kwargs)
