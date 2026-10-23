import numpy as np
import pytest

from Pelagia.processing.heuristic_refinement_v3 import (
    HeuristicV3RefinementBackend,
    HeuristicV3RefinementParameters,
    heuristic_v3_refine,
)
from Pelagia.processing.detection_refinement import RoiRefinementInput
from Pelagia.processing.heuristic_refinement_v2 import heuristic_v2_refine


def test_v3_recovers_locally_dim_tissue_across_internal_gradient():
    yy, xx = np.indices((64, 64))
    truth = (xx - 32) ** 2 / 17**2 + (yy - 32) ** 2 / 13**2 <= 1
    image = np.zeros((64, 64), dtype=np.uint8)
    image[truth & (yy < 32)] = 55
    image[truth & (yy >= 32)] = 105
    seed = ((xx - 32) ** 2 / 7**2 + (yy - 32) ** 2 / 5**2 <= 1)
    image[seed] = 155

    refined, audit = heuristic_v3_refine(image, seed)

    assert np.all(refined[seed] == 255)
    assert np.count_nonzero(refined) == np.count_nonzero(truth)
    assert np.all(refined[truth] == 255)
    assert audit["expanded_foreground_pixels"] > 0
    assert audit["solver"] == "scipy_maximum_flow_dinic_v1"


def test_v3_recovers_heterogeneous_boundary_beyond_single_threshold():
    yy, xx = np.indices((128, 128))
    truth = (xx - 64) ** 2 / 25**2 + (yy - 64) ** 2 / 19**2 <= 1
    diffuse = (xx - 64) ** 2 / 33**2 + (yy - 64) ** 2 / 25**2 <= 1
    seed = (xx - 64) ** 2 / 11**2 + (yy - 64) ** 2 / 8**2 <= 1
    image = np.zeros((128, 128), dtype=np.uint8)
    image[diffuse & (yy >= 64)] = 30
    image[truth & (yy < 64)] = 35
    image[truth & (yy >= 64)] = 100
    image[seed] = 155

    v2, _ = heuristic_v2_refine(image, seed)
    v3, _ = heuristic_v3_refine(image, seed)

    assert np.count_nonzero(v3 ^ (truth.astype(np.uint8) * 255)) == 0
    assert np.count_nonzero(v2 ^ (truth.astype(np.uint8) * 255)) > 0


def test_v3_preserves_seed_and_excludes_competing_seed():
    image = np.full((25, 25), 100, dtype=np.uint8)
    seed = np.zeros_like(image)
    seed[12, 7] = 255
    competing = np.zeros_like(image)
    competing[12, 17] = 255

    refined, audit = heuristic_v3_refine(image, seed, competing_seed_mask=competing)

    assert refined[12, 7] == 255
    assert refined[12, 17] == 0
    assert audit["competing_seed_pixels"] == 1


def test_v3_discards_disconnected_bright_islands():
    image = np.zeros((31, 31), dtype=np.uint8)
    image[4:10, 4:10] = 120
    image[22:27, 22:27] = 200
    seed = np.zeros_like(image)
    seed[6:8, 6:8] = 255

    refined, _ = heuristic_v3_refine(image, seed)

    assert np.all(refined[seed > 0] == 255)
    assert not np.any(refined[22:27, 22:27])


def test_v3_rejects_incompatible_seeds_and_excessive_graph():
    image = np.full((10, 10), 80, dtype=np.uint8)
    seed = np.zeros_like(image)
    seed[5, 5] = 255
    with pytest.raises(ValueError, match="overlap"):
        heuristic_v3_refine(image, seed, competing_seed_mask=seed)
    with pytest.raises(ValueError, match="max_graph_pixels"):
        heuristic_v3_refine(
            image, seed,
            parameters=HeuristicV3RefinementParameters(max_graph_pixels=10),
        )


def test_v3_backend_records_parameters_and_audit():
    image = np.zeros((9, 9), dtype=np.uint8)
    image[2:7, 2:7] = 90
    seed = np.zeros_like(image)
    seed[4, 4] = 255
    backend = HeuristicV3RefinementBackend()
    prediction = backend.refine_batch([
        RoiRefinementInput(detection_id="candidate-1", image=image, candidate_mask=seed)
    ])[0]

    assert prediction.metadata["refinement_algorithm"] == "heuristic_v3"
    assert prediction.metadata["refinement_algorithm_version"] == 4
    assert prediction.metadata["refinement_parameters"]["growth_penalty"] == 0.2
    assert prediction.metadata["heuristic_v3_audit"]["seed_foreground_pixels"] == 1


def test_v3_zero_growth_fills_ordinary_hole_and_preserves_competing_hole():
    image = np.full((9, 9), 100, dtype=np.uint8)
    ring = np.zeros_like(image)
    ring[1:8, 1] = ring[1:8, 7] = 255
    ring[1, 1:8] = ring[7, 1:8] = 255
    competitor = np.zeros_like(image)
    competitor[4, 4] = 255
    parameters = HeuristicV3RefinementParameters(max_growth_pixels=0)

    ordinary, ordinary_audit = heuristic_v3_refine(image, ring, parameters=parameters)
    protected, protected_audit = heuristic_v3_refine(
        image, ring, competing_seed_mask=competitor, parameters=parameters
    )

    assert np.all(ordinary[1:8, 1:8] == 255)
    assert ordinary_audit["hole_pixels_filled"] == 25
    assert protected[4, 4] == 0
    assert not np.any(protected[2:7, 2:7])
    assert protected_audit["protected_holes"] == 1
    assert protected_audit["final_mask_validation"] == "passed"
