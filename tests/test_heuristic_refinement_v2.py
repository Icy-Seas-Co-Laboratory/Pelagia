import numpy as np

from Pelagia.processing.heuristic_refinement_v2 import (
    HeuristicV2RefinementParameters,
    heuristic_v2_refine,
)


def _parameters(**overrides):
    values = {
        "max_growth_pixels": 10,
        "threshold_step": 2,
        "minimum_foreground_intensity": 5,
        "minimum_background_margin": 1,
        "minimum_boundary_coverage": 0.4,
        "minimum_score_improvement": 0,
    }
    values.update(overrides)
    return HeuristicV2RefinementParameters(**values)


def test_v2_expands_across_an_internal_gradient_without_losing_seed_pixels():
    image = np.full((31, 31), 10, dtype=np.uint8)
    image[7:24, 7:24] = 45
    image[10:21, 10:21] = 90
    seed = np.zeros_like(image)
    seed[12:19, 12:19] = 255

    refined, audit = heuristic_v2_refine(
        image,
        seed,
        candidate_generation_threshold=80,
        parameters=_parameters(),
    )

    assert np.all(refined[seed > 0] == 255)
    assert np.all(refined[7:24, 7:24] == 255)
    assert audit["selected_threshold"] == 45.0
    assert audit["recovered_below_start_pixels"] > 0
    assert audit["refined_foreground_pixels"] > audit["seed_foreground_pixels"]


def test_v2_retains_seed_when_diffuse_background_has_no_supported_boundary():
    image = np.full((31, 31), 50, dtype=np.uint8)
    image[12:19, 12:19] = 100
    seed = np.zeros_like(image)
    seed[12:19, 12:19] = 255

    refined, audit = heuristic_v2_refine(
        image,
        seed,
        candidate_generation_threshold=80,
        parameters=_parameters(),
    )

    np.testing.assert_array_equal(refined, seed)
    assert audit["expanded_foreground_pixels"] == 0
    assert audit["valid_expansions"] == 0


def test_v2_never_absorbs_a_competing_candidate_seed():
    image = np.full((41, 41), 10, dtype=np.uint8)
    image[10:31, 7:34] = 60
    image[15:26, 8:17] = 120
    image[15:26, 24:33] = 110
    seed = np.zeros_like(image)
    seed[18:23, 10:15] = 255
    competitor = np.zeros_like(image)
    competitor[18:23, 26:31] = 255

    refined, audit = heuristic_v2_refine(
        image,
        seed,
        competing_seed_mask=competitor,
        candidate_generation_threshold=100,
        parameters=_parameters(max_growth_pixels=15),
    )

    assert np.all(refined[seed > 0] == 255)
    assert not np.any(refined[competitor > 0])
    assert audit["competing_seed_pixels"] == np.count_nonzero(competitor)
    assert audit["competition_limited_pixels"] > 0


def test_v2_uses_nested_thresholds_and_prefers_larger_comparable_boundary():
    image = np.full((35, 35), 10, dtype=np.uint8)
    image[7:28, 7:28] = 50
    image[10:25, 10:25] = 70
    image[13:22, 13:22] = 100
    seed = np.zeros_like(image)
    seed[15:20, 15:20] = 255

    refined, audit = heuristic_v2_refine(
        image,
        seed,
        candidate_generation_threshold=90,
        # The outer boundary has slightly lower contrast but remains within
        # this configured evidence tolerance, so the larger valid mask wins.
        parameters=_parameters(comparable_score_tolerance=0.2),
    )

    assert audit["thresholds_evaluated"] >= 3
    assert audit["valid_expansions"] >= 2
    assert audit["selected_threshold"] == 50.0
    assert np.count_nonzero(refined) == 21 * 21


def test_v2_is_deterministic_for_same_image_and_inputs():
    image = np.full((29, 29), 8, dtype=np.uint8)
    image[6:23, 6:23] = 40
    image[10:19, 10:19] = 95
    seed = np.zeros_like(image)
    seed[12:17, 12:17] = 255
    parameters = _parameters()

    first_mask, first_audit = heuristic_v2_refine(
        image, seed, candidate_generation_threshold=80, parameters=parameters
    )
    second_mask, second_audit = heuristic_v2_refine(
        image, seed, candidate_generation_threshold=80, parameters=parameters
    )

    np.testing.assert_array_equal(first_mask, second_mask)
    assert first_audit == second_audit


def test_v2_zero_growth_fills_ordinary_hole_and_preserves_competing_hole():
    image = np.full((9, 9), 100, dtype=np.uint8)
    ring = np.zeros_like(image)
    ring[1:8, 1] = ring[1:8, 7] = 255
    ring[1, 1:8] = ring[7, 1:8] = 255
    competitor = np.zeros_like(image)
    competitor[4, 4] = 255
    parameters = _parameters(max_growth_pixels=0)

    ordinary, ordinary_audit = heuristic_v2_refine(image, ring, parameters=parameters)
    protected, protected_audit = heuristic_v2_refine(
        image, ring, competing_seed_mask=competitor, parameters=parameters
    )

    assert np.all(ordinary[1:8, 1:8] == 255)
    assert ordinary_audit["hole_fill_method"] == "padded_border_flood_fill_v1"
    assert ordinary_audit["final_mask_validation"] == "passed"
    assert protected[4, 4] == 0
    assert not np.any(protected[2:7, 2:7])
    assert protected_audit["protected_holes"] == 1
