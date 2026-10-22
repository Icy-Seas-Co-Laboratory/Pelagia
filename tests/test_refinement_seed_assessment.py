import numpy as np
import pytest

from Pelagia.domain import DetectionRecord
from Pelagia.processing.detection_refinement import (
    RoiRefinementOptions,
    RoiRefinementPrediction,
    identity_refine_detections,
    refine_detection,
    refine_detections,
)
from Pelagia.processing.heuristic_refinement_v2 import HeuristicV2RefinementBackend
from Pelagia.processing.heuristic_refinement_v3 import HeuristicV3RefinementBackend


FRAME_ID = "00000000-0000-7000-8000-000000000042"


def _detection(
    identifier: str,
    *,
    roi: np.ndarray,
    mask: np.ndarray,
    crop_x: int = 0,
    crop_y: int = 0,
    bbox_x: int = 0,
    bbox_y: int = 0,
    bbox_w: int = 1,
    bbox_h: int = 1,
    roi_index: int = 1,
) -> DetectionRecord:
    return DetectionRecord(
        id=identifier,
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=roi_index,
        bbox_x=bbox_x,
        bbox_y=bbox_y,
        bbox_w=bbox_w,
        bbox_h=bbox_h,
        crop_bbox_x=crop_x,
        crop_bbox_y=crop_y,
        crop_bbox_w=int(roi.shape[1]),
        crop_bbox_h=int(roi.shape[0]),
        area=float(np.count_nonzero(mask)),
        perimeter=4.0,
        major_axis_length=float(max(bbox_w, bbox_h)),
        minor_axis_length=float(min(bbox_w, bbox_h)),
        min_gray_value=float(np.min(roi)),
        mean_gray_value=float(np.mean(roi)),
        roi_payload=np.ascontiguousarray(roi).tobytes(order="C"),
        mask_payload=np.ascontiguousarray(mask).tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(roi.dtype),
        roi_shape=list(roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(mask.dtype),
        mask_shape=list(mask.shape),
    )


class _ExpandingIdentityProtectedBackend:
    method_name = "test:identity-protected"
    protects_candidate_identity = True

    def __init__(self) -> None:
        self.candidate_masks: list[np.ndarray] = []

    def refine_batch(self, inputs):
        predictions = []
        for item in inputs:
            self.candidate_masks.append(item.candidate_mask.copy())
            expanded = item.candidate_mask.copy()
            expanded[1:3, -1] = 255
            predictions.append(RoiRefinementPrediction(mask=expanded))
        return predictions


class _JoinComponentsBackend:
    method_name = "test:join-components"

    def refine_batch(self, inputs):
        predictions = []
        for item in inputs:
            joined = item.candidate_mask.copy()
            joined[2, 1:4] = 255
            predictions.append(RoiRefinementPrediction(mask=joined))
        return predictions


def test_backend_iterations_receive_original_seed_after_crop_expansion():
    roi = np.arange(16, dtype=np.uint8).reshape(4, 4)
    candidate = np.zeros((4, 4), dtype=np.uint8)
    candidate[1:3, 3] = 255
    detection = _detection(
        "det-expand",
        roi=roi,
        mask=candidate,
        crop_x=2,
        crop_y=2,
        bbox_x=5,
        bbox_y=3,
        bbox_w=1,
        bbox_h=2,
    )
    backend = _ExpandingIdentityProtectedBackend()
    frame = np.arange(64, dtype=np.uint8).reshape(8, 8)

    results = refine_detections(
        [detection],
        backend=backend,
        frame_loader=lambda _frame_id: frame,
        options=RoiRefinementOptions(
            max_iterations=2,
            expansion_pixels=1,
            overlap_reconciliation_enabled=False,
        ),
    )

    assert len(backend.candidate_masks) == 2
    np.testing.assert_array_equal(backend.candidate_masks[0], candidate)
    expected_expanded_seed = np.zeros((4, 5), dtype=np.uint8)
    expected_expanded_seed[:, :4] = candidate
    np.testing.assert_array_equal(backend.candidate_masks[1], expected_expanded_seed)
    assert np.count_nonzero(backend.candidate_masks[1]) == np.count_nonzero(candidate)
    assert results[0].metadata["refinement_expansion_count"] == 2


def test_tiled_iterations_receive_original_seed_after_crop_expansion():
    roi = np.arange(16, dtype=np.uint8).reshape(4, 4)
    candidate = np.zeros((4, 4), dtype=np.uint8)
    candidate[1:3, 3] = 255
    detection = _detection(
        "det-tiled", roi=roi, mask=candidate,
        crop_x=2, crop_y=2, bbox_x=5, bbox_y=3, bbox_w=1, bbox_h=2,
    )
    submitted: list[np.ndarray] = []

    def grow_right(_image: np.ndarray, seed: np.ndarray) -> np.ndarray:
        submitted.append(seed.copy())
        output = seed.copy()
        output[1:3, 3 if len(submitted) == 1 else 4] = 255
        return output

    refine_detection(
        detection,
        refiner=grow_right,
        frame_loader=lambda _frame_id: np.arange(64, dtype=np.uint8).reshape(8, 8),
        options=RoiRefinementOptions(max_iterations=2, expansion_pixels=1, tile_size=8),
    )

    assert len(submitted) == 2
    np.testing.assert_array_equal(submitted[0][:4, :4], candidate)
    expected = np.zeros((8, 8), dtype=np.uint8)
    expected[:4, :4] = candidate
    np.testing.assert_array_equal(submitted[1], expected)


def test_identity_refinement_records_split_merge_assessment():
    roi = np.full((5, 5), 20, dtype=np.uint8)
    candidate = np.zeros((5, 5), dtype=np.uint8)
    candidate[2, 1] = 255
    candidate[2, 3] = 255
    detection = _detection(
        "det-identity",
        roi=roi,
        mask=candidate,
        bbox_x=1,
        bbox_y=2,
        bbox_w=3,
    )

    result = identity_refine_detections(
        [detection],
        options=RoiRefinementOptions(overlap_reconciliation_enabled=False),
    )[0]

    assert result.metadata["split_merge_assessment"] == {
        "candidate_component_count": 2,
        "refined_component_count": 2,
        "candidate_pixels": 2,
        "refined_pixels": 2,
        "overlaps": [],
    }


def test_generic_backend_records_split_merge_assessment():
    roi = np.full((5, 5), 20, dtype=np.uint8)
    candidate = np.zeros((5, 5), dtype=np.uint8)
    candidate[2, 1] = 255
    candidate[2, 3] = 255
    detection = _detection(
        "det-generic",
        roi=roi,
        mask=candidate,
        bbox_x=1,
        bbox_y=2,
        bbox_w=3,
    )

    result = refine_detections(
        [detection],
        backend=_JoinComponentsBackend(),
        options=RoiRefinementOptions(overlap_reconciliation_enabled=False),
    )[0]

    assessment = result.metadata["split_merge_assessment"]
    assert assessment["candidate_component_count"] == 2
    assert assessment["refined_component_count"] == 1
    assert assessment["candidate_pixels"] == 2
    assert assessment["refined_pixels"] == 3
    assert assessment["overlaps"] == []


@pytest.mark.parametrize("backend", [HeuristicV2RefinementBackend(), HeuristicV3RefinementBackend()])
def test_seeded_heuristics_disable_destructive_overlap_reconciliation(backend):
    roi = np.full((5, 5), 10, dtype=np.uint8)
    roi[2, 1] = 120
    roi[2, 3] = 120
    left_seed = np.zeros_like(roi)
    left_seed[2, 1] = 255
    right_seed = np.zeros_like(roi)
    right_seed[2, 3] = 255
    detections = [
        _detection(
            "det-left",
            roi=roi,
            mask=left_seed,
            bbox_x=1,
            bbox_y=2,
            roi_index=1,
        ),
        _detection(
            "det-right",
            roi=roi,
            mask=right_seed,
            bbox_x=3,
            bbox_y=2,
            roi_index=2,
        ),
    ]

    results = refine_detections(
        detections,
        backend=backend,
        options=RoiRefinementOptions(overlap_reconciliation_enabled=True),
        all_candidate_seeds=detections,
    )

    assert {result.candidate_detection.id for result in results} == {"det-left", "det-right"}
    assert all(result.metadata["overlap_reconciliation_enabled"] is False for result in results)
    assert all(
        result.metadata["overlap_reconciliation_action"] == "unchecked" for result in results
    )
    assert all("split_merge_assessment" in result.metadata for result in results)


def test_v2_can_reconstruct_unselected_peer_seed_from_preprocessed_frame():
    roi = np.zeros((8, 8), dtype=np.uint8)
    roi[0, 0] = 120
    own_mask = np.zeros_like(roi)
    own_mask[0, 0] = 255
    own = _detection("det-own", roi=roi, mask=own_mask)
    peer_roi = np.array([[120, 0], [0, 0]], dtype=np.uint8)
    peer_mask = np.array([[255, 0], [0, 0]], dtype=np.uint8)
    peer = _detection("det-peer", roi=peer_roi, mask=peer_mask, crop_x=4, crop_y=4)
    peer.roi_payload = None
    peer.mask_payload = None
    frame = np.zeros((8, 8), dtype=np.uint8)
    frame[4, 4] = 120

    results = refine_detections(
        [own],
        backend=HeuristicV2RefinementBackend(),
        all_candidate_seeds=[own, peer],
        frame_loader=lambda _frame_id: frame,
        options=RoiRefinementOptions(max_iterations=1),
    )

    assert len(results) == 1
    assert results[0].metadata["heuristic_v2_audit"]["competing_seed_pixels"] == 1
    assert results[0].refined_mask[4, 4] == 0
