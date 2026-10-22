"""Regression coverage for explicit ROI measurement semantics."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from Pelagia.config import CoreConfig
from Pelagia.domain import DetectionRecord
from Pelagia.processing.detection_candidate import segment_frame
from Pelagia.processing.detection_refinement import DetectionRefinementResult
from Pelagia.processing.frame_model import FrameData
from Pelagia.processing.roi_assembly import assemble_candidate_rois
from Pelagia.processing.roi_filter import filter_candidate_rois
from Pelagia.processing.roi_measurements import measure_mask_contours


def test_candidate_area_is_bbox_area_for_all_assembly_methods():
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1:5, 1] = 255
    mask[4, 1:6] = 255

    component = assemble_candidate_rois(mask, method="connected_components")[0]
    contour = assemble_candidate_rois(mask, method="contours")[0]

    for candidate in (component, contour):
        assert candidate.area == 20.0
        assert candidate.metadata["roi_measurements"] == {
            "version": 1,
            "area_method": "bounding_box",
            "perimeter_method": "opencv_external_contours",
        }


def test_generation_gate_uses_bbox_perimeter_then_roi_filter_uses_contour_perimeter():
    mask = np.zeros((6, 7), dtype=np.uint8)
    mask[1:5, 1] = 255
    mask[4, 1:6] = 255

    candidate = assemble_candidate_rois(mask)[0]
    assert candidate.bbox_perimeter == 18
    assert candidate.perimeter < 17

    assert len(assemble_candidate_rois(mask, filter_options={"min_perimeter": 17})) == 1
    assert len(
        assemble_candidate_rois(
            mask,
            method="contours",
            filter_options={"min_perimeter": 17},
        )
    ) == 1
    assert filter_candidate_rois([candidate], min_perimeter=17) == []
    assert assemble_candidate_rois(mask, filter_options={"max_perimeter": 17}) == []
    assert assemble_candidate_rois(
        mask,
        method="contours",
        filter_options={"max_perimeter": 17},
    ) == []
    assert len(filter_candidate_rois([candidate], max_perimeter=17)) == 1

    candidate.area = 1.0
    assert len(filter_candidate_rois([candidate], min_area=20)) == 1


def test_candidate_generation_does_not_apply_a_second_contour_perimeter_filter():
    image = np.zeros((6, 7), dtype=np.uint8)
    image[1:5, 1] = 200
    image[4, 1:6] = 200
    frame = FrameData(
        sourcePath="",
        filename="synthetic",
        frameNumber=1,
        data=image,
        metadata={"frame_id": "frame-1", "run_id": "run-1"},
    )

    records = segment_frame(
        frame,
        context=SimpleNamespace(config=CoreConfig()),
        threshold=1,
        apply_preprocessing=False,
        mask_augmentation_enabled=False,
        min_perimeter=17,
        padding=0,
        encode_payloads=False,
    )

    assert len(records) == 1
    assert records[0].perimeter < 17


def test_refined_record_area_uses_external_contours_instead_of_pixel_count():
    mask = np.zeros((7, 8), dtype=np.uint8)
    mask[1:6, 1:7] = 255
    mask[2:5, 2:6] = 0
    candidate = DetectionRecord(
        run_id="run-1",
        frame_id="frame-1",
        roi_index=1,
        bbox_x=1,
        bbox_y=1,
        bbox_w=6,
        bbox_h=5,
        area=30.0,
        perimeter=18.0,
        major_axis_length=6.0,
        minor_axis_length=5.0,
        min_gray_value=1,
        mean_gray_value=1.0,
        roi_payload=None,
    )

    assert np.count_nonzero(mask) == 18
    assert measure_mask_contours(mask) == (20.0, 18.0)

    record = DetectionRefinementResult(
        candidate_detection=candidate,
        roi=np.ones_like(mask),
        candidate_mask=mask,
        refined_mask=mask,
        crop_bbox=(0, 0, 8, 7),
    ).as_detection_record(encoding="raw", mask_encoding="raw")

    assert record.area == 20.0
    assert record.perimeter == 18.0
    assert record.metadata["roi_measurements"] == {
        "version": 1,
        "area_method": "opencv_external_contours",
        "perimeter_method": "opencv_external_contours",
    }
