"""Regression coverage for candidate ROI materialization and recording options."""

from __future__ import annotations

import numpy as np

from Pelagia.processing.detection_recording import (
    build_candidate_detection_record,
    choose_roi_encoding,
)
from Pelagia.processing.frame_model import FrameData
from Pelagia.processing.frame_threshold import effective_foreground_intensity_threshold
from Pelagia.processing.roi_assembly import assemble_candidate_rois
from Pelagia.processing.roi_filter import filter_candidate_rois
from Pelagia.processing.timing import collect_result_timings


def test_connected_components_filter_before_masks_and_contours(monkeypatch):
    mask = np.zeros((24, 24), dtype=np.uint8)
    for row, column in ((1, 1), (1, 5), (1, 9), (5, 1), (5, 5)):
        mask[row, column] = 255
    mask[12:18, 12:18] = 255

    # Establish the pre-optimization result, where filtering follows complete
    # candidate construction, before spying on materialization.
    expected = filter_candidate_rois(
        assemble_candidate_rois(mask),
        min_width=3,
        min_height=3,
    )

    from Pelagia.processing import roi_assembly

    original_largest_contour = roi_assembly._largest_contour
    contour_calls = 0

    def count_contours(component_mask):
        nonlocal contour_calls
        contour_calls += 1
        return original_largest_contour(component_mask)

    monkeypatch.setattr(roi_assembly, "_largest_contour", count_contours)
    assembly_stats: dict[str, int | float] = {}
    actual = assemble_candidate_rois(
        mask,
        filter_options={"min_width": 3, "min_height": 3},
        assembly_stats=assembly_stats,
    )

    assert contour_calls == 1
    assert assembly_stats["source_component_count"] == 6
    assert assembly_stats["materialized_candidate_count"] == 1
    assert assembly_stats["filtered_before_materialization_count"] == 5
    assert [(roi.roi_index, roi.bbox_x, roi.bbox_y, roi.bbox_w, roi.bbox_h) for roi in actual] == [
        (roi.roi_index, roi.bbox_x, roi.bbox_y, roi.bbox_w, roi.bbox_h) for roi in expected
    ]
    np.testing.assert_array_equal(actual[0].mask, expected[0].mask)
    np.testing.assert_array_equal(actual[0].contour, expected[0].contour)


def test_candidate_threshold_provenance_uses_only_matching_intensity_space():
    image = np.array([[10, 10], [20, 100]], dtype=np.uint8)
    assert effective_foreground_intensity_threshold(image, method="manual", threshold=50) == 50.0
    assert effective_foreground_intensity_threshold(
        image, method="percentile_background",
        percentile_background_percentile=50,
        percentile_min_contrast=15,
    ) == 30.0
    assert effective_foreground_intensity_threshold(image, method="canny") is None
    assert effective_foreground_intensity_threshold(image.astype(np.uint16), method="manual", threshold=50) is None


def test_recording_with_resolved_options_does_not_load_defaults(monkeypatch):
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:5, 3:6] = 90
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=1,
        data=image,
        metadata={
            "frame_id": "00000000-0000-7000-8000-000000000042",
            "run_id": "00000000-0000-0000-0000-000000000001",
        },
    )
    candidate = assemble_candidate_rois(np.where(image, 255, 0).astype(np.uint8))[0]

    def fail_default_load():
        raise AssertionError("fully resolved recording options must not load defaults")

    monkeypatch.setattr("Pelagia.processing.detection_recording.default_processing_config", fail_default_load)
    record = build_candidate_detection_record(
        candidate,
        source_frame=frame,
        processed_frame=frame,
        encoding="raw",
        small_roi_encoding="zstd",
        large_roi_encoding="jpg",
        large_roi_min_pixels=50_000,
        roi_quality=90,
        mask_encoding="zstd",
    )

    assert record.roi_encoding == "raw"
    assert record.mask_encoding == "zstd"


def test_assembly_leaf_timings_are_opt_in_for_nested_callers():
    @collect_result_timings()
    def run_assembly():
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 255
        assemble_candidate_rois(mask)
        return {"frame_count": 1}

    result = run_assembly()

    assert result["timings"]["phases_ms"] == {}


def test_standalone_recording_loads_one_default_snapshot(monkeypatch):
    image = np.zeros((8, 8), dtype=np.uint8)
    image[2:5, 3:6] = 90
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=1,
        data=image,
        metadata={
            "frame_id": "00000000-0000-7000-8000-000000000042",
            "run_id": "00000000-0000-0000-0000-000000000001",
        },
    )
    candidate = assemble_candidate_rois(np.where(image, 255, 0).astype(np.uint8))[0]
    from Pelagia.processing import detection_recording

    original_default_processing_config = detection_recording.default_processing_config
    calls = 0

    def count_default_loads():
        nonlocal calls
        calls += 1
        return original_default_processing_config()

    monkeypatch.setattr(detection_recording, "default_processing_config", count_default_loads)
    record = build_candidate_detection_record(candidate, source_frame=frame, processed_frame=frame)

    assert calls == 1
    assert record.roi_encoding is not None
    assert choose_roi_encoding(
        np.zeros((1, 1), dtype=np.uint8),
        small_roi_encoding="",
        large_roi_encoding="",
        large_roi_min_pixels=1,
    ) == "jpg"


def test_prefilter_matches_postfilter_at_bbox_geometric_boundaries():
    mask = np.zeros((35, 50), dtype=np.uint8)
    mask[1:5, 1:5] = 255
    mask[10:16, 10:18] = 255
    mask[20:30, 30:42] = 255
    # A ring verifies that CC area remains bbox area rather than pixel count.
    mask[22:28, 32:40] = 0
    for connectivity in (4, 8):
        for dimension, boundary in (
            ('area', 48), ('width', 8), ('height', 6),
            ('width_plus_height', 14),
        ):
            for prefix in ('min', 'max'):
                options = {f'{prefix}_{dimension}': boundary}
                expected = filter_candidate_rois(
                    assemble_candidate_rois(mask, connectivity=connectivity), **options,
                )
                actual = assemble_candidate_rois(
                    mask, connectivity=connectivity, filter_options=options,
                )
                assert len(actual) == len(expected)
                for before, after in zip(expected, actual):
                    assert (before.roi_index, before.bbox_x, before.bbox_y, before.area,
                            before.perimeter, before.metadata) == (
                        after.roi_index, after.bbox_x, after.bbox_y, after.area,
                        after.perimeter, after.metadata,
                    )
                    np.testing.assert_array_equal(before.mask, after.mask)
                    np.testing.assert_array_equal(before.contour, after.contour)


def test_candidate_limit_preserves_source_and_survivor_counts_and_leaf_timings():
    from types import SimpleNamespace
    from Pelagia.config import CoreConfig
    from Pelagia.processing.detection_candidate import segment_frame

    image = np.zeros((40, 40), dtype=np.uint8)
    image[1, 1] = image[1, 5] = image[1, 9] = 200
    for offset in (5, 15, 25):
        image[10:16, offset:offset + 6] = 200
    frame = FrameData(
        sourcePath='', filename='synthetic', frameNumber=1, data=image,
        metadata={'frame_id': '00000000-0000-7000-8000-000000000042',
                  'run_id': '00000000-0000-0000-0000-000000000001'},
    )

    @collect_result_timings()
    def run():
        records = segment_frame(
            frame, context=SimpleNamespace(config=CoreConfig()), threshold=1,
            apply_preprocessing=False, mask_augmentation_enabled=False,
            min_width=3, min_height=3, max_detections=2,
            roi_encoding='raw', mask_encoding='raw', padding=0,
        )
        return {'frame_count': 1, 'records': records}

    result = run()
    records = result['records']
    assert [record.roi_index for record in records] == [1, 2]
    counts = records[0].metadata['stage_counts']
    assert counts['source_component_count'] == 6
    assert counts['materialized_candidate_count'] == 3
    assert counts['filtered_before_materialization_count'] == 3
    assert counts['filtered_candidate_count'] == 3
    assert counts['recorded_detection_count'] == 2
    assert records[0].metadata['candidate_limit_applied'] is True
    phases = result['timings']['phases_ms']
    assert {'segmentation.connected_component_labeling',
            'segmentation.connected_component_materialization',
            'segmentation.orientation', 'segmentation.record_preparation',
            'segmentation.record_finalization'} <= phases.keys()
    assert 'segmentation.roi_assembly' not in phases
    assert result['timings']['measured_ms'] <= result['timings']['total_ms']
