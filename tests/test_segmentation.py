import numpy as np
import pytest

from Pelagia.domain import DetectionRecord
from Pelagia.processing.frame_codec import decode_array_payload
from Pelagia.processing.detection_refinement import (
    IdentityRoiRefinementModel,
    RoiRefinementOptions,
    build_roi_tiles,
    merge_refined_tiles,
    predict_refined_tile_masks,
    refine_detection,
    refine_detections,
)
from Pelagia.processing.frame_model import FrameData
from Pelagia.processing.frame_preprocess import preprocess_frame_for_segmentation
from Pelagia.processing.frame_threshold import (
    threshold_adaptive_gaussian,
    threshold_adaptive_mean,
    threshold_bounded_otsu,
    threshold_bounded_otsu_canny,
    threshold_canny,
    threshold_ensemble_and,
    threshold_ensemble_or,
    threshold_hysteresis,
    threshold_manual,
    threshold_otsu,
    threshold_percentile_background,
    threshold_sobel_edges,
)
from Pelagia.processing.detection_candidate import live_detection_candidate_wrapper, live_segment_wrapper, segment_frame, threshold_frame
from Pelagia.processing.detection_recording import build_candidate_detection_record
from Pelagia.processing.mask_augmentation import augment_mask
from Pelagia.processing.roi_assembly import assemble_candidate_rois
from Pelagia.processing.roi_filter import filter_candidate_rois, should_store_roi_payload


FRAME_ID = "00000000-0000-7000-8000-000000000042"


class FakeDatabaseLogger:
    def __init__(self):
        self.events = []

    def log(self, **kwargs):
        self.events.append(kwargs)
        return {"id": len(self.events), **kwargs}


class SplitOnceRefinementModel:
    def __init__(self):
        self.calls = 0

    def predict(self, batch):
        output = np.asarray(batch[..., 1]).copy()
        if self.calls == 0:
            output[:, :, output.shape[2] // 2:] = 0
        self.calls += 1
        return output


class EmptyRefinementModel:
    def predict(self, batch):
        return np.zeros(np.asarray(batch).shape[:3], dtype=np.float32)


class FakeContext:
    def __init__(self):
        self.logger = FakeDatabaseLogger()


def test_threshold_otsu_caps_value_at_thresholding_maximum():
    data = np.arange(256, dtype=np.uint8).reshape(16, 16)

    thresholded = threshold_otsu(data, thresholding_maximum_value=100)

    expected = np.where(data > 100, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(thresholded, expected)


def test_threshold_manual_uses_explicit_threshold_without_thresholding_cap():
    data = np.arange(256, dtype=np.uint8).reshape(16, 16)

    thresholded = threshold_manual(data, threshold=127)

    expected = np.where(data > 127, 255, 0).astype(np.uint8)
    np.testing.assert_array_equal(thresholded, expected)


def test_threshold_bounded_otsu_rejects_implausibly_full_masks():
    data = np.full((12, 12), 20, dtype=np.uint8)
    data[2:10, 2:10] = 80

    mask = threshold_bounded_otsu(
        data,
        min_contrast=50,
        max_foreground_fraction=0.1,
    )

    np.testing.assert_array_equal(mask, np.zeros_like(data, dtype=np.uint8))


def test_adaptive_and_percentile_thresholds_return_binary_masks():
    data = np.tile(np.arange(32, dtype=np.uint8), (32, 1)) + 20
    data[10:15, 10:15] = 180

    masks = [
        threshold_adaptive_mean(data, block_size=9, c=3),
        threshold_adaptive_gaussian(data, block_size=9, c=3),
        threshold_percentile_background(data, background_percentile=50, min_contrast=20),
    ]

    for mask in masks:
        assert mask.shape == data.shape
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})


def test_hysteresis_and_sobel_thresholds_return_binary_masks():
    data = np.full((24, 24), 20, dtype=np.uint8)
    data[8:16, 8:16] = 40
    data[10:14, 10:14] = 80

    hysteresis = threshold_hysteresis(data, low_threshold=30, high_threshold=60)
    sobel = threshold_sobel_edges(data, percentile=80)

    assert np.count_nonzero(hysteresis) > 0
    assert np.count_nonzero(sobel) > 0
    assert set(np.unique(hysteresis)).issubset({0, 255})
    assert set(np.unique(sobel)).issubset({0, 255})


def test_threshold_ensembles_combine_callables():
    data = np.full((20, 20), 20, dtype=np.uint8)
    data[6:14, 6:14] = 120

    threshold_fns = (
        lambda image: threshold_manual(image, 80),
        lambda image: threshold_canny(image, canny_params=(20, 60), blur_kernel=(1, 1)),
    )
    mask_or = threshold_ensemble_or(data, threshold_fns)
    mask_and = threshold_ensemble_and(data, threshold_fns)

    assert np.count_nonzero(mask_or) >= np.count_nonzero(mask_and)
    assert set(np.unique(mask_or)).issubset({0, 255})
    assert set(np.unique(mask_and)).issubset({0, 255})


def test_threshold_bounded_otsu_canny_combines_primitives():
    data = np.full((20, 20), 20, dtype=np.uint8)
    data[6:14, 6:14] = 120

    otsu_only = threshold_bounded_otsu(data, min_contrast=20)
    edges = threshold_canny(data, canny_params=(20, 60))
    combined = threshold_bounded_otsu_canny(
        data,
        run_canny=True,
        canny_params=(20, 60),
        dilate_kernel=(1, 1),
        min_contrast=20,
    )

    assert np.count_nonzero(combined) >= np.count_nonzero(otsu_only)
    assert np.count_nonzero(combined) >= np.count_nonzero(edges)


def test_mask_augmentation_assembly_and_filtering_are_discrete_steps():
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:4, 2:4] = 255
    mask[6, 6] = 255

    augmented = augment_mask(
        mask,
        steps=["remove_small_components", "dilate"],
        min_component_area=2,
        dilate_kernel_size=(3, 3),
    )
    candidates = assemble_candidate_rois(augmented, method="connected_components")
    filtered = filter_candidate_rois(candidates, min_width=3, min_height=3)

    assert len(candidates) == 1
    assert len(filtered) == 1
    assert filtered[0].bbox_x == 1
    assert filtered[0].bbox_y == 1
    assert filtered[0].bbox_w == 4
    assert filtered[0].bbox_h == 4


def test_recording_can_store_masks_without_small_roi_image_payloads():
    data = np.zeros((8, 8), dtype=np.uint8)
    data[2:4, 2:4] = 90
    mask = np.zeros_like(data)
    mask[2:4, 2:4] = 255
    source = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=2,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )
    candidate = assemble_candidate_rois(mask)[0]

    assert should_store_roi_payload(candidate, min_area=5) is False
    detection = build_candidate_detection_record(
        candidate,
        source_frame=source,
        processed_frame=source,
        encoding="raw",
        store_roi_payload=False,
        always_store_mask=True,
    )

    assert detection.roi_payload is None
    assert detection.roi_encoding is None
    assert detection.mask_payload is not None
    assert detection.mask_encoding == "raw"


def test_preprocess_frame_for_segmentation_applies_ordered_steps():
    data = np.full((4, 4), 20, dtype=np.uint8)
    data[1:3, 1:3] = 80
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=1,
        data=data,
        mask=mask,
        bkg=10,
    )

    processed = preprocess_frame_for_segmentation(
        frame,
        apply_mask=True,
    )

    assert processed.metadata["preprocessing_steps"] == [
        "mask",
        "background_correction",
        "invert_intensity",
    ]
    assert processed.metadata["foreground_polarity"] == "bright"
    assert processed.metadata["background_method"] == "divide"
    assert processed.metadata["min_field_value"] == 50.0
    assert processed.metadata["max_field_value"] == 255.0
    assert processed.read().shape == data.shape
    assert processed.read()[0, 0] == 0
    assert processed.read()[1, 1] == 0


def test_preprocess_frame_for_segmentation_applies_flatfield_when_enabled():
    data = np.array(
        [
            [20, 40, 80],
            [20, 40, 80],
            [20, 40, 80],
        ],
        dtype=np.uint8,
    )
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=1,
        data=data,
        metadata={"flatfield_profile": [20.0, 40.0, 80.0]},
    )

    processed = preprocess_frame_for_segmentation(
        frame,
        min_field_value=1,
    )

    assert processed.metadata["preprocessing_steps"] == ["flatfield", "invert_intensity"]
    assert processed.metadata["flatfield_correction"] is True
    assert processed.read().shape == data.shape


def test_preprocess_frame_for_segmentation_crops_geometry_and_mask():
    data = np.arange(25, dtype=np.uint8).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2:4, 1:4] = 255
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=1,
        data=data,
        mask=mask,
        width=5,
        height=5,
        bbox_x=100,
        bbox_y=200,
    )

    processed = preprocess_frame_for_segmentation(
        frame,
        apply_mask=False,
        crop_enabled=True,
        crop_x=1,
        crop_y=2,
        crop_w=3,
        crop_h=2,
    )

    np.testing.assert_array_equal(processed.read(), 255 - data[2:4, 1:4])
    np.testing.assert_array_equal(processed.mask, mask[2:4, 1:4])
    assert processed.width == 3
    assert processed.height == 2
    assert processed.bbox_x == 101
    assert processed.bbox_y == 202
    assert processed.metadata["preprocessing_steps"] == ["crop", "invert_intensity"]
    assert processed.metadata["crop_bbox"] == (101, 202, 3, 2)


def test_refine_detection_identity_uses_candidate_mask():
    roi = np.array([[0, 20], [40, 80]], dtype=np.uint8)
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    detection = DetectionRecord(
        run_id="00000000-0000-0000-0000-000000000001",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=0,
        bbox_y=0,
        bbox_w=2,
        bbox_h=2,
        area=2,
        perimeter=4,
        major_axis_length=2,
        minor_axis_length=2,
        min_gray_value=20,
        mean_gray_value=30,
        roi_payload=roi.tobytes(order="C"),
        mask_payload=mask.tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(roi.dtype),
        roi_shape=list(roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(mask.dtype),
        mask_shape=list(mask.shape),
        id="00000000-0000-7000-8000-000000000099",
    )

    result = refine_detection(detection)

    np.testing.assert_array_equal(result.roi, roi)
    np.testing.assert_array_equal(result.candidate_mask, mask)
    np.testing.assert_array_equal(result.refined_mask, mask)
    assert result.method == "identity"
    refined_record = result.as_detection_record()
    assert refined_record.metadata["detection_stage"] == "refined"
    assert refined_record.metadata["refinement_tile_count"] == 1
    assert refined_record.bbox_x == 0
    assert refined_record.bbox_y == 0
    assert refined_record.bbox_w == 2
    assert refined_record.bbox_h == 2


def test_refine_detections_drops_empty_refined_masks():
    roi = np.array([[0, 20], [40, 80]], dtype=np.uint8)
    mask = np.array([[0, 255], [255, 0]], dtype=np.uint8)
    detection = DetectionRecord(
        run_id="00000000-0000-0000-0000-000000000001",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=0,
        bbox_y=0,
        bbox_w=2,
        bbox_h=2,
        area=2,
        perimeter=4,
        major_axis_length=2,
        minor_axis_length=2,
        min_gray_value=20,
        mean_gray_value=30,
        roi_payload=roi.tobytes(order="C"),
        mask_payload=mask.tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(roi.dtype),
        roi_shape=list(roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(mask.dtype),
        mask_shape=list(mask.shape),
        id="00000000-0000-7000-8000-000000000099",
    )

    result = refine_detection(detection, model=EmptyRefinementModel())

    assert result.metadata["refined_foreground_pixels"] == 0
    with pytest.raises(ValueError, match="empty mask"):
        result.as_detection_record()
    assert refine_detections([detection], model=EmptyRefinementModel()) == []


def test_refinement_builds_overlapping_padded_tiles():
    roi = np.arange(30, dtype=np.uint8).reshape(5, 6)
    mask = np.ones((5, 6), dtype=np.uint8) * 255
    detection = DetectionRecord(
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=12,
        bbox_y=20,
        bbox_w=6,
        bbox_h=5,
        crop_bbox_x=10,
        crop_bbox_y=18,
        crop_bbox_w=6,
        crop_bbox_h=5,
        area=30,
        perimeter=22,
        major_axis_length=6,
        minor_axis_length=5,
        min_gray_value=0,
        mean_gray_value=10,
        roi_payload=roi.tobytes(order="C"),
    )

    tiles = build_roi_tiles(
        detection,
        roi,
        mask,
        options=RoiRefinementOptions(tile_size=4, overlap_fraction=0.5),
    )

    assert len(tiles) == 4
    assert tiles[0].image.shape == (4, 4)
    assert tiles[0].frame_bbox == (10, 18, 4, 4)
    assert tiles[-1].local_bbox == (2, 1, 4, 4)
    assert tiles[-1].frame_bbox == (12, 19, 4, 4)


def test_identity_refinement_model_merges_tiles_to_candidate_mask():
    roi = np.arange(30, dtype=np.uint8).reshape(5, 6)
    mask = np.zeros((5, 6), dtype=np.uint8)
    mask[1:4, 2:5] = 255
    detection = DetectionRecord(
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=0,
        bbox_y=0,
        bbox_w=6,
        bbox_h=5,
        area=9,
        perimeter=12,
        major_axis_length=3,
        minor_axis_length=3,
        min_gray_value=0,
        mean_gray_value=10,
        roi_payload=roi.tobytes(order="C"),
    )
    options = RoiRefinementOptions(tile_size=4, overlap_fraction=0.5)
    tiles = build_roi_tiles(detection, roi, mask, options=options)
    tile_masks = predict_refined_tile_masks(
        tiles,
        model=IdentityRoiRefinementModel(),
        options=options,
    )

    merged = merge_refined_tiles(tiles, tile_masks, roi.shape)

    np.testing.assert_array_equal(merged, mask)


def test_refine_detection_loads_frame_only_when_edge_expansion_is_needed():
    roi = np.arange(16, dtype=np.uint8).reshape(4, 4)
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 3] = 255
    detection = DetectionRecord(
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=6,
        bbox_y=3,
        bbox_w=1,
        bbox_h=2,
        crop_bbox_x=3,
        crop_bbox_y=2,
        crop_bbox_w=4,
        crop_bbox_h=4,
        area=2,
        perimeter=4,
        major_axis_length=2,
        minor_axis_length=1,
        min_gray_value=20,
        mean_gray_value=30,
        roi_payload=roi.tobytes(order="C"),
        mask_payload=mask.tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(roi.dtype),
        roi_shape=list(roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(mask.dtype),
        mask_shape=list(mask.shape),
        id="det-1",
    )
    loaded = []
    frame = np.arange(64, dtype=np.uint8).reshape(8, 8)

    def load_frame(frame_id):
        loaded.append(frame_id)
        return frame

    result = refine_detection(
        detection,
        frame_loader=load_frame,
        options=RoiRefinementOptions(tile_size=4, overlap_fraction=0, expansion_pixels=1),
    )

    assert loaded == [FRAME_ID]
    assert result.metadata["refinement_frame_loaded"] is True
    assert result.metadata["refinement_expansion_count"] == 1
    assert result.crop_bbox == (3, 2, 5, 4)
    refined_record = result.as_detection_record(encoding="raw")
    assert refined_record.crop_bbox_w == 5
    assert refined_record.crop_bbox_h == 4
    assert refined_record.bbox_x == 6
    assert refined_record.bbox_y == 3
    assert refined_record.bbox_w == 1
    assert refined_record.bbox_h == 2


def test_refine_detection_does_not_load_frame_for_interior_mask():
    roi = np.arange(25, dtype=np.uint8).reshape(5, 5)
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 255
    detection = DetectionRecord(
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=10,
        bbox_y=10,
        bbox_w=1,
        bbox_h=1,
        crop_bbox_x=8,
        crop_bbox_y=8,
        crop_bbox_w=5,
        crop_bbox_h=5,
        area=1,
        perimeter=1,
        major_axis_length=1,
        minor_axis_length=1,
        min_gray_value=12,
        mean_gray_value=12,
        roi_payload=roi.tobytes(order="C"),
        mask_payload=mask.tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(roi.dtype),
        roi_shape=list(roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(mask.dtype),
        mask_shape=list(mask.shape),
    )

    result = refine_detection(
        detection,
        frame_loader=lambda frame_id: (_ for _ in ()).throw(AssertionError("frame loaded")),
        options=RoiRefinementOptions(tile_size=4, overlap_fraction=0.5),
    )

    assert result.metadata["refinement_frame_loaded"] is False
    assert result.metadata["refinement_expansion_count"] == 0


def test_refine_detection_loads_initial_roi_from_frame_when_payload_is_missing():
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[2, 2] = 255
    detection = DetectionRecord(
        id="det-no-roi",
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=4,
        bbox_y=3,
        bbox_w=1,
        bbox_h=1,
        crop_bbox_x=2,
        crop_bbox_y=1,
        crop_bbox_w=4,
        crop_bbox_h=4,
        area=1,
        perimeter=1,
        major_axis_length=1,
        minor_axis_length=1,
        min_gray_value=12,
        mean_gray_value=12,
        roi_payload=None,
        mask_payload=mask.tobytes(order="C"),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(mask.dtype),
        mask_shape=list(mask.shape),
    )
    frame = np.arange(49, dtype=np.uint8).reshape(7, 7)
    loaded = []

    def load_frame(frame_id):
        loaded.append(frame_id)
        return frame

    result = refine_detection(
        detection,
        frame_loader=load_frame,
        options=RoiRefinementOptions(tile_size=4, overlap_fraction=0.5),
    )

    assert loaded == [FRAME_ID]
    np.testing.assert_array_equal(result.roi, frame[1:5, 2:6])
    assert result.metadata["refinement_frame_loaded"] is True
    assert result.metadata["refinement_initial_roi_source"] == "frame"


def test_refine_detections_reconciles_contained_overlaps():
    large_roi = np.arange(16, dtype=np.uint8).reshape(4, 4)
    large_mask = np.ones((4, 4), dtype=np.uint8) * 255
    small_roi = np.arange(4, dtype=np.uint8).reshape(2, 2)
    small_mask = np.ones((2, 2), dtype=np.uint8) * 255
    large = DetectionRecord(
        id="det-large",
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=1,
        bbox_x=0,
        bbox_y=0,
        bbox_w=4,
        bbox_h=4,
        crop_bbox_x=0,
        crop_bbox_y=0,
        crop_bbox_w=4,
        crop_bbox_h=4,
        area=16,
        perimeter=16,
        major_axis_length=4,
        minor_axis_length=4,
        min_gray_value=0,
        mean_gray_value=8,
        roi_payload=large_roi.tobytes(order="C"),
        mask_payload=large_mask.tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(large_roi.dtype),
        roi_shape=list(large_roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(large_mask.dtype),
        mask_shape=list(large_mask.shape),
    )
    small = DetectionRecord(
        id="det-small",
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=2,
        bbox_x=1,
        bbox_y=1,
        bbox_w=2,
        bbox_h=2,
        crop_bbox_x=1,
        crop_bbox_y=1,
        crop_bbox_w=2,
        crop_bbox_h=2,
        area=4,
        perimeter=8,
        major_axis_length=2,
        minor_axis_length=2,
        min_gray_value=0,
        mean_gray_value=2,
        roi_payload=small_roi.tobytes(order="C"),
        mask_payload=small_mask.tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(small_roi.dtype),
        roi_shape=list(small_roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(small_mask.dtype),
        mask_shape=list(small_mask.shape),
    )

    results = refine_detections(
        [small, large],
        options=RoiRefinementOptions(tile_size=4, overlap_fraction=0),
    )

    assert len(results) == 1
    assert results[0].candidate_detection.id == "det-large"
    assert results[0].metadata["overlap_reconciliation_action"] == "kept"
    assert results[0].metadata["consumed_candidate_detection_ids"] == ["det-small"]
    consumed = results[0].metadata["overlap_reconciliation_consumed"][0]
    assert consumed["candidate_detection_id"] == "det-small"
    assert consumed["consumed_containment_fraction"] == 1.0
    assert consumed["keeper_mask_area"] == 16


def test_refine_detections_can_skip_overlap_reconciliation():
    roi = np.ones((2, 2), dtype=np.uint8)
    mask = np.ones((2, 2), dtype=np.uint8) * 255
    detections = [
        DetectionRecord(
            id=f"det-{index}",
            run_id="run-1",
            frame_id=FRAME_ID,
            roi_index=index,
            bbox_x=0,
            bbox_y=0,
            bbox_w=2,
            bbox_h=2,
            area=4,
            perimeter=8,
            major_axis_length=2,
            minor_axis_length=2,
            min_gray_value=1,
            mean_gray_value=1,
            roi_payload=roi.tobytes(order="C"),
            mask_payload=mask.tobytes(order="C"),
            roi_encoding="raw",
            roi_format="raw_ndarray_c_order",
            roi_dtype=str(roi.dtype),
            roi_shape=list(roi.shape),
            mask_encoding="raw",
            mask_format="raw_ndarray_c_order",
            mask_dtype=str(mask.dtype),
            mask_shape=list(mask.shape),
        )
        for index in (1, 2)
    ]

    results = refine_detections(
        detections,
        options=RoiRefinementOptions(
            tile_size=4,
            overlap_fraction=0,
            overlap_reconciliation_enabled=False,
        ),
    )

    assert len(results) == 2
    assert all(result.metadata["overlap_reconciliation_enabled"] is False for result in results)


def test_refine_detections_discovers_residual_split_children_without_loading_frame():
    roi = np.full((6, 8), 10, dtype=np.uint8)
    roi[2:4, 1:3] = 80
    roi[2:4, 5:7] = 90
    mask = np.zeros_like(roi)
    mask[2:4, 1:3] = 255
    mask[2:4, 5:7] = 255
    detection = DetectionRecord(
        id="det-composite",
        run_id="run-1",
        frame_id=FRAME_ID,
        roi_index=3,
        bbox_x=11,
        bbox_y=22,
        bbox_w=6,
        bbox_h=2,
        crop_bbox_x=10,
        crop_bbox_y=20,
        crop_bbox_w=8,
        crop_bbox_h=6,
        area=8,
        perimeter=16,
        major_axis_length=6,
        minor_axis_length=2,
        min_gray_value=80,
        mean_gray_value=85,
        roi_payload=roi.tobytes(order="C"),
        mask_payload=mask.tobytes(order="C"),
        roi_encoding="raw",
        roi_format="raw_ndarray_c_order",
        roi_dtype=str(roi.dtype),
        roi_shape=list(roi.shape),
        mask_encoding="raw",
        mask_format="raw_ndarray_c_order",
        mask_dtype=str(mask.dtype),
        mask_shape=list(mask.shape),
        metadata={"assembly_method": "connected_components", "padding": 0},
    )

    results = refine_detections(
        [detection],
        model=SplitOnceRefinementModel(),
        frame_loader=lambda frame_id: (_ for _ in ()).throw(AssertionError("frame loaded")),
        options=RoiRefinementOptions(
            tile_size=8,
            overlap_fraction=0,
            residual_discovery_enabled=True,
            residual_min_area=1,
            residual_min_width=1,
            residual_min_height=1,
            residual_padding=1,
            overlap_reconciliation_enabled=False,
        ),
        method="split_once",
    )

    assert len(results) == 2
    parent, child = results
    assert parent.candidate_detection.id == "det-composite"
    assert parent.metadata["residual_discovery_child_count"] == 1
    assert child.candidate_detection.metadata["synthetic_candidate"] is True
    assert child.metadata["residual_discovery_action"] == "split_child"
    assert child.metadata["split_from_candidate_detection_id"] == "det-composite"
    assert child.bbox == (15, 22, 2, 2)
    assert child.crop_bbox == (14, 21, 4, 4)
    child_record = child.as_detection_record(encoding="raw")
    assert child_record.metadata["synthetic_candidate"] is True
    assert child_record.metadata["split_from_candidate_detection_id"] == "det-composite"
    decoded_child_roi = decode_array_payload(
        child_record.roi_payload,
        {
            "array_encoding": child_record.roi_encoding,
            "dtype": child_record.roi_dtype,
            "shape": child_record.roi_shape,
        },
    )
    np.testing.assert_array_equal(decoded_child_roi, roi[1:5, 4:8])


def test_segment_frame_returns_roi_detection_records_with_raw_payload():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        bbox_x=100,
        bbox_y=200,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )

    detections = segment_frame(
        frame,
        threshold=1,
        apply_preprocessing=False,
        min_perimeter=0,
        min_width=0,
        min_height=0,
        padding=0,
        roi_encoding="raw",
        store_roi_payload_min_width=0,
        store_roi_payload_min_height=0,
        store_roi_payload_min_width_plus_height=0,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.run_id == "00000000-0000-0000-0000-000000000001"
    assert detection.frame_id == FRAME_ID
    assert detection.roi_index == 1
    assert detection.bbox_x == 103
    assert detection.bbox_y == 202
    assert detection.bbox_w == 4
    assert detection.bbox_h == 3
    assert detection.area == 12
    assert detection.min_gray_value == 50
    assert detection.mean_gray_value == 50
    assert detection.roi_encoding == "raw"
    assert detection.roi_shape == [3, 4]
    assert detection.mask_encoding == "raw"
    assert detection.mask_shape == [3, 4]
    assert detection.mask_payload is not None

    roi_metadata = {
        "array_encoding": detection.roi_encoding,
        "dtype": detection.roi_dtype,
        "shape": detection.roi_shape,
    }
    decoded = decode_array_payload(detection.roi_payload, roi_metadata)
    np.testing.assert_array_equal(decoded, np.full((3, 4), 50, dtype=np.uint8))

    mask_metadata = {
        "array_encoding": detection.mask_encoding,
        "dtype": detection.mask_dtype,
        "shape": detection.mask_shape,
    }
    decoded_mask = decode_array_payload(detection.mask_payload, mask_metadata)
    np.testing.assert_array_equal(decoded_mask, np.full((3, 4), 255, dtype=np.uint8))


def test_segment_frame_writes_structured_log_events():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "asset_id": "00000000-0000-0000-0000-000000000002",
            "frame_id": FRAME_ID,
        },
    )
    context = FakeContext()

    detections = segment_frame(
        frame,
        threshold=1,
        apply_preprocessing=False,
        min_perimeter=0,
        min_width=0,
        min_height=0,
        padding=0,
        roi_encoding="raw",
        context=context,
    )

    assert len(detections) == 1
    assert [event["event_type"] for event in context.logger.events] == [
        "segmentation.frame_started",
        "segmentation.frame_completed",
    ]
    completed = context.logger.events[-1]
    assert completed["duration_ms"] >= 0
    assert completed["run_id"] == "00000000-0000-0000-0000-000000000001"
    assert completed["asset_id"] == "00000000-0000-0000-0000-000000000002"
    assert completed["payload"]["detection_count"] == 1
    assert completed["payload"]["source_component_count"] == 1


def test_threshold_frame_returns_binary_mask():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )

    result = threshold_frame(
        frame,
        threshold=1,
        apply_preprocessing=False,
    )

    assert result.threshold_mask.dtype == np.uint8
    assert result.threshold_mask.shape == (10, 10)
    assert int(np.count_nonzero(result.threshold_mask)) == 12
    assert result.metadata["stage_counts"]["threshold_foreground_pixels"] == 12
    assert result.metadata["threshold_method"] == "manual"


def test_live_detection_candidate_wrapper_returns_transient_detection_records(monkeypatch):
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )
    monkeypatch.setattr("Pelagia.processing.frame_store.retrieve_frame", lambda frame_id, context: frame)

    detections = live_detection_candidate_wrapper(
        FRAME_ID,
        threshold=1,
        apply_preprocessing=False,
        min_perimeter=0,
        min_width=0,
        min_height=0,
        padding=0,
        store_roi_payload_min_width=0,
        store_roi_payload_min_height=0,
        store_roi_payload_min_width_plus_height=0,
    )

    assert len(detections) == 1
    assert detections[0].roi_payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert detections[0].mask_payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert detections[0].roi_encoding == "png"
    assert live_segment_wrapper(
        FRAME_ID,
        threshold=1,
        apply_preprocessing=False,
        min_perimeter=0,
        min_width=0,
        min_height=0,
        padding=0,
        store_roi_payload_min_width=0,
        store_roi_payload_min_height=0,
        store_roi_payload_min_width_plus_height=0,
    )[0].roi_encoding == "png"


def test_segment_frame_stores_padded_roi_context_and_mask():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        bbox_x=100,
        bbox_y=200,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )

    detections = segment_frame(
        frame,
        threshold=1,
        apply_preprocessing=False,
        min_perimeter=0,
        min_width=0,
        min_height=0,
        padding=1,
        roi_encoding="raw",
        store_roi_payload_min_width=0,
        store_roi_payload_min_height=0,
        store_roi_payload_min_width_plus_height=0,
    )

    assert len(detections) == 1
    detection = detections[0]
    assert detection.bbox_x == 103
    assert detection.bbox_y == 202
    assert detection.bbox_w == 4
    assert detection.bbox_h == 3
    assert detection.metadata["roi_bbox"] == (102, 201, 6, 5)
    assert detection.metadata["actual_padding"] == {
        "left": 1,
        "top": 1,
        "right": 1,
        "bottom": 1,
    }

    roi_metadata = {
        "array_encoding": detection.roi_encoding,
        "dtype": detection.roi_dtype,
        "shape": detection.roi_shape,
    }
    decoded = decode_array_payload(detection.roi_payload, roi_metadata)
    expected_crop = np.zeros((5, 6), dtype=np.uint8)
    expected_crop[1:4, 1:5] = 50
    np.testing.assert_array_equal(decoded, expected_crop)

    mask_metadata = {
        "array_encoding": detection.mask_encoding,
        "dtype": detection.mask_dtype,
        "shape": detection.mask_shape,
    }
    decoded_mask = decode_array_payload(detection.mask_payload, mask_metadata)
    expected_mask = np.zeros((5, 6), dtype=np.uint8)
    expected_mask[1:4, 1:5] = 255
    np.testing.assert_array_equal(decoded_mask, expected_mask)


def test_segment_frame_filters_by_bbox_perimeter():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )

    assert segment_frame(
        frame,
        threshold=1,
        apply_preprocessing=False,
        min_perimeter=15,
        roi_encoding="raw",
    ) == []


def test_candidate_recording_auto_uses_png_for_small_roi_payloads():
    data = np.full((4, 4), 10, dtype=np.uint8)
    data[1:3, 1:3] = 255
    source = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1:3, 1:3] = 255
    candidate = assemble_candidate_rois(mask)[0]

    detection = build_candidate_detection_record(
        candidate,
        source_frame=source,
        processed_frame=source,
        encoding="auto",
        zstd_min_bytes=100,
    )

    assert detection.roi_payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert detection.roi_encoding == "png"
    assert detection.roi_format == "png"
