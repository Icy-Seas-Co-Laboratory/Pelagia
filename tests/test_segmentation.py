import numpy as np

from Pelagia.domain import DetectionRecord
from Pelagia.processing.frame_codec import decode_array_payload
from Pelagia.processing.detection_refinement import refine_detection
from Pelagia.processing.frame_model import FrameData
from Pelagia.processing.frame_preprocess import preprocess_frame_for_segmentation
from Pelagia.processing.frame_threshold import (
    threshold_adaptive_gaussian,
    threshold_adaptive_mean,
    threshold_bounded_otsu,
    threshold_bounded_otsu_canny,
    threshold_boundedotsu_canny,
    threshold_canny,
    threshold_ensemble_and,
    threshold_ensemble_or,
    threshold_hysteresis,
    threshold_manual,
    threshold_otsu,
    threshold_percentile_background,
    threshold_sobel_edges,
)
from Pelagia.processing.detection_candidate import live_segment_wrapper, segment_frame, store_roi


FRAME_ID = "00000000-0000-7000-8000-000000000042"


class FakeDatabaseLogger:
    def __init__(self):
        self.events = []

    def log(self, **kwargs):
        self.events.append(kwargs)
        return {"id": len(self.events), **kwargs}


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


def test_threshold_boundedotsu_canny_combines_primitives():
    data = np.full((20, 20), 20, dtype=np.uint8)
    data[6:14, 6:14] = 120

    otsu_only = threshold_bounded_otsu(data, min_contrast=20)
    edges = threshold_canny(data, canny_params=(20, 60))
    combined = threshold_boundedotsu_canny(
        data,
        run_canny=True,
        canny_params=(20, 60),
        dilate_kernel=(1, 1),
        min_contrast=20,
    )
    alias_combined = threshold_bounded_otsu_canny(
        data,
        run_canny=True,
        canny_params=(20, 60),
        dilate_kernel=(1, 1),
        min_contrast=20,
    )

    assert np.count_nonzero(combined) >= np.count_nonzero(otsu_only)
    assert np.count_nonzero(combined) >= np.count_nonzero(edges)
    np.testing.assert_array_equal(combined, alias_combined)


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
    )

    processed = preprocess_frame_for_segmentation(
        frame,
        flatfield_correction=False,
        background_correction=True,
        background=10,
        invert_intensity=True,
    )

    assert processed.metadata["preprocessing_steps"] == [
        "mask",
        "background_correction",
        "invert_intensity",
    ]
    assert processed.metadata["foreground_polarity"] == "bright"
    assert processed.read().shape == data.shape
    assert processed.read()[0, 0] == 255
    assert processed.read()[1, 1] == 185


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
        flatfield_correction=False,
        apply_mask=False,
        crop_enabled=True,
        crop_x=1,
        crop_y=2,
        crop_w=3,
        crop_h=2,
    )

    np.testing.assert_array_equal(processed.read(), data[2:4, 1:4])
    np.testing.assert_array_equal(processed.mask, mask[2:4, 1:4])
    assert processed.width == 3
    assert processed.height == 2
    assert processed.bbox_x == 101
    assert processed.bbox_y == 202
    assert processed.metadata["preprocessing_steps"] == ["crop"]
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
    assert result.as_detection_record().metadata["detection_stage"] == "refined"


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
        flatfield_correction=False,
        min_perimeter=0,
        padding=0,
        roi_encoding="raw",
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
        flatfield_correction=False,
        min_perimeter=0,
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


def test_live_segment_wrapper_returns_transient_detection_records(monkeypatch):
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

    detections = live_segment_wrapper(
        FRAME_ID,
        threshold=1,
        flatfield_correction=False,
        min_perimeter=0,
        padding=0,
    )

    assert len(detections) == 1
    assert detections[0].roi_payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert detections[0].mask_payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert detections[0].roi_encoding == "png"


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
        flatfield_correction=False,
        min_perimeter=0,
        padding=1,
        roi_encoding="raw",
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
        flatfield_correction=False,
        min_perimeter=15,
        roi_encoding="raw",
    ) == []


def test_store_roi_auto_uses_png_for_small_roi_payloads():
    source = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=np.full((4, 4), 10, dtype=np.uint8),
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": FRAME_ID,
        },
    )
    roi = FrameData(
        sourcePath="/tmp/",
        filename="frame.png",
        frameNumber=7,
        data=np.full((2, 2), 255, dtype=np.uint8),
        width=2,
        height=2,
        bbox_x=1,
        bbox_y=1,
        parent_frame_id=FRAME_ID,
    )
    contour = np.array([[[1, 1]], [[2, 1]], [[2, 2]], [[1, 2]]], dtype=np.int32)

    detection = store_roi(
        roi,
        source_frame=source,
        roi_index=1,
        contour=contour,
        area=4,
        encoding="auto",
        zstd_min_bytes=100,
    )

    assert detection.roi_payload.startswith(b"\x89PNG\r\n\x1a\n")
    assert detection.roi_encoding == "png"
    assert detection.roi_format == "png"
