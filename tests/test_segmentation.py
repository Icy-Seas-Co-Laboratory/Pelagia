import numpy as np

from Pelagia.processing.frame_codec import decode_array_payload
from Pelagia.processing.frame_model import FrameData
from Pelagia.processing.segmentation import segment_frame, store_roi


def test_segment_frame_returns_roi_detection_records_with_raw_payload():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        destPath="/tmp/out",
        filename="frame.png",
        frameNumber=7,
        data=data,
        bbox_x=100,
        bbox_y=200,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": 42,
        },
    )

    detections = segment_frame(frame, threshold=1, roi_encoding="raw")

    assert len(detections) == 1
    detection = detections[0]
    assert detection.run_id == "00000000-0000-0000-0000-000000000001"
    assert detection.frame_id == 42
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


def test_segment_frame_stores_padded_roi_context_and_mask():
    data = np.zeros((10, 10), dtype=np.uint8)
    data[2:5, 3:7] = 50
    frame = FrameData(
        sourcePath="/tmp/",
        destPath="/tmp/out",
        filename="frame.png",
        frameNumber=7,
        data=data,
        bbox_x=100,
        bbox_y=200,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": 42,
        },
    )

    detections = segment_frame(frame, threshold=1, padding=1, roi_encoding="raw")

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
        destPath="/tmp/out",
        filename="frame.png",
        frameNumber=7,
        data=data,
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": 42,
        },
    )

    assert segment_frame(frame, threshold=1, min_perimeter=15, roi_encoding="raw") == []


def test_store_roi_auto_uses_png_for_small_roi_payloads():
    source = FrameData(
        sourcePath="/tmp/",
        destPath="/tmp/out",
        filename="frame.png",
        frameNumber=7,
        data=np.full((4, 4), 10, dtype=np.uint8),
        metadata={
            "run_id": "00000000-0000-0000-0000-000000000001",
            "frame_id": 42,
        },
    )
    roi = FrameData(
        sourcePath="/tmp/",
        destPath="/tmp/out",
        filename="frame.png",
        frameNumber=7,
        data=np.full((2, 2), 255, dtype=np.uint8),
        width=2,
        height=2,
        bbox_x=1,
        bbox_y=1,
        parent_frame_id=42,
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
