from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from ..domain import DetectionRecord
from .frame_codec import encode_array_payload
from .frame_model import FrameData


ThresholdFn = Callable[[np.ndarray], np.ndarray]


def _frame_metadata_value(frame: FrameData, key: str, default: Any = None) -> Any:
    if hasattr(frame, key):
        value = getattr(frame, key)
        if value is not None:
            return value
    return frame.metadata.get(key, default)


def _as_grayscale_array(frame: FrameData) -> np.ndarray:
    data = frame.read()
    if data is None:
        raise ValueError("Frame has no image data to segment.")

    array = np.asarray(data)
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D grayscale or 3D color frame, got shape {array.shape}.")

    channels = array.shape[2]
    if channels == 1:
        return np.ascontiguousarray(array[:, :, 0])
    if channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    raise ValueError(f"Expected frame with 1, 3, or 4 channels, got {channels}.")


def calc_threshold(gray: np.ndarray, threshold: int | float | None = None) -> np.ndarray:
    """Create an 8-bit binary segmentation mask from a grayscale frame."""
    if gray.ndim != 2:
        raise ValueError("Thresholding expects a 2D grayscale image.")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    if threshold is None:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    else:
        _, binary = cv2.threshold(gray, float(threshold), 255, cv2.THRESH_BINARY)
    return np.ascontiguousarray(binary)


def _component_axis_lengths(contour: np.ndarray, width: int, height: int) -> tuple[float, float]:
    if len(contour) >= 5:
        try:
            (_, _), axes, _ = cv2.fitEllipse(contour)
            major, minor = sorted((float(axes[0]), float(axes[1])), reverse=True)
            return major, minor
        except cv2.error:
            pass
    return float(max(width, height)), float(min(width, height))


def _component_gray_stats(gray_crop: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    pixels = gray_crop[mask > 0]
    if pixels.size == 0:
        return 0, 0.0
    return int(np.min(pixels)), float(np.mean(pixels))


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Segmentation mask must be a 2D image.")
    return np.ascontiguousarray((array > 0).astype(np.uint8) * 255)


def _choose_roi_encoding(roi: np.ndarray, encoding: str | None, zstd_min_bytes: int) -> str:
    requested = str(encoding or "zstd").lower()
    if requested == "auto":
        return "png" if roi.nbytes < zstd_min_bytes else "zstd"
    return requested


def _padded_bounds(
    x: int,
    y: int,
    width: int,
    height: int,
    padding: int,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    pad = max(0, int(padding))
    x0 = max(0, int(x) - pad)
    y0 = max(0, int(y) - pad)
    x1 = min(image_width, int(x) + int(width) + pad)
    y1 = min(image_height, int(y) + int(height) + pad)
    return x0, y0, x1, y1


def store_roi(
    roi_frame: FrameData,
    *,
    source_frame: FrameData,
    roi_index: int,
    contour: np.ndarray,
    area: float,
    encoding: str | None = "zstd",
    zstd_min_bytes: int = 1024,
    extra_metadata: dict[str, Any] | None = None,
) -> DetectionRecord:
    """
    Build a DetectionRecord for an ROI, storing encoded ROI bytes in Postgres-ready form.
    """
    roi_data = roi_frame.read()
    if roi_data is None:
        raise ValueError("ROI frame has no image data to store.")

    roi_array = np.ascontiguousarray(roi_data)
    if roi_array.ndim < 2:
        raise ValueError("ROI data must have at least two dimensions.")
    roi_frame.validate_geometry(roi_array)
    roi_frame.validate_mask()

    selected_encoding = _choose_roi_encoding(roi_array, encoding, zstd_min_bytes)
    payload, array_encoding, array_format = encode_array_payload(roi_array, selected_encoding)

    frame_id = _frame_metadata_value(source_frame, "frame_id")
    run_id = _frame_metadata_value(source_frame, "run_id")
    if frame_id is None:
        raise ValueError("Source frame metadata must include frame_id before storing ROIs.")
    if run_id is None:
        raise ValueError("Source frame metadata must include run_id before storing ROIs.")

    if roi_frame.mask is not None:
        mask_array = _as_binary_mask(roi_frame.mask)
    else:
        mask_array = _as_binary_mask(roi_array)
    mask_payload, mask_encoding, mask_format = encode_array_payload(mask_array, selected_encoding)

    min_gray, mean_gray = _component_gray_stats(roi_array, mask_array)
    major_axis, minor_axis = _component_axis_lengths(contour, roi_frame.width, roi_frame.height)

    metadata = dict(extra_metadata or {})
    bbox = metadata.get("object_bbox") or roi_frame.get_bbox()
    bbox_x, bbox_y, bbox_w, bbox_h = bbox
    crop_bbox = metadata.get("roi_bbox") or roi_frame.get_bbox()
    crop_bbox_x, crop_bbox_y, crop_bbox_w, crop_bbox_h = crop_bbox

    metadata.update(
        {
            "source_frame_number": source_frame.frameNumber,
            "parent_frame_id": frame_id,
        }
    )

    return DetectionRecord(
        run_id=str(run_id),
        frame_id=int(frame_id),
        roi_index=int(roi_index),
        bbox_x=int(bbox_x),
        bbox_y=int(bbox_y),
        bbox_w=int(bbox_w),
        bbox_h=int(bbox_h),
        area=float(area),
        perimeter=float(cv2.arcLength(contour, True)),
        major_axis_length=major_axis,
        minor_axis_length=minor_axis,
        min_gray_value=min_gray,
        mean_gray_value=mean_gray,
        roi_payload=payload,
        mask_payload=mask_payload,
        crop_bbox_x=int(crop_bbox_x),
        crop_bbox_y=int(crop_bbox_y),
        crop_bbox_w=int(crop_bbox_w),
        crop_bbox_h=int(crop_bbox_h),
        roi_encoding=array_encoding,
        roi_format=array_format,
        roi_dtype=str(roi_array.dtype),
        roi_shape=list(roi_array.shape),
        mask_encoding=mask_encoding,
        mask_format=mask_format,
        mask_dtype=str(mask_array.dtype),
        mask_shape=list(mask_array.shape),
        metadata=metadata,
    )


def segment_frame(
    frame: FrameData,
    *,
    threshold: int | float | ThresholdFn | None = None,
    min_perimeter: int | float = 0,
    max_perimeter: int | float | None = None,
    padding: int = 0,
    roi_encoding: str | None = "zstd",
    zstd_min_bytes: int = 1024,
) -> list[DetectionRecord]:
    """Segment one frame into connected-component ROI detection records."""
    gray = _as_grayscale_array(frame)
    thresh = threshold(gray) if callable(threshold) else calc_threshold(gray, threshold)
    thresh = _as_binary_mask(thresh)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    roi_records: list[DetectionRecord] = []
    frame_height, frame_width = gray.shape[:2]

    for lab in range(1, num):
        x, y, width, height, area = stats[lab]
        bbox_perimeter = 2 * int(width) + 2 * int(height)
        if bbox_perimeter < min_perimeter:
            continue
        if max_perimeter is not None and bbox_perimeter > max_perimeter:
            continue

        roi_labels = labels[y : y + height, x : x + width]
        component = np.ascontiguousarray((roi_labels == lab).astype(np.uint8) * 255)
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        crop_x0, crop_y0, crop_x1, crop_y1 = _padded_bounds(
            int(x),
            int(y),
            int(width),
            int(height),
            padding,
            frame_width,
            frame_height,
        )
        padded_labels = labels[crop_y0:crop_y1, crop_x0:crop_x1]
        padded_mask = np.ascontiguousarray((padded_labels == lab).astype(np.uint8) * 255)
        padded_crop = np.ascontiguousarray(gray[crop_y0:crop_y1, crop_x0:crop_x1])

        contour = max(contours, key=cv2.contourArea)
        contour = contour.copy()
        contour[:, 0, 0] += int(x) + frame.bbox_x
        contour[:, 0, 1] += int(y) + frame.bbox_y

        roi_frame = FrameData(
            sourcePath=frame.sourcePath,
            destPath=frame.destPath,
            filename=frame.filename,
            frameNumber=frame.frameNumber,
            data=padded_crop,
            mask=padded_mask,
            width=int(crop_x1 - crop_x0),
            height=int(crop_y1 - crop_y0),
            bbox_x=frame.bbox_x + crop_x0,
            bbox_y=frame.bbox_y + crop_y0,
            parent_frame_id=_frame_metadata_value(frame, "frame_id"),
            tileNumber=frame.tileNumber,
            sourceFrameStart=frame.sourceFrameStart,
            sourceFrameEnd=frame.sourceFrameEnd,
            frameType="roi",
            channel=frame.channel,
            timestamp=frame.timestamp,
            metadata={
                "source_frame_type": frame.frameType,
                "component_label": int(lab),
                "object_bbox": (
                    frame.bbox_x + int(x),
                    frame.bbox_y + int(y),
                    int(width),
                    int(height),
                ),
                "roi_bbox": (
                    frame.bbox_x + crop_x0,
                    frame.bbox_y + crop_y0,
                    int(crop_x1 - crop_x0),
                    int(crop_y1 - crop_y0),
                ),
                "padding": int(padding),
                "actual_padding": {
                    "left": int(x) - crop_x0,
                    "top": int(y) - crop_y0,
                    "right": crop_x1 - (int(x) + int(width)),
                    "bottom": crop_y1 - (int(y) + int(height)),
                },
            },
        )

        roi_records.append(
            store_roi(
                roi_frame,
                source_frame=frame,
                roi_index=len(roi_records) + 1,
                contour=contour,
                area=float(area),
                encoding=roi_encoding,
                zstd_min_bytes=zstd_min_bytes,
                extra_metadata=roi_frame.metadata,
            )
        )

    return roi_records
