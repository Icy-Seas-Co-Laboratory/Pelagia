"""Convert candidate ROIs into persistable detection records and payloads."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..domain import DetectionRecord, normalize_collections
from .defaults import default_processing_config
from .frame_codec import encode_array_payload
from .frame_model import FrameData
from .mask_augmentation import as_binary_mask
from .roi_assembly import CandidateRoi
from .timing import measure_phase


def frame_metadata_value(frame: FrameData, key: str, default: Any = None) -> Any:
    """Read a value from a FrameData attribute or metadata dictionary."""
    if hasattr(frame, key):
        value = getattr(frame, key)
        if value is not None:
            return value
    return frame.metadata.get(key, default)


def build_candidate_detection_record(
    candidate: CandidateRoi,
    *,
    source_frame: FrameData,
    processed_frame: FrameData,
    roi_index: int | None = None,
    padding: int = 0,
    encoding: str | None = None,
    zstd_min_bytes: int | None = None,
    store_roi_payload: bool = True,
    always_store_mask: bool = True,
    extra_metadata: dict[str, Any] | None = None,
) -> DetectionRecord:
    """Build a candidate DetectionRecord from an assembled ROI."""
    processed_data = processed_frame.read()
    if processed_data is None:
        raise ValueError("Processed frame has no image data to record.")

    image = np.ascontiguousarray(processed_data)
    if image.ndim < 2:
        raise ValueError("Processed frame data must have at least two dimensions.")
    image_height, image_width = image.shape[:2]

    crop_x0, crop_y0, crop_x1, crop_y1 = padded_bounds(
        candidate.bbox_x,
        candidate.bbox_y,
        candidate.bbox_w,
        candidate.bbox_h,
        padding,
        image_width,
        image_height,
    )
    object_array = np.ascontiguousarray(
        image[
            candidate.bbox_y:candidate.bbox_y + candidate.bbox_h,
            candidate.bbox_x:candidate.bbox_x + candidate.bbox_w,
        ]
    )
    object_mask = as_binary_mask(candidate.mask)
    if object_mask.shape[:2] != object_array.shape[:2]:
        object_mask = candidate_mask_crop(
            candidate,
            crop_x0=candidate.bbox_x,
            crop_y0=candidate.bbox_y,
            crop_x1=candidate.bbox_x + candidate.bbox_w,
            crop_y1=candidate.bbox_y + candidate.bbox_h,
        )
    roi_array = (
        np.ascontiguousarray(image[crop_y0:crop_y1, crop_x0:crop_x1])
        if store_roi_payload
        else None
    )
    mask_crop = (
        candidate_mask_crop(
            candidate,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            crop_x1=crop_x1,
            crop_y1=crop_y1,
        )
        if always_store_mask
        else None
    )

    frame_id = frame_metadata_value(source_frame, "frame_id")
    run_id = frame_metadata_value(source_frame, "run_id")
    if frame_id is None:
        raise ValueError("Source frame metadata must include frame_id before storing ROIs.")
    if run_id is None:
        raise ValueError("Source frame metadata must include run_id before storing ROIs.")

    encoding_reference = roi_array if roi_array is not None else object_array
    selected_encoding = choose_roi_encoding(encoding_reference, encoding, zstd_min_bytes)
    if store_roi_payload:
        if roi_array is None:
            raise RuntimeError("ROI payload was requested but no ROI array was prepared.")
        with measure_phase("segmentation.roi_encode"):
            payload, array_encoding, array_format = encode_array_payload(
                roi_array,
                selected_encoding,
            )
        roi_dtype = str(roi_array.dtype)
        roi_shape = list(roi_array.shape)
    else:
        payload = None
        array_encoding = None
        array_format = None
        roi_dtype = None
        roi_shape = []

    if always_store_mask:
        if mask_crop is None:
            raise RuntimeError("Mask payload was requested but no mask crop was prepared.")
        with measure_phase("segmentation.mask_encode"):
            mask_payload, mask_encoding, mask_format = encode_array_payload(
                mask_crop,
                selected_encoding,
            )
        mask_dtype = str(mask_crop.dtype)
        mask_shape = list(mask_crop.shape)
    else:
        mask_payload = None
        mask_encoding = None
        mask_format = None
        mask_dtype = None
        mask_shape = []

    min_gray, mean_gray = component_gray_stats(object_array, object_mask)
    major_axis, minor_axis = component_axis_lengths(
        candidate.contour,
        candidate.bbox_w,
        candidate.bbox_h,
    )

    # Cropped/preprocessed frames carry their original-frame offset in bbox_x/y.
    frame_x_offset = int(getattr(processed_frame, "bbox_x", 0) or 0)
    frame_y_offset = int(getattr(processed_frame, "bbox_y", 0) or 0)
    object_bbox = (
        frame_x_offset + candidate.bbox_x,
        frame_y_offset + candidate.bbox_y,
        candidate.bbox_w,
        candidate.bbox_h,
    )
    crop_bbox = (
        frame_x_offset + crop_x0,
        frame_y_offset + crop_y0,
        int(crop_x1 - crop_x0),
        int(crop_y1 - crop_y0),
    )

    metadata = dict(extra_metadata or {})
    metadata.update(candidate.metadata)
    metadata.update(
        {
            "collections": normalize_collections(source_frame.metadata.get("collections")),
            "source_frame_number": source_frame.frameNumber,
            "parent_frame_id": frame_id,
            "source_frame_type": processed_frame.frameType,
            "detection_stage": "candidate",
            "mask_kind": "candidate",
            "foreground_polarity": "bright",
            "object_bbox": object_bbox,
            "roi_bbox": crop_bbox,
            "padding": int(max(0, padding)),
            "actual_padding": {
                "left": int(candidate.bbox_x - crop_x0),
                "top": int(candidate.bbox_y - crop_y0),
                "right": int(crop_x1 - (candidate.bbox_x + candidate.bbox_w)),
                "bottom": int(crop_y1 - (candidate.bbox_y + candidate.bbox_h)),
            },
            "roi_payload_stored": bool(store_roi_payload),
            "mask_payload_stored": bool(mask_payload is not None),
        }
    )

    return DetectionRecord(
        run_id=str(run_id),
        frame_id=str(frame_id),
        roi_index=int(roi_index or candidate.roi_index),
        bbox_x=int(object_bbox[0]),
        bbox_y=int(object_bbox[1]),
        bbox_w=int(object_bbox[2]),
        bbox_h=int(object_bbox[3]),
        area=float(candidate.area),
        perimeter=float(candidate.perimeter),
        major_axis_length=major_axis,
        minor_axis_length=minor_axis,
        min_gray_value=min_gray,
        mean_gray_value=mean_gray,
        roi_payload=payload,
        mask_payload=mask_payload,
        crop_bbox_x=int(crop_bbox[0]),
        crop_bbox_y=int(crop_bbox[1]),
        crop_bbox_w=int(crop_bbox[2]),
        crop_bbox_h=int(crop_bbox[3]),
        roi_encoding=array_encoding,
        roi_format=array_format,
        roi_dtype=roi_dtype,
        roi_shape=roi_shape,
        mask_encoding=mask_encoding,
        mask_format=mask_format,
        mask_dtype=mask_dtype,
        mask_shape=mask_shape,
        metadata=metadata,
    )


def choose_roi_encoding(roi: np.ndarray, encoding: str | None, zstd_min_bytes: int | None) -> str:
    defaults = default_processing_config().roi_recording
    requested = str(encoding or defaults.roi_encoding).lower()
    resolved_zstd_min_bytes = defaults.zstd_min_bytes if zstd_min_bytes is None else zstd_min_bytes
    if requested == "auto":
        # Small crops favor image compression; larger arrays favor fast lossless zstd.
        return "png" if roi.nbytes < resolved_zstd_min_bytes else "zstd"
    return requested


def candidate_mask_crop(
    candidate: CandidateRoi,
    *,
    crop_x0: int,
    crop_y0: int,
    crop_x1: int,
    crop_y1: int,
) -> np.ndarray:
    """Return a crop-sized mask from a candidate's bbox-local mask."""
    crop_width = int(crop_x1 - crop_x0)
    crop_height = int(crop_y1 - crop_y0)
    if crop_width < 1 or crop_height < 1:
        raise ValueError("Mask crop bounds must produce a positive crop.")

    local_mask = as_binary_mask(candidate.mask)
    mask_x0 = int(getattr(candidate, "mask_x", candidate.bbox_x))
    mask_y0 = int(getattr(candidate, "mask_y", candidate.bbox_y))
    mask_x1 = mask_x0 + int(local_mask.shape[1])
    mask_y1 = mask_y0 + int(local_mask.shape[0])

    intersect_x0 = max(int(crop_x0), mask_x0)
    intersect_y0 = max(int(crop_y0), mask_y0)
    intersect_x1 = min(int(crop_x1), mask_x1)
    intersect_y1 = min(int(crop_y1), mask_y1)

    mask_crop = np.zeros((crop_height, crop_width), dtype=np.uint8)
    if intersect_x1 <= intersect_x0 or intersect_y1 <= intersect_y0:
        return np.ascontiguousarray(mask_crop)

    source_x0 = intersect_x0 - mask_x0
    source_y0 = intersect_y0 - mask_y0
    source_x1 = intersect_x1 - mask_x0
    source_y1 = intersect_y1 - mask_y0
    dest_x0 = intersect_x0 - int(crop_x0)
    dest_y0 = intersect_y0 - int(crop_y0)
    dest_x1 = intersect_x1 - int(crop_x0)
    dest_y1 = intersect_y1 - int(crop_y0)
    mask_crop[dest_y0:dest_y1, dest_x0:dest_x1] = local_mask[
        source_y0:source_y1,
        source_x0:source_x1,
    ]
    return np.ascontiguousarray(mask_crop)


def padded_bounds(
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


def component_axis_lengths(contour: np.ndarray, width: int, height: int) -> tuple[float, float]:
    if len(contour) >= 5:
        try:
            (_, _), axes, _ = cv2.fitEllipse(contour)
            major, minor = sorted((float(axes[0]), float(axes[1])), reverse=True)
            return major, minor
        except cv2.error:
            pass
    return float(max(width, height)), float(min(width, height))


def component_gray_stats(gray_crop: np.ndarray, mask: np.ndarray) -> tuple[int, float]:
    pixels = gray_crop[mask > 0]
    if pixels.size == 0:
        return 0, 0.0
    return int(np.min(pixels)), float(np.mean(pixels))
