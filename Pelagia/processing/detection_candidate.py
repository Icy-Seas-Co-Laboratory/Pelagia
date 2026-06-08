from __future__ import annotations

import time
from typing import Any, Callable

import cv2
import numpy as np

from ..domain import DetectionRecord, FrameRecord, normalize_collections
from ._logging import log_processing_event, processing_core_logger
from .defaults import default_processing_config
from .frame_codec import encode_array_payload
from .frame_model import FrameData
from .frame_preprocess import as_grayscale_array, preprocess_frame_for_segmentation
from .frame_threshold import threshold_manual, threshold_otsu


ThresholdFn = Callable[[np.ndarray], np.ndarray]
_CORE_LOGGER = processing_core_logger("detection_candidate")


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
    return as_grayscale_array(data)


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


def _choose_roi_encoding(roi: np.ndarray, encoding: str | None, zstd_min_bytes: int | None) -> str:
    defaults = default_processing_config().segmentation
    requested = str(encoding or defaults.roi_encoding).lower()
    resolved_zstd_min_bytes = defaults.zstd_min_bytes if zstd_min_bytes is None else zstd_min_bytes
    if requested == "auto":
        return "png" if roi.nbytes < resolved_zstd_min_bytes else "zstd"
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
    encoding: str | None = None,
    zstd_min_bytes: int | None = None,
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
            "collections": normalize_collections(source_frame.metadata.get("collections")),
            "source_frame_number": source_frame.frameNumber,
            "parent_frame_id": frame_id,
        }
    )

    return DetectionRecord(
        run_id=str(run_id),
        frame_id=str(frame_id),
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


def live_segment_wrapper(
    frame_id: str,
    *,
    threshold: int | float | ThresholdFn | None = None,
    frame_payload_kind: str = "original",
    apply_preprocessing: bool | None = None,
    apply_mask: bool | None = None,
    crop_enabled: bool | None = None,
    crop_x: int | None = None,
    crop_y: int | None = None,
    crop_w: int | None = None,
    crop_h: int | None = None,
    flatfield_correction: bool | None = None,
    flatfield_q: float | None = None,
    flatfield_axis: int | None = None,
    background_correction: bool | None = None,
    background_percentile: int | float | None = None,
    invert_intensity: bool | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    padding: int | None = None,
    roi_encoding: str | None = "png",
    zstd_min_bytes: int | None = None,
    context: Any = None,
) -> list[DetectionRecord]:
    """Load one frame, segment it, and return transient detection records."""
    from .frame_store import retrieve_frame

    started = time.perf_counter()
    log_processing_event(
        context,
        "debug",
        "segmentation.live_started",
        "Live frame segmentation started",
        payload={"frame_id": frame_id},
        logger="pelagia.processing.detection_candidate",
        core_logger=_CORE_LOGGER,
    )
    try:
        frame = retrieve_frame(frame_id, context=context, payload_kind=frame_payload_kind)
    except TypeError as exc:
        if "payload_kind" not in str(exc):
            raise
        frame = retrieve_frame(frame_id, context=context)
    frame_record = None
    if context is not None and getattr(context, "repository", None) is not None:
        frame_record = context.repository.get_frame_record(frame_id)
    try:
        detections = segment_frame(
            frame=frame,
            frame_record=frame_record,
            threshold=threshold,
            apply_preprocessing=(
                frame_payload_kind in {"original", "raw"}
                if apply_preprocessing is None
                else apply_preprocessing
            ),
            flatfield_correction=flatfield_correction,
            flatfield_q=flatfield_q,
            flatfield_axis=flatfield_axis,
            apply_mask=apply_mask,
            crop_enabled=crop_enabled,
            crop_x=crop_x,
            crop_y=crop_y,
            crop_w=crop_w,
            crop_h=crop_h,
            background_correction=background_correction,
            background_percentile=background_percentile,
            invert_intensity=invert_intensity,
            min_perimeter=min_perimeter,
            max_perimeter=max_perimeter,
            padding=padding,
            roi_encoding=roi_encoding,
            zstd_min_bytes=zstd_min_bytes,
            context=context,
        )
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        _CORE_LOGGER.exception("Live frame segmentation failed frame_id=%s", frame_id)
        log_processing_event(
            context,
            "error",
            "segmentation.live_failed",
            "Live frame segmentation failed",
            run_id=_frame_metadata_value(frame, "run_id"),
            asset_id=_frame_metadata_value(frame, "asset_id"),
            duration_ms=duration_ms,
            payload={
                "frame_id": frame_id,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            logger="pelagia.processing.detection_candidate",
            core_logger=_CORE_LOGGER,
        )
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    log_processing_event(
        context,
        "info",
        "segmentation.live_completed",
        "Live frame segmentation completed",
        run_id=_frame_metadata_value(frame, "run_id"),
        asset_id=_frame_metadata_value(frame, "asset_id"),
        duration_ms=duration_ms,
        payload={"frame_id": frame_id, "detection_count": len(detections)},
        logger="pelagia.processing.detection_candidate",
        core_logger=_CORE_LOGGER,
    )
    return detections


def segment_frame(
    frame: FrameData,
    *,
    frame_record: FrameRecord | None = None,
    threshold: int | float | ThresholdFn | None = None,
    apply_preprocessing: bool = True,
    apply_mask: bool | None = None,
    crop_enabled: bool | None = None,
    crop_x: int | None = None,
    crop_y: int | None = None,
    crop_w: int | None = None,
    crop_h: int | None = None,
    flatfield_correction: bool | None = None,
    flatfield_q: float | None = None,
    flatfield_axis: int | None = None,
    background_correction: bool | None = None,
    background: np.ndarray | int | float | None = None,
    background_percentile: int | float | None = None,
    invert_intensity: bool | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    padding: int | None = None,
    roi_encoding: str | None = None,
    zstd_min_bytes: int | None = None,
    context: Any = None,
) -> list[DetectionRecord]:
    """Segment one frame into connected-component ROI detection records."""
    started = time.perf_counter()
    defaults = default_processing_config().segmentation
    preprocessing_defaults = (
        context.config.processing.preprocessing
        if context is not None and getattr(context, "config", None) is not None
        else default_processing_config().preprocessing
    )
    thresholding_defaults = (
        context.config.processing.thresholding
        if context is not None and getattr(context, "config", None) is not None
        else default_processing_config().thresholding
    )
    flatfield_defaults = (
        context.config.processing.flatfield
        if context is not None and getattr(context, "config", None) is not None
        else default_processing_config().flatfield
    )
    resolved_flatfield_correction = (
        flatfield_defaults.flatfield_correction
        if flatfield_correction is None
        else flatfield_correction
    )
    resolved_flatfield_q = flatfield_defaults.flatfield_q if flatfield_q is None else flatfield_q
    resolved_flatfield_axis = (
        flatfield_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis
    )
    resolved_apply_mask = preprocessing_defaults.apply_mask if apply_mask is None else apply_mask
    resolved_crop_enabled = (
        preprocessing_defaults.crop_enabled if crop_enabled is None else crop_enabled
    )
    resolved_crop_x = preprocessing_defaults.crop_x if crop_x is None else crop_x
    resolved_crop_y = preprocessing_defaults.crop_y if crop_y is None else crop_y
    resolved_crop_w = preprocessing_defaults.crop_w if crop_w is None else crop_w
    resolved_crop_h = preprocessing_defaults.crop_h if crop_h is None else crop_h
    resolved_background_correction = (
        preprocessing_defaults.background_correction
        if background_correction is None
        else background_correction
    )
    resolved_background_percentile = (
        preprocessing_defaults.background_percentile
        if background_percentile is None
        else background_percentile
    )
    resolved_invert_intensity = (
        preprocessing_defaults.invert_intensity if invert_intensity is None else invert_intensity
    )
    resolved_min_perimeter = defaults.min_perimeter if min_perimeter is None else min_perimeter
    resolved_max_perimeter = defaults.max_perimeter if max_perimeter is None else max_perimeter
    resolved_padding = defaults.padding if padding is None else padding
    resolved_roi_encoding = defaults.roi_encoding if roi_encoding is None else roi_encoding
    resolved_zstd_min_bytes = defaults.zstd_min_bytes if zstd_min_bytes is None else zstd_min_bytes

    frame_id = _frame_metadata_value(frame, "frame_id")
    run_id = _frame_metadata_value(frame, "run_id")
    asset_id = _frame_metadata_value(frame, "asset_id")
    payload_base = {
        "frame_id": frame_id,
        "filename": frame.filename,
        "frame_number": frame.frameNumber,
        "tile_number": frame.tileNumber,
        "flatfield_correction": bool(resolved_flatfield_correction),
        "apply_preprocessing": bool(apply_preprocessing),
        "flatfield_q": resolved_flatfield_q,
        "flatfield_axis": resolved_flatfield_axis,
        "background_correction": bool(resolved_background_correction),
        "background_percentile": resolved_background_percentile,
        "intensity_inverted": bool(resolved_invert_intensity),
        "apply_mask": bool(resolved_apply_mask),
        "crop_enabled": bool(resolved_crop_enabled),
        "crop_x": resolved_crop_x,
        "crop_y": resolved_crop_y,
        "crop_w": resolved_crop_w,
        "crop_h": resolved_crop_h,
        "min_perimeter": resolved_min_perimeter,
        "max_perimeter": resolved_max_perimeter,
        "padding": resolved_padding,
        "roi_encoding": resolved_roi_encoding,
    }
    log_processing_event(
        context,
        "debug",
        "segmentation.frame_started",
        "Frame segmentation started",
        run_id=run_id,
        asset_id=asset_id,
        payload=payload_base,
        logger="pelagia.processing.detection_candidate",
        core_logger=_CORE_LOGGER,
    )
    try:
        if apply_preprocessing:
            frame = preprocess_frame_for_segmentation(
                frame,
                flatfield_correction=resolved_flatfield_correction,
                flatfield_q=resolved_flatfield_q,
                flatfield_axis=resolved_flatfield_axis,
                apply_mask=resolved_apply_mask,
                crop_enabled=resolved_crop_enabled,
                crop_x=resolved_crop_x,
                crop_y=resolved_crop_y,
                crop_w=resolved_crop_w,
                crop_h=resolved_crop_h,
                background_correction=resolved_background_correction,
                background=background,
                background_percentile=resolved_background_percentile,
                invert_intensity=resolved_invert_intensity,
                context=context,
            )
        gray = _as_grayscale_array(frame)
        if callable(threshold):
            thresh = threshold(gray)
        elif threshold is None:
            thresh = threshold_otsu(
                gray,
                thresholding_maximum_value=thresholding_defaults.thresholding_maximum_value,
            )
        else:
            thresh = threshold_manual(gray, threshold)
        thresh = _as_binary_mask(thresh)

        num, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        roi_records: list[DetectionRecord] = []
        frame_height, frame_width = gray.shape[:2]

        for lab in range(1, num):
            x, y, width, height, area = stats[lab]
            bbox_perimeter = 2 * int(width) + 2 * int(height)
            if bbox_perimeter < resolved_min_perimeter:
                continue
            if resolved_max_perimeter is not None and bbox_perimeter > resolved_max_perimeter:
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
                resolved_padding,
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
                    "detection_stage": "candidate",
                    "mask_kind": "candidate",
                    "foreground_polarity": "bright",
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
                    "padding": int(resolved_padding),
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
                    encoding=resolved_roi_encoding,
                    zstd_min_bytes=resolved_zstd_min_bytes,
                    extra_metadata=roi_frame.metadata,
                )
            )
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        _CORE_LOGGER.exception("Frame segmentation failed frame_id=%s run_id=%s asset_id=%s", frame_id, run_id, asset_id)
        log_processing_event(
            context,
            "error",
            "segmentation.frame_failed",
            "Frame segmentation failed",
            run_id=run_id,
            asset_id=asset_id,
            duration_ms=duration_ms,
            payload={
                **payload_base,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            logger="pelagia.processing.detection_candidate",
            core_logger=_CORE_LOGGER,
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000
    _CORE_LOGGER.debug(
        "Segmented frame frame_id=%s run_id=%s asset_id=%s detections=%s duration_ms=%.2f",
        frame_id,
        run_id,
        asset_id,
        len(roi_records),
        duration_ms,
    )
    log_processing_event(
        context,
        "info",
        "segmentation.frame_completed",
        "Frame segmentation completed",
        run_id=run_id,
        asset_id=asset_id,
        duration_ms=duration_ms,
        payload={
            **payload_base,
            "source_component_count": max(0, num - 1),
            "detection_count": len(roi_records),
            "frame_shape": list(gray.shape),
        },
        logger="pelagia.processing.detection_candidate",
        core_logger=_CORE_LOGGER,
    )
    return roi_records
