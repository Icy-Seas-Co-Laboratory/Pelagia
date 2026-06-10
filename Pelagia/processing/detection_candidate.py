from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np

from ..domain import DetectionRecord, FrameRecord
from ._logging import log_processing_event, processing_core_logger
from .defaults import default_processing_config
from .detection_recording import (
    build_candidate_detection_record,
    frame_metadata_value,
)
from .frame_model import FrameData
from .frame_preprocess import as_grayscale_array, preprocess_frame_for_segmentation
from .frame_threshold import calculate_threshold_mask
from .mask_augmentation import as_binary_mask, augment_mask
from .roi_assembly import assemble_candidate_rois
from .roi_filter import filter_candidate_rois, should_store_roi_payload


ThresholdFn = Callable[[np.ndarray], np.ndarray]
_CORE_LOGGER = processing_core_logger("detection_candidate")


def _frame_metadata_value(frame: FrameData, key: str, default: Any = None) -> Any:
    return frame_metadata_value(frame, key, default)


def _as_grayscale_array(frame: FrameData) -> np.ndarray:
    data = frame.read()
    if data is None:
        raise ValueError("Frame has no image data to segment.")
    return as_grayscale_array(data)


def live_segment_wrapper(
    frame_id: str,
    *,
    threshold: int | float | ThresholdFn | None = None,
    threshold_method: str | None = None,
    manual_threshold: int | float | None = None,
    thresholding_maximum_value: int | float | None = None,
    bounded_otsu_min_contrast: int | float | None = None,
    bounded_otsu_max_foreground_fraction: float | None = None,
    canny_enabled: bool | None = None,
    canny_low_threshold: int | float | None = None,
    canny_high_threshold: int | float | None = None,
    canny_blur_kernel: int | None = None,
    dilate_kernel_w: int | None = None,
    dilate_kernel_h: int | None = None,
    dilate_iterations: int | None = None,
    erode_kernel_w: int | None = None,
    erode_kernel_h: int | None = None,
    erode_iterations: int | None = None,
    open_kernel_w: int | None = None,
    open_kernel_h: int | None = None,
    open_iterations: int | None = None,
    close_kernel_w: int | None = None,
    close_kernel_h: int | None = None,
    close_iterations: int | None = None,
    fill_holes: bool | None = None,
    remove_small_components: bool | None = None,
    min_component_area: int | float | None = None,
    clear_border: bool | None = None,
    adaptive_block_size: int | None = None,
    adaptive_c: int | float | None = None,
    percentile_background_percentile: int | float | None = None,
    percentile_min_contrast: int | float | None = None,
    hysteresis_low_threshold: int | float | None = None,
    hysteresis_high_threshold: int | float | None = None,
    hysteresis_connectivity: int | None = None,
    sobel_percentile: int | float | None = None,
    sobel_threshold: int | float | None = None,
    sobel_kernel_size: int | None = None,
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
    mask_augmentation_enabled: bool | None = None,
    mask_augmentation_steps: list[str] | tuple[str, ...] | None = None,
    roi_assembly_method: str | None = None,
    roi_assembly_connectivity: int | None = None,
    min_area: int | float | None = None,
    max_area: int | float | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    min_width: int | float | None = None,
    max_width: int | float | None = None,
    min_height: int | float | None = None,
    max_height: int | float | None = None,
    min_width_plus_height: int | float | None = None,
    max_width_plus_height: int | float | None = None,
    padding: int | None = None,
    roi_encoding: str | None = "png",
    zstd_min_bytes: int | None = None,
    store_roi_payload_min_area: int | float | None = None,
    store_roi_payload_min_width: int | float | None = None,
    store_roi_payload_min_height: int | float | None = None,
    store_roi_payload_min_width_plus_height: int | float | None = None,
    always_store_mask: bool | None = None,
    encode_payloads: bool = True,
    max_detections: int | None = None,
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
            threshold_method=threshold_method,
            manual_threshold=manual_threshold,
            thresholding_maximum_value=thresholding_maximum_value,
            bounded_otsu_min_contrast=bounded_otsu_min_contrast,
            bounded_otsu_max_foreground_fraction=bounded_otsu_max_foreground_fraction,
            canny_enabled=canny_enabled,
            canny_low_threshold=canny_low_threshold,
            canny_high_threshold=canny_high_threshold,
            canny_blur_kernel=canny_blur_kernel,
            dilate_kernel_w=dilate_kernel_w,
            dilate_kernel_h=dilate_kernel_h,
            dilate_iterations=dilate_iterations,
            erode_kernel_w=erode_kernel_w,
            erode_kernel_h=erode_kernel_h,
            erode_iterations=erode_iterations,
            open_kernel_w=open_kernel_w,
            open_kernel_h=open_kernel_h,
            open_iterations=open_iterations,
            close_kernel_w=close_kernel_w,
            close_kernel_h=close_kernel_h,
            close_iterations=close_iterations,
            fill_holes=fill_holes,
            remove_small_components=remove_small_components,
            min_component_area=min_component_area,
            clear_border=clear_border,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            percentile_background_percentile=percentile_background_percentile,
            percentile_min_contrast=percentile_min_contrast,
            hysteresis_low_threshold=hysteresis_low_threshold,
            hysteresis_high_threshold=hysteresis_high_threshold,
            hysteresis_connectivity=hysteresis_connectivity,
            sobel_percentile=sobel_percentile,
            sobel_threshold=sobel_threshold,
            sobel_kernel_size=sobel_kernel_size,
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
            mask_augmentation_enabled=mask_augmentation_enabled,
            mask_augmentation_steps=mask_augmentation_steps,
            roi_assembly_method=roi_assembly_method,
            roi_assembly_connectivity=roi_assembly_connectivity,
            min_area=min_area,
            max_area=max_area,
            min_perimeter=min_perimeter,
            max_perimeter=max_perimeter,
            min_width=min_width,
            max_width=max_width,
            min_height=min_height,
            max_height=max_height,
            min_width_plus_height=min_width_plus_height,
            max_width_plus_height=max_width_plus_height,
            padding=padding,
            roi_encoding=roi_encoding,
            zstd_min_bytes=zstd_min_bytes,
            store_roi_payload_min_area=store_roi_payload_min_area,
            store_roi_payload_min_width=store_roi_payload_min_width,
            store_roi_payload_min_height=store_roi_payload_min_height,
            store_roi_payload_min_width_plus_height=store_roi_payload_min_width_plus_height,
            always_store_mask=always_store_mask,
            encode_payloads=encode_payloads,
            max_detections=max_detections,
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
    threshold_method: str | None = None,
    manual_threshold: int | float | None = None,
    thresholding_maximum_value: int | float | None = None,
    bounded_otsu_min_contrast: int | float | None = None,
    bounded_otsu_max_foreground_fraction: float | None = None,
    canny_enabled: bool | None = None,
    canny_low_threshold: int | float | None = None,
    canny_high_threshold: int | float | None = None,
    canny_blur_kernel: int | None = None,
    dilate_kernel_w: int | None = None,
    dilate_kernel_h: int | None = None,
    dilate_iterations: int | None = None,
    erode_kernel_w: int | None = None,
    erode_kernel_h: int | None = None,
    erode_iterations: int | None = None,
    open_kernel_w: int | None = None,
    open_kernel_h: int | None = None,
    open_iterations: int | None = None,
    close_kernel_w: int | None = None,
    close_kernel_h: int | None = None,
    close_iterations: int | None = None,
    fill_holes: bool | None = None,
    remove_small_components: bool | None = None,
    min_component_area: int | float | None = None,
    clear_border: bool | None = None,
    adaptive_block_size: int | None = None,
    adaptive_c: int | float | None = None,
    percentile_background_percentile: int | float | None = None,
    percentile_min_contrast: int | float | None = None,
    hysteresis_low_threshold: int | float | None = None,
    hysteresis_high_threshold: int | float | None = None,
    hysteresis_connectivity: int | None = None,
    sobel_percentile: int | float | None = None,
    sobel_threshold: int | float | None = None,
    sobel_kernel_size: int | None = None,
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
    mask_augmentation_enabled: bool | None = None,
    mask_augmentation_steps: list[str] | tuple[str, ...] | None = None,
    roi_assembly_method: str | None = None,
    roi_assembly_connectivity: int | None = None,
    min_area: int | float | None = None,
    max_area: int | float | None = None,
    min_perimeter: int | float | None = None,
    max_perimeter: int | float | None = None,
    min_width: int | float | None = None,
    max_width: int | float | None = None,
    min_height: int | float | None = None,
    max_height: int | float | None = None,
    min_width_plus_height: int | float | None = None,
    max_width_plus_height: int | float | None = None,
    padding: int | None = None,
    roi_encoding: str | None = None,
    zstd_min_bytes: int | None = None,
    store_roi_payload_min_area: int | float | None = None,
    store_roi_payload_min_width: int | float | None = None,
    store_roi_payload_min_height: int | float | None = None,
    store_roi_payload_min_width_plus_height: int | float | None = None,
    always_store_mask: bool | None = None,
    encode_payloads: bool = True,
    max_detections: int | None = None,
    context: Any = None,
) -> list[DetectionRecord]:
    """Segment one frame into connected-component ROI detection records."""
    started = time.perf_counter()
    config = context.config.processing if context is not None and getattr(context, "config", None) is not None else default_processing_config()
    mask_defaults = config.mask_augmentation
    assembly_defaults = config.roi_assembly
    filter_defaults = config.roi_filter
    recording_defaults = config.roi_recording
    preprocessing_defaults = (
        config.preprocessing
    )
    thresholding_defaults = (
        config.thresholding
    )
    flatfield_defaults = (
        config.flatfield
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
    resolved_mask_augmentation_enabled = (
        mask_defaults.enabled if mask_augmentation_enabled is None else mask_augmentation_enabled
    )
    explicit_manual_threshold = threshold is not None and threshold_method is None
    resolved_mask_augmentation_steps = (
        []
        if explicit_manual_threshold and mask_augmentation_steps is None
        else (
            list(mask_defaults.steps)
            if mask_augmentation_steps is None
            else [str(step) for step in mask_augmentation_steps]
        )
    )
    resolved_roi_assembly_method = (
        assembly_defaults.method if roi_assembly_method is None else roi_assembly_method
    )
    resolved_roi_assembly_connectivity = (
        assembly_defaults.connectivity
        if roi_assembly_connectivity is None
        else roi_assembly_connectivity
    )
    resolved_min_area = filter_defaults.min_area if min_area is None else min_area
    resolved_max_area = filter_defaults.max_area if max_area is None else max_area
    resolved_min_perimeter = filter_defaults.min_perimeter if min_perimeter is None else min_perimeter
    resolved_max_perimeter = filter_defaults.max_perimeter if max_perimeter is None else max_perimeter
    resolved_min_width = filter_defaults.min_width if min_width is None else min_width
    resolved_max_width = filter_defaults.max_width if max_width is None else max_width
    resolved_min_height = filter_defaults.min_height if min_height is None else min_height
    resolved_max_height = filter_defaults.max_height if max_height is None else max_height
    resolved_min_width_plus_height = (
        filter_defaults.min_width_plus_height
        if min_width_plus_height is None
        else min_width_plus_height
    )
    resolved_max_width_plus_height = (
        filter_defaults.max_width_plus_height
        if max_width_plus_height is None
        else max_width_plus_height
    )
    resolved_padding = recording_defaults.padding if padding is None else padding
    resolved_roi_encoding = recording_defaults.roi_encoding if roi_encoding is None else roi_encoding
    resolved_zstd_min_bytes = recording_defaults.zstd_min_bytes if zstd_min_bytes is None else zstd_min_bytes
    resolved_store_roi_payload_min_area = (
        recording_defaults.store_roi_payload_min_area
        if store_roi_payload_min_area is None
        else store_roi_payload_min_area
    )
    resolved_store_roi_payload_min_width = (
        recording_defaults.store_roi_payload_min_width
        if store_roi_payload_min_width is None
        else store_roi_payload_min_width
    )
    resolved_store_roi_payload_min_height = (
        recording_defaults.store_roi_payload_min_height
        if store_roi_payload_min_height is None
        else store_roi_payload_min_height
    )
    resolved_store_roi_payload_min_width_plus_height = (
        recording_defaults.store_roi_payload_min_width_plus_height
        if store_roi_payload_min_width_plus_height is None
        else store_roi_payload_min_width_plus_height
    )
    resolved_always_store_mask = recording_defaults.always_store_mask if always_store_mask is None else always_store_mask
    resolved_threshold_method = str(
        threshold_method or ("manual" if explicit_manual_threshold else thresholding_defaults.method)
    ).lower()
    resolved_manual_threshold = (
        thresholding_defaults.manual_threshold if manual_threshold is None else manual_threshold
    )
    resolved_thresholding_maximum_value = (
        thresholding_defaults.thresholding_maximum_value
        if thresholding_maximum_value is None
        else thresholding_maximum_value
    )
    resolved_bounded_otsu_min_contrast = (
        thresholding_defaults.bounded_otsu_min_contrast
        if bounded_otsu_min_contrast is None
        else bounded_otsu_min_contrast
    )
    resolved_bounded_otsu_max_foreground_fraction = (
        thresholding_defaults.bounded_otsu_max_foreground_fraction
        if bounded_otsu_max_foreground_fraction is None
        else bounded_otsu_max_foreground_fraction
    )
    resolved_canny_enabled = thresholding_defaults.canny_enabled if canny_enabled is None else canny_enabled
    resolved_canny_low_threshold = (
        thresholding_defaults.canny_low_threshold if canny_low_threshold is None else canny_low_threshold
    )
    resolved_canny_high_threshold = (
        thresholding_defaults.canny_high_threshold if canny_high_threshold is None else canny_high_threshold
    )
    resolved_canny_blur_kernel = thresholding_defaults.canny_blur_kernel if canny_blur_kernel is None else canny_blur_kernel
    resolved_dilate_kernel_w = mask_defaults.dilate_kernel_w if dilate_kernel_w is None else dilate_kernel_w
    resolved_dilate_kernel_h = mask_defaults.dilate_kernel_h if dilate_kernel_h is None else dilate_kernel_h
    resolved_dilate_iterations = mask_defaults.dilate_iterations if dilate_iterations is None else dilate_iterations
    resolved_erode_kernel_w = mask_defaults.erode_kernel_w if erode_kernel_w is None else erode_kernel_w
    resolved_erode_kernel_h = mask_defaults.erode_kernel_h if erode_kernel_h is None else erode_kernel_h
    resolved_erode_iterations = mask_defaults.erode_iterations if erode_iterations is None else erode_iterations
    resolved_open_kernel_w = mask_defaults.open_kernel_w if open_kernel_w is None else open_kernel_w
    resolved_open_kernel_h = mask_defaults.open_kernel_h if open_kernel_h is None else open_kernel_h
    resolved_open_iterations = mask_defaults.open_iterations if open_iterations is None else open_iterations
    resolved_close_kernel_w = mask_defaults.close_kernel_w if close_kernel_w is None else close_kernel_w
    resolved_close_kernel_h = mask_defaults.close_kernel_h if close_kernel_h is None else close_kernel_h
    resolved_close_iterations = mask_defaults.close_iterations if close_iterations is None else close_iterations
    resolved_fill_holes = mask_defaults.fill_holes if fill_holes is None else fill_holes
    resolved_remove_small_components = (
        mask_defaults.remove_small_components if remove_small_components is None else remove_small_components
    )
    resolved_min_component_area = mask_defaults.min_component_area if min_component_area is None else min_component_area
    resolved_clear_border = mask_defaults.clear_border if clear_border is None else clear_border
    resolved_adaptive_block_size = thresholding_defaults.adaptive_block_size if adaptive_block_size is None else adaptive_block_size
    resolved_adaptive_c = thresholding_defaults.adaptive_c if adaptive_c is None else adaptive_c
    resolved_percentile_background_percentile = (
        thresholding_defaults.percentile_background_percentile
        if percentile_background_percentile is None
        else percentile_background_percentile
    )
    resolved_percentile_min_contrast = (
        thresholding_defaults.percentile_min_contrast if percentile_min_contrast is None else percentile_min_contrast
    )
    resolved_hysteresis_low_threshold = (
        thresholding_defaults.hysteresis_low_threshold if hysteresis_low_threshold is None else hysteresis_low_threshold
    )
    resolved_hysteresis_high_threshold = (
        thresholding_defaults.hysteresis_high_threshold if hysteresis_high_threshold is None else hysteresis_high_threshold
    )
    resolved_hysteresis_connectivity = (
        thresholding_defaults.hysteresis_connectivity if hysteresis_connectivity is None else hysteresis_connectivity
    )
    resolved_sobel_percentile = thresholding_defaults.sobel_percentile if sobel_percentile is None else sobel_percentile
    resolved_sobel_threshold = thresholding_defaults.sobel_threshold if sobel_threshold is None else sobel_threshold
    resolved_sobel_kernel_size = thresholding_defaults.sobel_kernel_size if sobel_kernel_size is None else sobel_kernel_size

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
        "mask_augmentation_enabled": bool(resolved_mask_augmentation_enabled),
        "mask_augmentation_steps": resolved_mask_augmentation_steps,
        "dilate_kernel_w": resolved_dilate_kernel_w,
        "dilate_kernel_h": resolved_dilate_kernel_h,
        "dilate_iterations": resolved_dilate_iterations,
        "erode_kernel_w": resolved_erode_kernel_w,
        "erode_kernel_h": resolved_erode_kernel_h,
        "erode_iterations": resolved_erode_iterations,
        "open_kernel_w": resolved_open_kernel_w,
        "open_kernel_h": resolved_open_kernel_h,
        "open_iterations": resolved_open_iterations,
        "close_kernel_w": resolved_close_kernel_w,
        "close_kernel_h": resolved_close_kernel_h,
        "close_iterations": resolved_close_iterations,
        "fill_holes": bool(resolved_fill_holes),
        "remove_small_components": bool(resolved_remove_small_components),
        "min_component_area": resolved_min_component_area,
        "clear_border": bool(resolved_clear_border),
        "roi_assembly_method": resolved_roi_assembly_method,
        "roi_assembly_connectivity": resolved_roi_assembly_connectivity,
        "min_area": resolved_min_area,
        "max_area": resolved_max_area,
        "min_perimeter": resolved_min_perimeter,
        "max_perimeter": resolved_max_perimeter,
        "min_width": resolved_min_width,
        "max_width": resolved_max_width,
        "min_height": resolved_min_height,
        "max_height": resolved_max_height,
        "min_width_plus_height": resolved_min_width_plus_height,
        "max_width_plus_height": resolved_max_width_plus_height,
        "padding": resolved_padding,
        "roi_encoding": resolved_roi_encoding,
        "always_store_mask": bool(resolved_always_store_mask),
        "encode_payloads": bool(encode_payloads),
        "max_detections": max_detections,
        "threshold_method": resolved_threshold_method,
        "manual_threshold": resolved_manual_threshold,
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
        stage_durations_ms: dict[str, float] = {}
        source_frame = frame
        if apply_preprocessing:
            stage_started = time.perf_counter()
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
            stage_durations_ms["preprocessing"] = (time.perf_counter() - stage_started) * 1000
        else:
            stage_durations_ms["preprocessing"] = 0.0
        stage_started = time.perf_counter()
        gray = _as_grayscale_array(frame)
        stage_durations_ms["grayscale"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        threshold_mask = calculate_threshold_mask(
            gray,
            threshold=threshold,
            method=resolved_threshold_method,
            manual_threshold=resolved_manual_threshold,
            thresholding_maximum_value=resolved_thresholding_maximum_value,
            bounded_otsu_min_contrast=resolved_bounded_otsu_min_contrast,
            bounded_otsu_max_foreground_fraction=resolved_bounded_otsu_max_foreground_fraction,
            canny_enabled=resolved_canny_enabled,
            canny_low_threshold=resolved_canny_low_threshold,
            canny_high_threshold=resolved_canny_high_threshold,
            canny_blur_kernel=resolved_canny_blur_kernel,
            adaptive_block_size=resolved_adaptive_block_size,
            adaptive_c=resolved_adaptive_c,
            percentile_background_percentile=resolved_percentile_background_percentile,
            percentile_min_contrast=resolved_percentile_min_contrast,
            hysteresis_low_threshold=resolved_hysteresis_low_threshold,
            hysteresis_high_threshold=resolved_hysteresis_high_threshold,
            hysteresis_connectivity=resolved_hysteresis_connectivity,
            sobel_percentile=resolved_sobel_percentile,
            sobel_threshold=resolved_sobel_threshold,
            sobel_kernel_size=resolved_sobel_kernel_size,
        )
        threshold_mask = as_binary_mask(threshold_mask)
        stage_durations_ms["thresholding"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        augmented_mask = augment_mask(
            threshold_mask,
            enabled=bool(resolved_mask_augmentation_enabled),
            steps=resolved_mask_augmentation_steps,
            dilate_kernel_size=(int(resolved_dilate_kernel_w), int(resolved_dilate_kernel_h)),
            dilate_iterations=resolved_dilate_iterations,
            erode_kernel_size=(int(resolved_erode_kernel_w), int(resolved_erode_kernel_h)),
            erode_iterations=resolved_erode_iterations,
            open_kernel_size=(int(resolved_open_kernel_w), int(resolved_open_kernel_h)),
            open_iterations=resolved_open_iterations,
            close_kernel_size=(int(resolved_close_kernel_w), int(resolved_close_kernel_h)),
            close_iterations=resolved_close_iterations,
            fill_holes=resolved_fill_holes,
            remove_small_components=resolved_remove_small_components,
            min_component_area=resolved_min_component_area,
            clear_border=resolved_clear_border,
        )
        stage_durations_ms["mask_augmentation"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        assembled_candidates = assemble_candidate_rois(
            augmented_mask,
            method=str(resolved_roi_assembly_method),
            connectivity=int(resolved_roi_assembly_connectivity),
        )
        stage_durations_ms["roi_assembly"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        candidates = filter_candidate_rois(
            assembled_candidates,
            min_area=resolved_min_area,
            max_area=resolved_max_area,
            min_perimeter=resolved_min_perimeter,
            max_perimeter=resolved_max_perimeter,
            min_width=resolved_min_width,
            max_width=resolved_max_width,
            min_height=resolved_min_height,
            max_height=resolved_max_height,
            min_width_plus_height=resolved_min_width_plus_height,
            max_width_plus_height=resolved_max_width_plus_height,
        )
        filtered_candidate_count = len(candidates)
        if max_detections is not None:
            candidates = candidates[:max(0, int(max_detections))]
        stage_durations_ms["roi_filter"] = (time.perf_counter() - stage_started) * 1000
        stage_started = time.perf_counter()
        roi_records: list[DetectionRecord] = []
        for candidate in candidates:
            store_payload = bool(encode_payloads) and should_store_roi_payload(
                candidate,
                min_area=resolved_store_roi_payload_min_area,
                min_width=resolved_store_roi_payload_min_width,
                min_height=resolved_store_roi_payload_min_height,
                min_width_plus_height=resolved_store_roi_payload_min_width_plus_height,
            )
            roi_records.append(
                build_candidate_detection_record(
                    candidate,
                    source_frame=source_frame,
                    processed_frame=frame,
                    roi_index=candidate.roi_index,
                    padding=int(resolved_padding),
                    encoding=resolved_roi_encoding,
                    zstd_min_bytes=resolved_zstd_min_bytes,
                    store_roi_payload=store_payload,
                    always_store_mask=bool(encode_payloads) and bool(resolved_always_store_mask),
                    extra_metadata={
                        "threshold_method": resolved_threshold_method,
                        "mask_augmentation_steps": resolved_mask_augmentation_steps,
                        "payloads_encoded": bool(encode_payloads),
                    },
                )
            )
        stage_durations_ms["roi_recording"] = (time.perf_counter() - stage_started) * 1000
        stage_metadata = {
            "stage_counts": {
                "threshold_foreground_pixels": int(np.count_nonzero(threshold_mask)),
                "augmented_foreground_pixels": int(np.count_nonzero(augmented_mask)),
                "assembled_candidate_count": len(assembled_candidates),
                "filtered_candidate_count": filtered_candidate_count,
                "recordable_candidate_count": len(candidates),
                "recorded_detection_count": len(roi_records),
            },
            "candidate_limit": max_detections,
            "candidate_limit_applied": len(candidates) < filtered_candidate_count,
            "stage_durations_ms": {
                key: round(value, 3) for key, value in stage_durations_ms.items()
            },
            "processed_frame_shape": list(gray.shape),
            "bbox_coordinate_space": "processed_frame_with_frame_bbox_offset",
        }
        for record in roi_records:
            record.metadata.update(stage_metadata)
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
            "source_component_count": len(assembled_candidates),
            "detection_count": len(roi_records),
            "frame_shape": list(gray.shape),
        },
        logger="pelagia.processing.detection_candidate",
        core_logger=_CORE_LOGGER,
    )
    return roi_records
