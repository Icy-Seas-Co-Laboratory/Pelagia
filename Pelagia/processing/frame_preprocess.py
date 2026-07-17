"""Prepare stored frames for segmentation using configured corrections."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .defaults import default_processing_config
from .frame_correction import (
    divide_background,
    flatfield_correction as correct_flatfield,
    flatfield_profile_correction,
)
from .frame_model import FrameData
from .timing import measure_phase


def as_grayscale_array(data: np.ndarray) -> np.ndarray:
    """Return a contiguous grayscale array without changing intensity polarity."""
    array = np.asarray(data)
    if array.ndim == 2:
        return np.ascontiguousarray(array)
    if array.ndim != 3:
        raise ValueError(f"Expected a 2D grayscale or 3D color frame, got shape {array.shape}.")

    channels = array.shape[2]
    if channels == 1:
        return np.ascontiguousarray(array[:, :, 0])
    if channels == 3:
        return np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_BGR2GRAY))
    if channels == 4:
        return np.ascontiguousarray(cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY))
    raise ValueError(f"Expected frame with 1, 3, or 4 channels, got {channels}.")


def as_binary_mask(mask: np.ndarray) -> np.ndarray:
    """Normalize any 2D mask-like array to uint8 values 0 and 255."""
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("Frame mask must be a 2D image.")
    return np.ascontiguousarray((array > 0).astype(np.uint8) * 255)


def apply_frame_mask(data: np.ndarray, mask: np.ndarray | None, *, fill_value: int | float = 0) -> np.ndarray:
    """Apply a valid-pixel mask to image data, setting masked-out pixels to fill_value."""
    if mask is None:
        return np.ascontiguousarray(data)

    image = np.asarray(data)
    binary_mask = as_binary_mask(mask)
    if image.shape[:2] != binary_mask.shape[:2]:
        raise ValueError(
            f"Frame mask shape {binary_mask.shape[:2]} does not match image shape {image.shape[:2]}."
        )

    masked = np.array(image, copy=True)
    masked[binary_mask == 0] = fill_value
    return np.ascontiguousarray(masked)


def _resolve_crop_bounds(
    image_shape: tuple[int, ...],
    *,
    crop_enabled: bool | None,
    crop_x: int | None,
    crop_y: int | None,
    crop_w: int | None,
    crop_h: int | None,
) -> tuple[int, int, int, int] | None:
    coordinates = (crop_x, crop_y, crop_w, crop_h)
    has_coordinates = any(value is not None for value in coordinates)
    if crop_enabled is False:
        return None
    if not has_coordinates:
        return None
    if not all(value is not None for value in coordinates):
        raise ValueError("Crop requires crop_x, crop_y, crop_w, and crop_h.")

    x = int(crop_x)
    y = int(crop_y)
    width = int(crop_w)
    height = int(crop_h)
    if width < 1 or height < 1:
        raise ValueError("Crop width and height must be >= 1.")
    if x < 0 or y < 0:
        raise ValueError("Crop x and y must be >= 0.")

    image_height, image_width = image_shape[:2]
    x1 = x + width
    y1 = y + height
    if x1 > image_width or y1 > image_height:
        raise ValueError(
            f"Crop bbox {(x, y, width, height)} exceeds image bounds {(image_width, image_height)}."
        )
    return x, y, x1, y1


def crop_frame_data(
    data: np.ndarray,
    *,
    crop_x: int,
    crop_y: int,
    crop_w: int,
    crop_h: int,
) -> np.ndarray:
    """Crop image data using x/y origin plus width/height coordinates."""
    bounds = _resolve_crop_bounds(
        np.asarray(data).shape,
        crop_enabled=True,
        crop_x=crop_x,
        crop_y=crop_y,
        crop_w=crop_w,
        crop_h=crop_h,
    )
    if bounds is None:
        return np.ascontiguousarray(data)
    x0, y0, x1, y1 = bounds
    return np.ascontiguousarray(np.asarray(data)[y0:y1, x0:x1])


def invert_image_intensity(data: np.ndarray) -> np.ndarray:
    """Invert image intensity so downstream thresholding sees bright foreground."""
    image = np.asarray(data)
    if np.issubdtype(image.dtype, np.integer):
        info = np.iinfo(image.dtype)
        inverted = info.max - image
    else:
        max_value = float(np.nanmax(image)) if image.size else 1.0
        inverted = max_value - image
    return np.ascontiguousarray(inverted.astype(image.dtype, copy=False))


def preprocess_frame_for_segmentation(
    frame: FrameData,
    *,
    mask: np.ndarray | None = None,
    apply_mask: bool | None = None,
    crop_enabled: bool | None = None,
    crop_x: int | None = None,
    crop_y: int | None = None,
    crop_w: int | None = None,
    crop_h: int | None = None,
    flatfield_correction: bool | None = None,
    flatfield_q: float | None = None,
    flatfield_axis: int | None = None,
    flatfield_profile: np.ndarray | list[float] | tuple[float, ...] | None = None,
    flatfield_min_field_value: int | float | None = None,
    flatfield_max_field_value: int | float | None = None,
    background_correction: bool | None = None,
    background: np.ndarray | int | float | None = None,
    background_min_field_value: int | float | None = None,
    background_max_field_value: int | float | None = None,
    invert_intensity: bool | None = None,
    context: Any = None,
) -> FrameData:
    """
    Convert a runtime frame into the bright-foreground image used for thresholding.

    Processing order follows Pelagia's candidate segmentation model:
    crop -> mask -> flatfield -> background correction -> optional inversion.
    """
    data = frame.read()
    if data is None:
        raise ValueError("Frame has no image data to preprocess.")

    with measure_phase("processing.grayscale"):
        image = as_grayscale_array(data)
    original_image_shape = image.shape
    processing_defaults = (
        context.config.processing.preprocessing
        if context is not None and getattr(context, "config", None) is not None
        else default_processing_config().preprocessing
    )
    flatfield_defaults = (
        context.config.processing.flatfield
        if context is not None and getattr(context, "config", None) is not None
        else default_processing_config().flatfield
    )
    resolved_apply_mask = processing_defaults.apply_mask if apply_mask is None else apply_mask
    resolved_crop_enabled = (
        processing_defaults.crop_enabled if crop_enabled is None else crop_enabled
    )
    resolved_crop_x = processing_defaults.crop_x if crop_x is None else crop_x
    resolved_crop_y = processing_defaults.crop_y if crop_y is None else crop_y
    resolved_crop_w = processing_defaults.crop_w if crop_w is None else crop_w
    resolved_crop_h = processing_defaults.crop_h if crop_h is None else crop_h
    resolved_background_correction = (
        processing_defaults.background_correction
        if background_correction is None
        else background_correction
    )
    resolved_background_min_field_value = (
        processing_defaults.background_min_field_value
        if background_min_field_value is None
        else background_min_field_value
    )
    resolved_background_max_field_value = (
        processing_defaults.background_max_field_value
        if background_max_field_value is None
        else background_max_field_value
    )
    resolved_invert_intensity = (
        processing_defaults.invert_intensity if invert_intensity is None else invert_intensity
    )

    source_mask = mask if mask is not None else frame.mask
    source_background = background if background is not None else frame.bkg
    source_flatfield_profile = (
        flatfield_profile
        if flatfield_profile is not None
        else (frame.metadata or {}).get("flatfield_profile")
    )
    source_flatfield_metadata = dict((frame.metadata or {}).get("flatfield_metadata") or {})
    source_flatfield_axis = int(
        source_flatfield_metadata.get(
            "flatfield_axis",
            flatfield_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis,
        )
    )
    if source_flatfield_axis not in {0, 1}:
        raise ValueError("Stored flatfield profile axis must be 0 or 1.")
    with measure_phase("processing.crop"):
        crop_bounds = _resolve_crop_bounds(
            image.shape,
            crop_enabled=resolved_crop_enabled,
            crop_x=resolved_crop_x,
            crop_y=resolved_crop_y,
            crop_w=resolved_crop_w,
            crop_h=resolved_crop_h,
        )
        if crop_bounds is not None:
            x0, y0, x1, y1 = crop_bounds
            image = np.ascontiguousarray(image[y0:y1, x0:x1])
            if source_flatfield_profile is not None:
                profile_slice = slice(x0, x1) if source_flatfield_axis == 0 else slice(y0, y1)
                source_flatfield_profile = np.asarray(source_flatfield_profile)[profile_slice]
            if source_mask is not None:
                source_mask = crop_frame_data(
                    source_mask,
                    crop_x=x0,
                    crop_y=y0,
                    crop_w=x1 - x0,
                    crop_h=y1 - y0,
                )
            if (
                isinstance(source_background, np.ndarray)
                and source_background.shape[:2] == original_image_shape[:2]
            ):
                source_background = crop_frame_data(
                    source_background,
                    crop_x=x0,
                    crop_y=y0,
                    crop_w=x1 - x0,
                    crop_h=y1 - y0,
                )
        else:
            x0 = y0 = 0

    if resolved_apply_mask and source_mask is not None:
        with measure_phase("processing.mask"):
            image = apply_frame_mask(image, source_mask)

    resolved_flatfield_correction = (
        flatfield_defaults.flatfield_correction if flatfield_correction is None else flatfield_correction
    )
    resolved_flatfield_q = flatfield_defaults.flatfield_q if flatfield_q is None else flatfield_q
    resolved_flatfield_axis = (
        source_flatfield_axis
        if flatfield_axis is None and source_flatfield_profile is not None
        else flatfield_defaults.flatfield_axis if flatfield_axis is None else flatfield_axis
    )
    resolved_flatfield_min_field_value = (
        flatfield_defaults.flatfield_min_field_value
        if flatfield_min_field_value is None
        else flatfield_min_field_value
    )
    resolved_flatfield_max_field_value = (
        flatfield_defaults.flatfield_max_field_value
        if flatfield_max_field_value is None
        else flatfield_max_field_value
    )

    uses_stored_flatfield_profile = False
    if resolved_flatfield_correction:
        uses_stored_flatfield_profile = (
            source_flatfield_profile is not None
            and source_flatfield_axis == resolved_flatfield_axis
        )
        with measure_phase("processing.flatfield"):
            if uses_stored_flatfield_profile:
                image = flatfield_profile_correction(
                    image,
                    source_flatfield_profile,
                    axis=resolved_flatfield_axis,
                    min_field_value=resolved_flatfield_min_field_value,
                    max_field_value=resolved_flatfield_max_field_value,
                )
            else:
                image = correct_flatfield(
                    image,
                    q=resolved_flatfield_q,
                    axis=resolved_flatfield_axis,
                    min_field_value=resolved_flatfield_min_field_value,
                    max_field_value=resolved_flatfield_max_field_value,
                )

    if resolved_background_correction:
        if source_background is None:
            raise ValueError(
                "background_correction requires a generated background field on the frame."
            )
        with measure_phase("processing.background_correction"):
            image = divide_background(
                image,
                background=source_background,
                min_field_value=resolved_background_min_field_value,
                max_field_value=resolved_background_max_field_value,
            )

    if resolved_invert_intensity:
        with measure_phase("processing.invert_intensity"):
            image = invert_image_intensity(image)

    metadata = dict(frame.metadata or {})
    metadata.pop("flatfield_profile", None)
    preprocessing_steps = list(metadata.get("preprocessing_steps") or [])
    preprocessing_steps.extend(
        step
        for step, enabled in (
            ("crop", crop_bounds is not None),
            ("mask", bool(resolved_apply_mask and source_mask is not None)),
            ("flatfield", bool(resolved_flatfield_correction)),
            ("background_correction", bool(resolved_background_correction)),
            ("invert_intensity", bool(resolved_invert_intensity)),
        )
        if enabled
    )
    crop_bbox = None if crop_bounds is None else (
        frame.bbox_x + int(x0),
        frame.bbox_y + int(y0),
        int(image.shape[1]),
        int(image.shape[0]),
    )
    metadata.update(
        {
            "preprocessing_steps": preprocessing_steps,
            "candidate_image_kind": "preprocessed",
            "foreground_polarity": "bright",
            "frame_mask_applied": bool(resolved_apply_mask and source_mask is not None),
            "flatfield_profile_method": (
                ("window_column_mean" if resolved_flatfield_axis == 0 else "window_row_mean")
                if resolved_flatfield_correction and uses_stored_flatfield_profile
                else "frame_quantile"
            ),
            "crop_enabled": bool(crop_bounds is not None),
            "crop_bbox": crop_bbox,
            "flatfield_correction": bool(resolved_flatfield_correction),
            "flatfield_q": resolved_flatfield_q,
            "flatfield_axis": resolved_flatfield_axis,
            "flatfield_min_field_value": resolved_flatfield_min_field_value,
            "flatfield_max_field_value": resolved_flatfield_max_field_value,
            "background_correction": bool(resolved_background_correction),
            "background_method": "divide",
            "background_min_field_value": resolved_background_min_field_value,
            "background_max_field_value": resolved_background_max_field_value,
            "intensity_inverted": bool(resolved_invert_intensity),
        }
    )

    return FrameData(
        sourcePath=frame.sourcePath,
        filename=frame.filename,
        frameNumber=frame.frameNumber,
        data=image,
        mask=source_mask,
        width=int(image.shape[1]),
        height=int(image.shape[0]),
        bbox_x=frame.bbox_x + int(x0),
        bbox_y=frame.bbox_y + int(y0),
        parent_frame_id=frame.parent_frame_id,
        bkg=source_background,
        tileNumber=frame.tileNumber,
        sourceFrameStart=frame.sourceFrameStart,
        sourceFrameEnd=frame.sourceFrameEnd,
        frameType=frame.frameType,
        channel=frame.channel,
        timestamp=frame.timestamp,
        metadata=metadata,
        imageReadFlag=frame.imageReadFlag,
        cacheRead=frame.cacheRead,
    )
