import os
from typing import Any

import cv2
import numpy as np

from ..services.context import AppContext
from .defaults import default_processing_config
from .frame_model import FrameData
from .frame_store import store_frame
from .frame_time import parse_filename_timestamp_utc, timestamp_for_frame


def convert_frame_to_grayscale(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
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


def ingest_video_file(
    input_path,
    n_tile=None,
    *,
    context: AppContext | None = None,
    run_id: str | None = None,
    asset_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    flatfield_correction: bool | None = None,
    flatfield_q: float | None = None,
    flatfield_axis: int | None = None,
    adaptive_background_subtraction: bool | None = None,
    adaptive_background_period: int | None = None,
    frame_mask: bool | None = None,
    frame_mask_path: str | None = None,
) -> list[dict[str, Any]]:
    defaults = (
        context.config.processing.video_ingest
        if context is not None
        else default_processing_config().video_ingest
    )
    n_tile = defaults.n_tile if n_tile is None else n_tile
    flatfield_correction = (
        defaults.flatfield_correction if flatfield_correction is None else flatfield_correction
    )
    flatfield_q = defaults.flatfield_q if flatfield_q is None else flatfield_q
    flatfield_axis = defaults.flatfield_axis if flatfield_axis is None else flatfield_axis
    adaptive_background_subtraction = (
        defaults.adaptive_background_subtraction
        if adaptive_background_subtraction is None
        else adaptive_background_subtraction
    )
    adaptive_background_period = (
        defaults.adaptive_background_period
        if adaptive_background_period is None
        else adaptive_background_period
    )
    frame_mask = defaults.frame_mask if frame_mask is None else frame_mask
    frame_mask_path = defaults.frame_mask_path if frame_mask_path is None else frame_mask_path

    if n_tile < 1:
        raise ValueError("n_tile must be >= 1.")
    if adaptive_background_period < 1:
        raise ValueError("adaptive_background_period must be >= 1.")

    input_path = os.fspath(input_path)
    source_path = os.path.dirname(os.path.abspath(input_path))
    filename = os.path.basename(input_path)
    frame_metadata = dict(metadata or {})
    if run_id is not None:
        frame_metadata["run_id"] = run_id
    if asset_id is not None:
        frame_metadata["asset_id"] = asset_id
    if flatfield_correction:
        frame_metadata["flatfield_correction"] = True
        frame_metadata["flatfield_q"] = flatfield_q
        frame_metadata["flatfield_axis"] = flatfield_axis
    frame_metadata["adaptive_background_subtraction"] = bool(adaptive_background_subtraction)
    frame_metadata["adaptive_background_period"] = int(adaptive_background_period)
    frame_metadata["frame_mask"] = bool(frame_mask)
    frame_metadata["frame_mask_path"] = frame_mask_path

    video = cv2.VideoCapture(input_path)
    if not video.isOpened():
        raise ValueError(f"Could not open video file: {input_path}")

    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    start_timestamp = parse_filename_timestamp_utc(filename)
    if start_timestamp is not None:
        frame_metadata["source_timestamp_utc"] = start_timestamp.isoformat()
    if fps > 0:
        frame_metadata["fps"] = fps
        frame_metadata["frame_interval_seconds"] = 1.0 / fps

    frame_buffer = []
    stored_frames = []
    n = 1
    tile_number = 1

    try:
        while video.isOpened():
            good_return, frame = video.read()
            if not good_return:
                break
            if frame is not None:
                frame = convert_frame_to_grayscale(frame)
                frame_buffer.append(frame)
                if len(frame_buffer) == n_tile:
                    tiled = np.vstack(frame_buffer)
                    stored_frames.append(
                        store_frame(
                            FrameData(
                                sourcePath=source_path,
                                filename=filename,
                                frameNumber=n,
                                data=tiled,
                                tileNumber=tile_number,
                                sourceFrameStart=n - len(frame_buffer) + 1,
                                sourceFrameEnd=n,
                                frameType="line",
                                timestamp=timestamp_for_frame(start_timestamp, fps, n),
                                metadata=frame_metadata.copy(),
                            ),
                            context=context,
                        )
                    )
                    tile_number += 1
                    frame_buffer = []
                n += 1

        if frame_buffer:
            tiled = np.vstack(frame_buffer)
            stored_frames.append(
                store_frame(
                    FrameData(
                        sourcePath=source_path,
                        filename=filename,
                        frameNumber=n - 1,
                        data=tiled,
                        tileNumber=tile_number,
                        sourceFrameStart=n - len(frame_buffer),
                        sourceFrameEnd=n - 1,
                        frameType="line",
                        timestamp=timestamp_for_frame(start_timestamp, fps, n - 1),
                        metadata=frame_metadata.copy(),
                    ),
                    context=context,
                )
            )
    finally:
        video.release()

    return stored_frames
