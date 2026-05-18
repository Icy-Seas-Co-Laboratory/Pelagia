import os
from typing import Any

import cv2
import numpy as np

from ..services.context import AppContext
from .frame_model import Frame
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
    n_tile=1,
    *,
    context: AppContext | None = None,
    run_id: str | None = None,
    asset_id: str | None = None,
    dest_path=None,
    metadata: dict[str, Any] | None = None,
    flatfield_correction: bool = True,
    flatfield_q: float = 0.9,
    flatfield_axis: int = 0,
) -> list[dict[str, Any]]:
    if n_tile < 1:
        raise ValueError("n_tile must be >= 1.")

    input_path = os.fspath(input_path)
    source_path = os.path.dirname(os.path.abspath(input_path))
    filename = os.path.basename(input_path)
    dest_path = os.fspath(dest_path) if dest_path is not None else source_path
    frame_metadata = dict(metadata or {})
    if run_id is not None:
        frame_metadata["run_id"] = run_id
    if asset_id is not None:
        frame_metadata["asset_id"] = asset_id
    if flatfield_correction:
        frame_metadata["flatfield_correction"] = True
        frame_metadata["flatfield_q"] = flatfield_q

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
                            Frame(
                                sourcePath=source_path,
                                destPath=dest_path,
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
                    Frame(
                        sourcePath=source_path,
                        destPath=dest_path,
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
