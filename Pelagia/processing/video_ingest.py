import os
import time
from typing import Any

import cv2
import numpy as np

from ..services.context import AppContext
from ._logging import log_processing_event, processing_core_logger
from .defaults import default_processing_config
from .frame_model import FrameData
from .frame_store import store_frame
from .frame_time import parse_filename_timestamp_utc, timestamp_for_frame


_CORE_LOGGER = processing_core_logger("video_ingest")
_PROGRESS_LOG_TILE_INTERVAL = 100


def _log_database_event(
    context: AppContext | None,
    level: str,
    event_type: str,
    message: str,
    *,
    run_id: str | None = None,
    asset_id: str | None = None,
    duration_ms: float | None = None,
    payload: dict[str, Any] | None = None,
) -> None:
    log_processing_event(
        context,
        level,
        event_type,
        message,
        run_id=run_id,
        asset_id=asset_id,
        duration_ms=duration_ms,
        payload=payload,
        logger="pelagia.processing.video_ingest",
        core_logger=_CORE_LOGGER,
    )


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
    started = time.perf_counter()
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
    ingest_payload = {
        "input_path": input_path,
        "source_path": source_path,
        "filename": filename,
        "n_tile": int(n_tile),
        "flatfield_correction": bool(flatfield_correction),
        "flatfield_q": flatfield_q,
        "flatfield_axis": flatfield_axis,
        "adaptive_background_subtraction": bool(adaptive_background_subtraction),
        "adaptive_background_period": int(adaptive_background_period),
        "frame_mask": bool(frame_mask),
        "frame_mask_path": frame_mask_path,
    }
    _CORE_LOGGER.info(
        "Starting video ingest for %s run_id=%s asset_id=%s n_tile=%s",
        input_path,
        run_id,
        asset_id,
        n_tile,
    )
    _log_database_event(
        context,
        "info",
        "video_ingest.started",
        "Video ingest started",
        run_id=run_id,
        asset_id=asset_id,
        payload=ingest_payload,
    )
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
        duration_ms = (time.perf_counter() - started) * 1000
        _CORE_LOGGER.error("Could not open video file %s", input_path)
        _log_database_event(
            context,
            "error",
            "video_ingest.open_failed",
            "Could not open video file",
            run_id=run_id,
            asset_id=asset_id,
            duration_ms=duration_ms,
            payload=ingest_payload,
        )
        raise ValueError(f"Could not open video file: {input_path}")

    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    start_timestamp = parse_filename_timestamp_utc(filename)
    if start_timestamp is not None:
        frame_metadata["source_timestamp_utc"] = start_timestamp.isoformat()
    if fps > 0:
        frame_metadata["fps"] = fps
        frame_metadata["frame_interval_seconds"] = 1.0 / fps
    _log_database_event(
        context,
        "debug",
        "video_ingest.video_opened",
        "Video file opened",
        run_id=run_id,
        asset_id=asset_id,
        payload={
            **ingest_payload,
            "fps": fps,
            "source_timestamp_utc": None if start_timestamp is None else start_timestamp.isoformat(),
        },
    )

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
                    if tile_number == 1 or tile_number % _PROGRESS_LOG_TILE_INTERVAL == 0:
                        _log_database_event(
                            context,
                            "debug",
                            "video_ingest.tile_stored",
                            "Video ingest tile stored",
                            run_id=run_id,
                            asset_id=asset_id,
                            payload={
                                "filename": filename,
                                "tile_number": tile_number,
                                "source_frame_start": n - len(frame_buffer) + 1,
                                "source_frame_end": n,
                                "stored_tile_count": len(stored_frames),
                            },
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
            _log_database_event(
                context,
                "debug",
                "video_ingest.tile_stored",
                "Video ingest final tile stored",
                run_id=run_id,
                asset_id=asset_id,
                payload={
                    "filename": filename,
                    "tile_number": tile_number,
                    "source_frame_start": n - len(frame_buffer),
                    "source_frame_end": n - 1,
                    "stored_tile_count": len(stored_frames),
                    "partial_tile": True,
                },
            )
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        _CORE_LOGGER.exception(
            "Video ingest failed for %s run_id=%s asset_id=%s",
            input_path,
            run_id,
            asset_id,
        )
        _log_database_event(
            context,
            "error",
            "video_ingest.failed",
            "Video ingest failed",
            run_id=run_id,
            asset_id=asset_id,
            duration_ms=duration_ms,
            payload={
                **ingest_payload,
                "source_frame_count": max(0, n - 1),
                "stored_tile_count": len(stored_frames),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise
    finally:
        video.release()

    duration_ms = (time.perf_counter() - started) * 1000
    _CORE_LOGGER.info(
        "Completed video ingest for %s run_id=%s asset_id=%s source_frames=%s stored_tiles=%s duration_ms=%.2f",
        input_path,
        run_id,
        asset_id,
        max(0, n - 1),
        len(stored_frames),
        duration_ms,
    )
    _log_database_event(
        context,
        "info",
        "video_ingest.completed",
        "Video ingest completed",
        run_id=run_id,
        asset_id=asset_id,
        duration_ms=duration_ms,
        payload={
            **ingest_payload,
            "fps": fps,
            "source_frame_count": max(0, n - 1),
            "stored_tile_count": len(stored_frames),
        },
    )
    return stored_frames
