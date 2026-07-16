"""Discover, decode, tile, and persist image and video sources."""

import os
import shutil
import subprocess
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..services.context import AppContext
from ._logging import log_processing_event, processing_core_logger
from .defaults import default_processing_config
from .frame_model import FrameData
from .frame_store import store_frame
from .frame_time import parse_filename_timestamp_utc, timestamp_for_frame
from .timing import measure_phase


_CORE_LOGGER = processing_core_logger("ingest")
_PROGRESS_LOG_TILE_INTERVAL = 100
IMAGE_FRAME_EXTENSIONS = {
    ".bmp",
    ".dib",
    ".jpeg",
    ".jpg",
    ".jp2",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
VIDEO_FRAME_EXTENSIONS = {
    ".avi",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp4",
    ".mpeg",
    ".mpg",
    ".wmv",
}
IngestProgressCallback = Callable[[dict[str, Any]], None]


@dataclass(frozen=True, slots=True)
class IngestSource:
    """A concrete ingestable source discovered from a path."""

    path: Path
    kind: str
    recursive: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "kind": self.kind,
            "recursive": self.recursive,
        }


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
        logger="pelagia.processing.ingest",
        core_logger=_CORE_LOGGER,
    )


def _emit_ingest_progress(
    progress_callback: IngestProgressCallback | None,
    event: str,
    payload: dict[str, Any],
) -> None:
    if progress_callback is None:
        return
    progress_callback({"event": event, **payload})


def _estimated_tile_count(source_frame_count: int, n_tile: int) -> int | None:
    if source_frame_count < 1 or n_tile < 1:
        return None
    return (source_frame_count + n_tile - 1) // n_tile


def _open_ffmpeg_capture_with_options(
    input_path: str,
    options: str,
    *,
    decoder_threads: int,
):
    existing_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = options
    try:
        api_preference = getattr(cv2, "CAP_FFMPEG", 0)
        thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
        if thread_property is not None:
            return cv2.VideoCapture(
                input_path,
                api_preference,
                [int(thread_property), int(decoder_threads)],
            )
        return cv2.VideoCapture(input_path, api_preference)
    finally:
        if existing_options is None:
            os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
        else:
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = existing_options


def _open_video_capture(
    input_path: str,
    *,
    prefer_software_decode: bool,
    decoder_threads: int,
):
    decoder_threads = max(1, int(decoder_threads))
    existing_options = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS", "")
    capture_options = existing_options
    if "threads;" not in capture_options.lower():
        capture_options = f"{capture_options}|threads;{decoder_threads}" if capture_options else f"threads;{decoder_threads}"
    if prefer_software_decode:
        software_options = capture_options
        if "hwaccel" not in software_options.lower():
            software_options = f"{software_options}|hwaccel;none"
            try:
                capture = _open_ffmpeg_capture_with_options(
                    input_path,
                    software_options,
                    decoder_threads=decoder_threads,
                )
            except (TypeError, cv2.error):
                capture = None
            if capture is not None:
                if capture.isOpened():
                    return capture, "software_ffmpeg_options"
                capture.release()

        hw_prop = getattr(cv2, "CAP_PROP_HW_ACCELERATION", None)
        software_value = getattr(cv2, "VIDEO_ACCELERATION_NONE", None)
        api_preference = getattr(cv2, "CAP_FFMPEG", 0)
        if hw_prop is not None and software_value is not None:
            params = [int(hw_prop), int(software_value)]
            thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
            if thread_property is not None:
                params.extend([int(thread_property), decoder_threads])
            try:
                capture = cv2.VideoCapture(
                    input_path,
                    api_preference,
                    params,
                )
            except (TypeError, cv2.error):
                capture = None
            if capture is not None:
                if capture.isOpened():
                    return capture, "software"
                capture.release()

    decode_mode = "default_after_software_attempts" if prefer_software_decode else "default"
    try:
        return (
            _open_ffmpeg_capture_with_options(
                input_path,
                capture_options,
                decoder_threads=decoder_threads,
            ),
            decode_mode,
        )
    except (TypeError, cv2.error):
        return cv2.VideoCapture(input_path), decode_mode


def _video_dimensions(video, input_path: str) -> tuple[int, int]:
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width > 0 and height > 0:
        return width, height

    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return 0, 0
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        input_path,
    ]
    try:
        result = subprocess.run(command, check=False, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return 0, 0
    if result.returncode != 0:
        return 0, 0
    line = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    parts = line.split("x", 1)
    if len(parts) != 2:
        return 0, 0
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        return 0, 0


def _decode_with_ffmpeg_cli(
    input_path: str,
    *,
    width: int,
    height: int,
    frame_callback: Callable[[np.ndarray], None],
    decoder_threads: int,
) -> tuple[int, str]:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg executable is not available for software video decode fallback.")
    if width < 1 or height < 1:
        raise RuntimeError("Cannot use ffmpeg fallback because video width/height could not be determined.")

    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-filter_threads",
        str(max(1, int(decoder_threads))),
        "-hwaccel",
        "none",
        "-threads:v",
        str(max(1, int(decoder_threads))),
        "-i",
        input_path,
        "-map",
        "0:v:0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "-",
    ]
    frame_nbytes = width * height
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError as exc:
        raise RuntimeError(f"Could not start ffmpeg software decode fallback: {exc}") from exc

    frames_read = 0
    stderr_text = ""
    try:
        assert process.stdout is not None
        while True:
            chunk = process.stdout.read(frame_nbytes)
            if not chunk:
                break
            if len(chunk) != frame_nbytes:
                raise RuntimeError(
                    f"ffmpeg software decode fallback produced an incomplete frame "
                    f"({len(chunk)} of {frame_nbytes} bytes)."
                )
            frame = np.frombuffer(chunk, dtype=np.uint8).reshape((height, width)).copy()
            frame_callback(frame)
            frames_read += 1
    finally:
        if process.stdout is not None:
            process.stdout.close()
        if process.stderr is not None:
            stderr_text = process.stderr.read().decode("utf-8", errors="replace").strip()
            process.stderr.close()
        return_code = process.wait()

    if return_code != 0 and frames_read == 0:
        message = stderr_text or f"ffmpeg exited with status {return_code}"
        raise RuntimeError(f"ffmpeg software decode fallback failed: {message}")
    return frames_read, stderr_text


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


def is_supported_image_file(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in IMAGE_FRAME_EXTENSIONS


def is_supported_video_file(path: str | os.PathLike[str]) -> bool:
    return Path(path).suffix.lower() in VIDEO_FRAME_EXTENSIONS


def _direct_image_files(folder_path: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in folder_path.iterdir()
            if path.is_file() and is_supported_image_file(path)
        ),
        key=lambda path: path.name.lower(),
    )


def discover_ingest_sources(
    input_path: str | os.PathLike[str],
    *,
    recursive: bool = True,
) -> list[IngestSource]:
    """
    Discover video files and image-sequence folders below a path.

    Image folders are represented as one source per directory containing
    supported image files directly. Video files are represented individually.
    """
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)

    if root.is_file():
        if is_supported_video_file(root):
            return [IngestSource(path=root, kind="video")]
        if is_supported_image_file(root):
            return [IngestSource(path=root.parent, kind="image_sequence")]
        raise ValueError(f"Unsupported ingest file type: {root.suffix or root.name}")

    if not root.is_dir():
        raise ValueError(f"Ingest path is neither a file nor a directory: {root}")

    sources: list[IngestSource] = []
    folders = [root]
    if recursive:
        folders.extend(path for path in root.rglob("*") if path.is_dir())

    seen_folders: set[Path] = set()
    for folder in folders:
        if folder in seen_folders:
            continue
        if _direct_image_files(folder):
            sources.append(
                IngestSource(
                    path=folder,
                    kind="image_sequence",
                    recursive=False,
                )
            )
            seen_folders.add(folder)

    video_iterator = root.rglob("*") if recursive else root.iterdir()
    for path in video_iterator:
        if path.is_file() and is_supported_video_file(path):
            sources.append(IngestSource(path=path, kind="video"))

    return sorted(sources, key=lambda source: (source.kind, str(source.path).lower()))


def list_image_frame_files(folder_path: str | os.PathLike[str], *, recursive: bool = False) -> list[Path]:
    """Return image files in stable frame order for an image-sequence folder."""
    root = Path(folder_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    if not root.is_dir():
        raise NotADirectoryError(root)

    iterator = root.rglob("*") if recursive else root.iterdir()
    files = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_FRAME_EXTENSIONS
    ]
    return sorted(files, key=lambda path: (str(path.parent.relative_to(root)), path.name.lower()))


def ingest(
    input_path,
    *,
    recursive: bool = True,
    context: AppContext | None = None,
    run_id: str | None = None,
    asset_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    n_tile: int | None = None,
    adaptive_background_subtraction: bool | None = None,
    adaptive_background_period: int | None = None,
    apply_mask: bool | None = None,
    mask_path: str | None = None,
    progress_callback: IngestProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Generic direct-ingest dispatcher for a single registered source."""
    sources = discover_ingest_sources(input_path, recursive=recursive)
    if not sources:
        return []
    if len(sources) > 1:
        raise ValueError(
            "Direct ingest requires one source. Use discover_ingest_sources() "
            "to register and queue multiple discovered sources as separate assets."
        )

    source = sources[0]
    if source.kind == "image_sequence":
        return ingest_image_folder(
            source.path,
            recursive=source.recursive,
            context=context,
            run_id=run_id,
            asset_id=asset_id,
            metadata=metadata,
            progress_callback=progress_callback,
        )
    if source.kind == "video":
        return ingest_video_file(
            source.path,
            n_tile=n_tile,
            context=context,
            run_id=run_id,
            asset_id=asset_id,
            metadata=metadata,
            adaptive_background_subtraction=adaptive_background_subtraction,
            adaptive_background_period=adaptive_background_period,
            apply_mask=apply_mask,
            mask_path=mask_path,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unsupported ingest source kind: {source.kind!r}")


def ingest_image_folder(
    folder_path,
    *,
    recursive: bool = False,
    context: AppContext | None = None,
    run_id: str | None = None,
    asset_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    progress_callback: IngestProgressCallback | None = None,
) -> list[dict[str, Any]]:
    """Ingest a folder of image files as one stored frame per image."""
    started = time.perf_counter()
    root = Path(folder_path).expanduser().resolve()
    image_files = list_image_frame_files(root, recursive=recursive)
    ingest_payload = {
        "input_path": str(root),
        "source_path": str(root),
        "recursive": bool(recursive),
        "image_extension_count": len(IMAGE_FRAME_EXTENSIONS),
        "source_frame_count": len(image_files),
    }
    _CORE_LOGGER.info(
        "Starting image folder ingest for %s run_id=%s asset_id=%s frames=%s",
        root,
        run_id,
        asset_id,
        len(image_files),
    )
    _log_database_event(
        context,
        "info",
        "image_folder_ingest.started",
        "Image folder ingest started",
        run_id=run_id,
        asset_id=asset_id,
        payload=ingest_payload,
    )
    _emit_ingest_progress(
        progress_callback,
        "started",
        {**ingest_payload, "source_frames_read": 0, "stored_frame_count": 0},
    )

    if not image_files:
        duration_ms = (time.perf_counter() - started) * 1000
        _log_database_event(
            context,
            "warning",
            "image_folder_ingest.empty",
            "Image folder contained no supported image files",
            run_id=run_id,
            asset_id=asset_id,
            duration_ms=duration_ms,
            payload=ingest_payload,
        )
        _emit_ingest_progress(
            progress_callback,
            "completed",
            {**ingest_payload, "source_frames_read": 0, "stored_frame_count": 0},
        )
        return []

    frame_metadata = dict(metadata or {})
    if run_id is not None:
        frame_metadata["run_id"] = run_id
    if asset_id is not None:
        frame_metadata["asset_id"] = asset_id
    frame_metadata["source_folder"] = str(root)
    frame_metadata["source_type"] = "image_folder"
    frame_metadata["recursive"] = bool(recursive)

    stored_frames: list[dict[str, Any]] = []
    try:
        for frame_index, image_path in enumerate(image_files, start=1):
            with measure_phase("ingest.image_decode"):
                image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError(f"Could not read image file: {image_path}")
            with measure_phase("ingest.grayscale"):
                image = convert_frame_to_grayscale(image)
            metadata_for_frame = frame_metadata.copy()
            metadata_for_frame.update(
                {
                    "frame_index": frame_index,
                    "source_image_path": str(image_path),
                    "source_image_relative_path": str(image_path.relative_to(root)),
                    "source_image_filename": image_path.name,
                    "image_extension": image_path.suffix.lower(),
                }
            )
            stored_frames.append(
                store_frame(
                    FrameData(
                        sourcePath=str(image_path.parent),
                        filename=image_path.name,
                        frameNumber=frame_index,
                        data=image,
                        tileNumber=frame_index,
                        sourceFrameStart=frame_index,
                        sourceFrameEnd=frame_index,
                        frameType="image",
                        metadata=metadata_for_frame,
                    ),
                    context=context,
                )
            )
            if frame_index == 1 or frame_index % _PROGRESS_LOG_TILE_INTERVAL == 0:
                _log_database_event(
                    context,
                    "debug",
                    "image_folder_ingest.frame_stored",
                    "Image folder ingest frame stored",
                    run_id=run_id,
                    asset_id=asset_id,
                    payload={
                        **ingest_payload,
                        "frame_index": frame_index,
                        "filename": image_path.name,
                        "stored_frame_count": len(stored_frames),
                    },
                )
            _emit_ingest_progress(
                progress_callback,
                "frame_stored",
                {
                    **ingest_payload,
                    "frame_index": frame_index,
                    "source_frames_read": frame_index,
                    "stored_frame_count": len(stored_frames),
                    "filename": image_path.name,
                    "source_image_relative_path": str(image_path.relative_to(root)),
                },
            )
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        _CORE_LOGGER.exception(
            "Image folder ingest failed for %s run_id=%s asset_id=%s",
            root,
            run_id,
            asset_id,
        )
        _log_database_event(
            context,
            "error",
            "image_folder_ingest.failed",
            "Image folder ingest failed",
            run_id=run_id,
            asset_id=asset_id,
            duration_ms=duration_ms,
            payload={
                **ingest_payload,
                "stored_frame_count": len(stored_frames),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        _emit_ingest_progress(
            progress_callback,
            "failed",
            {
                **ingest_payload,
                "source_frames_read": len(stored_frames),
                "stored_frame_count": len(stored_frames),
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
        )
        raise

    duration_ms = (time.perf_counter() - started) * 1000
    _CORE_LOGGER.info(
        "Completed image folder ingest for %s run_id=%s asset_id=%s frames=%s duration_ms=%.2f",
        root,
        run_id,
        asset_id,
        len(stored_frames),
        duration_ms,
    )
    _log_database_event(
        context,
        "info",
        "image_folder_ingest.completed",
        "Image folder ingest completed",
        run_id=run_id,
        asset_id=asset_id,
        duration_ms=duration_ms,
        payload={
            **ingest_payload,
            "stored_frame_count": len(stored_frames),
        },
    )
    _emit_ingest_progress(
        progress_callback,
        "completed",
        {
            **ingest_payload,
            "source_frames_read": len(image_files),
            "stored_frame_count": len(stored_frames),
        },
    )
    return stored_frames


def ingest_video_file(
    input_path,
    n_tile=None,
    *,
    context: AppContext | None = None,
    run_id: str | None = None,
    asset_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    adaptive_background_subtraction: bool | None = None,
    adaptive_background_period: int | None = None,
    apply_mask: bool | None = None,
    mask_path: str | None = None,
    opencv_threads: int | None = None,
    decoder_threads: int | None = None,
    progress_callback: IngestProgressCallback | None = None,
) -> list[dict[str, Any]]:
    started = time.perf_counter()
    ingest_defaults = (
        context.config.processing.video_ingest
        if context is not None
        else default_processing_config().video_ingest
    )
    preprocessing_defaults = (
        context.config.processing.preprocessing
        if context is not None
        else default_processing_config().preprocessing
    )
    n_tile = ingest_defaults.n_tile if n_tile is None else n_tile
    prefer_software_decode = bool(ingest_defaults.prefer_software_decode)
    opencv_threads = max(
        1,
        int(ingest_defaults.opencv_threads if opencv_threads is None else opencv_threads),
    )
    decoder_threads = max(
        1,
        int(ingest_defaults.decoder_threads if decoder_threads is None else decoder_threads),
    )
    cv2.setNumThreads(opencv_threads)
    adaptive_background_subtraction = (
        preprocessing_defaults.adaptive_background_subtraction
        if adaptive_background_subtraction is None
        else adaptive_background_subtraction
    )
    adaptive_background_period = (
        preprocessing_defaults.adaptive_background_period
        if adaptive_background_period is None
        else adaptive_background_period
    )
    apply_mask = preprocessing_defaults.apply_mask if apply_mask is None else apply_mask
    mask_path = preprocessing_defaults.mask_path if mask_path is None else mask_path

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
        "prefer_software_decode": prefer_software_decode,
        "opencv_threads": opencv_threads,
        "decoder_threads": decoder_threads,
        "adaptive_background_subtraction": bool(adaptive_background_subtraction),
        "adaptive_background_period": int(adaptive_background_period),
        "apply_mask": bool(apply_mask),
        "mask_path": mask_path,
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
    _emit_ingest_progress(
        progress_callback,
        "started",
        {**ingest_payload, "source_frames_read": 0, "stored_tile_count": 0},
    )
    frame_metadata = dict(metadata or {})
    if run_id is not None:
        frame_metadata["run_id"] = run_id
    if asset_id is not None:
        frame_metadata["asset_id"] = asset_id
    frame_metadata["adaptive_background_subtraction"] = bool(adaptive_background_subtraction)
    frame_metadata["adaptive_background_period"] = int(adaptive_background_period)
    frame_metadata["apply_mask"] = bool(apply_mask)
    frame_metadata["mask_path"] = mask_path

    with measure_phase("ingest.video_open"):
        video, video_decode_mode = _open_video_capture(
            input_path,
            prefer_software_decode=prefer_software_decode,
            decoder_threads=decoder_threads,
        )
    ingest_payload["video_decode_mode"] = video_decode_mode
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
        _emit_ingest_progress(
            progress_callback,
            "failed",
            {
                **ingest_payload,
                "source_frames_read": 0,
                "stored_tile_count": 0,
                "error_type": "ValueError",
                "error_message": f"Could not open video file: {input_path}",
            },
        )
        raise ValueError(f"Could not open video file: {input_path}")

    fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
    source_frame_count = max(0, int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0))
    estimated_tile_count = _estimated_tile_count(source_frame_count, int(n_tile))
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
            "source_frame_count": source_frame_count,
            "estimated_tile_count": estimated_tile_count,
            "video_decode_mode": video_decode_mode,
            "source_timestamp_utc": None if start_timestamp is None else start_timestamp.isoformat(),
        },
    )
    _emit_ingest_progress(
        progress_callback,
        "video_opened",
        {
            **ingest_payload,
            "fps": fps,
            "source_frame_count": source_frame_count,
            "estimated_tile_count": estimated_tile_count,
            "video_decode_mode": video_decode_mode,
            "source_frames_read": 0,
            "stored_tile_count": 0,
            "source_timestamp_utc": None if start_timestamp is None else start_timestamp.isoformat(),
        },
    )

    frame_buffer = []
    stored_frames = []
    n = 1
    tile_number = 1

    def store_tile(tile: np.ndarray, source_start: int, source_end: int, *, partial_tile: bool = False) -> None:
        nonlocal tile_number
        stored_frames.append(
            store_frame(
                FrameData(
                    sourcePath=source_path,
                    filename=filename,
                    frameNumber=source_end,
                    data=tile,
                    tileNumber=tile_number,
                    sourceFrameStart=source_start,
                    sourceFrameEnd=source_end,
                    frameType="line",
                    timestamp=timestamp_for_frame(start_timestamp, fps, source_end),
                    metadata=frame_metadata.copy(),
                ),
                context=context,
            )
        )
        event_payload = {
            "filename": filename,
            "tile_number": tile_number,
            "source_frame_start": source_start,
            "source_frame_end": source_end,
            "stored_tile_count": len(stored_frames),
        }
        if partial_tile:
            event_payload["partial_tile"] = True
        if partial_tile or tile_number == 1 or tile_number % _PROGRESS_LOG_TILE_INTERVAL == 0:
            _log_database_event(
                context,
                "debug",
                "video_ingest.tile_stored",
                "Video ingest final tile stored" if partial_tile else "Video ingest tile stored",
                run_id=run_id,
                asset_id=asset_id,
                payload=event_payload,
            )
        _emit_ingest_progress(
            progress_callback,
            "tile_stored",
            {
                **ingest_payload,
                "fps": fps,
                "source_frame_count": source_frame_count,
                "estimated_tile_count": estimated_tile_count,
                "source_frames_read": source_end,
                **event_payload,
            },
        )
        tile_number += 1

    def store_source_frame(frame: np.ndarray) -> None:
        nonlocal frame_buffer, n
        with measure_phase("ingest.grayscale"):
            frame = convert_frame_to_grayscale(frame)
        frame_buffer.append(frame)
        if len(frame_buffer) == n_tile:
            source_start = n - len(frame_buffer) + 1
            source_end = n
            # Multi-frame tiles are stacked vertically to preserve the legacy line-scan layout.
            with measure_phase("ingest.tile_assembly"):
                tile = np.vstack(frame_buffer)
            store_tile(tile, source_start, source_end)
            frame_buffer = []
        n += 1

    def flush_partial_tile() -> None:
        nonlocal frame_buffer
        if not frame_buffer:
            return
        source_start = n - len(frame_buffer)
        source_end = n - 1
        with measure_phase("ingest.tile_assembly"):
            tile = np.vstack(frame_buffer)
        store_tile(tile, source_start, source_end, partial_tile=True)
        frame_buffer = []

    try:
        while video.isOpened():
            with measure_phase("ingest.video_decode"):
                good_return, frame = video.read()
            if not good_return:
                break
            if frame is not None:
                store_source_frame(frame)

        # Retry only when OpenCV opened the source but produced no usable frames.
        if n == 1 and prefer_software_decode:
            width, height = _video_dimensions(video, input_path)
            fallback_payload = {
                **ingest_payload,
                "opencv_video_decode_mode": video_decode_mode,
                "ffmpeg_width": width,
                "ffmpeg_height": height,
            }
            _log_database_event(
                context,
                "warning",
                "video_ingest.decoder_fallback",
                "OpenCV decoded zero frames; trying ffmpeg software fallback",
                run_id=run_id,
                asset_id=asset_id,
                payload=fallback_payload,
            )
            _emit_ingest_progress(
                progress_callback,
                "decoder_fallback",
                {
                    **fallback_payload,
                    "source_frames_read": 0,
                    "stored_tile_count": 0,
                },
            )
            video_decode_mode = "ffmpeg_cli_software_fallback"
            ingest_payload["video_decode_mode"] = video_decode_mode
            ffmpeg_frames_read, ffmpeg_stderr = _decode_with_ffmpeg_cli(
                input_path,
                width=width,
                height=height,
                frame_callback=store_source_frame,
                decoder_threads=decoder_threads,
            )
            flush_partial_tile()
            fallback_completed_payload = {
                **ingest_payload,
                "ffmpeg_frames_read": ffmpeg_frames_read,
                "ffmpeg_stderr": ffmpeg_stderr,
                "source_frames_read": max(0, n - 1),
                "stored_tile_count": len(stored_frames),
            }
            _log_database_event(
                context,
                "info",
                "video_ingest.decoder_fallback_completed",
                "ffmpeg software fallback decoded video frames",
                run_id=run_id,
                asset_id=asset_id,
                payload=fallback_completed_payload,
            )
            _emit_ingest_progress(
                progress_callback,
                "decoder_fallback_completed",
                fallback_completed_payload,
            )

        flush_partial_tile()
        if n == 1:
            raise ValueError(
                f"Decoded zero frames from video: {input_path}. "
                f"OpenCV decode mode was {video_decode_mode!r}."
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
        _emit_ingest_progress(
            progress_callback,
            "failed",
            {
                **ingest_payload,
                "source_frame_count": source_frame_count,
                "estimated_tile_count": estimated_tile_count,
                "source_frames_read": max(0, n - 1),
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
            "source_frame_count": max(source_frame_count, max(0, n - 1)),
            "estimated_tile_count": estimated_tile_count,
            "stored_tile_count": len(stored_frames),
        },
    )
    _emit_ingest_progress(
        progress_callback,
        "completed",
        {
            **ingest_payload,
            "fps": fps,
            "source_frame_count": max(source_frame_count, max(0, n - 1)),
            "estimated_tile_count": estimated_tile_count,
            "source_frames_read": max(0, n - 1),
            "stored_tile_count": len(stored_frames),
        },
    )
    return stored_frames
