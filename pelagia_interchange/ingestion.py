"""Optional FFmpeg-backed ingestion of video directories."""
from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Sequence

from .builder import DatasetBuilder
from .dataset import Dataset
from .exceptions import InterchangeError
from .metadata import Metadata
from .models import StorageFormat


DEFAULT_FFMPEG_QSCALE = 2
from .util import hash_file

DEFAULT_VIDEO_EXTENSIONS = (".avi", ".mov", ".mp4", ".m4v", ".mkv", ".mpg", ".mpeg", ".mts", ".m2ts")


class VideoIngestionError(InterchangeError):
    """Video discovery, probing, or decoding could not complete safely."""


@dataclass(frozen=True, slots=True)
class VideoProbe:
    path: Path
    container: str | None
    codec: str | None
    pixel_format: str | None
    width: int | None
    height: int | None
    frame_rate: tuple[int, int] | None
    frame_count: int
    creation_time: str | None


@dataclass(frozen=True, slots=True)
class VideoIngestionResult:
    dataset: Dataset
    source_files: int
    frames: int
    previews: int


def discover_videos(directory: str | Path, *, recursive: bool = False, extensions: Sequence[str] = DEFAULT_VIDEO_EXTENSIONS) -> list[Path]:
    root = Path(directory)
    if not root.is_dir():
        raise VideoIngestionError(f"video input directory does not exist: {root}")
    allowed = {value.lower() if value.startswith(".") else "." + value.lower() for value in extensions}
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted((path for path in iterator if path.is_file() and path.suffix.lower() in allowed),
                  key=lambda path: path.relative_to(root).as_posix().casefold())


def _executable(value: str, label: str) -> str:
    resolved = shutil.which(value)
    if resolved is None:
        raise VideoIngestionError(f"{label} executable was not found: {value!r}")
    return resolved


def _fraction(value: str | None) -> tuple[int, int] | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        numerator, denominator = value.split("/", 1)
        result = (int(numerator), int(denominator))
    except (ValueError, TypeError) as exc:
        raise VideoIngestionError(f"invalid probed frame rate {value!r}") from exc
    return result if result[1] else None


def probe_video(path: str | Path, *, ffprobe: str = "ffprobe") -> VideoProbe:
    executable = _executable(ffprobe, "ffprobe")
    command = [executable, "-v", "error", "-count_frames", "-select_streams", "v:0",
               "-show_entries", "format=format_name:format_tags=creation_time:stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,nb_read_frames:stream_tags=creation_time",
               "-of", "json", str(Path(path))]
    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode:
        raise VideoIngestionError(f"ffprobe failed for {path}: {completed.stderr.strip() or 'unknown error'}")
    try:
        document = json.loads(completed.stdout)
        stream = document["streams"][0]
        container = document.get("format", {})
        count_value = stream.get("nb_read_frames") or stream.get("nb_frames")
        frame_count = int(count_value)
    except (json.JSONDecodeError, KeyError, IndexError, TypeError, ValueError) as exc:
        raise VideoIngestionError(f"ffprobe did not report a usable video stream and frame count for {path}") from exc
    if frame_count <= 0:
        raise VideoIngestionError(f"ffprobe reported no decodable video frames for {path}")
    creation_time = stream.get("tags", {}).get("creation_time") or container.get("tags", {}).get("creation_time")
    return VideoProbe(Path(path), container.get("format_name"), stream.get("codec_name"), stream.get("pix_fmt"),
                      stream.get("width"), stream.get("height"), _fraction(stream.get("avg_frame_rate")),
                      frame_count, creation_time)


def _iter_mjpeg(stream: object, *, maximum_frame_bytes: int = 256 * 1024 * 1024) -> Iterator[bytes]:
    buffer = bytearray()
    read = getattr(stream, "read")
    while chunk := read(1024 * 1024):
        buffer.extend(chunk)
        while True:
            start = buffer.find(b"\xff\xd8")
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break
            if start:
                del buffer[:start]
            end = buffer.find(b"\xff\xd9", 2)
            if end < 0:
                if len(buffer) > maximum_frame_bytes:
                    raise VideoIngestionError("an encoded frame exceeded the safety limit")
                break
            end += 2
            yield bytes(buffer[:end])
            del buffer[:end]
    if buffer:
        raise VideoIngestionError("FFmpeg produced a truncated JPEG frame")


def _ffmpeg_version(executable: str) -> str | None:
    completed = subprocess.run([executable, "-version"], check=False, capture_output=True, text=True)
    first = completed.stdout.splitlines()[0].split() if completed.stdout else []
    return first[2] if len(first) >= 3 else None


def _passthrough_timing_arguments(executable: str) -> tuple[list[str], str]:
    """Select frame-passthrough syntax supported by the installed FFmpeg.

    FFmpeg 5 introduced the per-stream ``-fps_mode`` option. Older supported
    installations provide the equivalent global ``-vsync 0`` spelling.
    Inspecting the executable's own help is more reliable than parsing vendor
    version strings, which may include backported options.
    """
    completed = subprocess.run(
        [executable, "-hide_banner", "-h", "full"],
        check=False,
        capture_output=True,
        text=True,
    )
    help_text = f"{completed.stdout}\n{completed.stderr}"
    if "-fps_mode" in help_text:
        return ["-fps_mode", "passthrough"], "fps_mode_passthrough"
    return ["-vsync", "0"], "vsync_0"


def _thumbnail(payload: bytes, *, ffmpeg: str, width: int) -> bytes:
    command = [ffmpeg, "-v", "error", "-i", "pipe:0", "-vf", f"scale=w='min(iw,{width})':h=-2:flags=lanczos",
               "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "4", "-f", "image2pipe", "pipe:1"]
    completed = subprocess.run(command, input=payload, check=False, capture_output=True)
    if completed.returncode or not completed.stdout:
        raise VideoIngestionError(f"thumbnail generation failed: {completed.stderr.decode('utf-8', errors='replace').strip()}")
    return completed.stdout


def _contact_sheet(thumbnails: Sequence[bytes], *, ffmpeg: str) -> bytes:
    columns = min(4, len(thumbnails)); rows = math.ceil(len(thumbnails) / columns)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        for number, payload in enumerate(thumbnails):
            (root / f"thumb_{number:03d}.jpg").write_bytes(payload)
        command = [ffmpeg, "-v", "error", "-framerate", "1", "-start_number", "0",
                   "-i", str(root / "thumb_%03d.jpg"), "-vf", f"tile={columns}x{rows}:padding=4:margin=4:color=black",
                   "-frames:v", "1", "-c:v", "mjpeg", "-q:v", "4", "-f", "image2pipe", "pipe:1"]
        completed = subprocess.run(command, check=False, capture_output=True)
    if completed.returncode or not completed.stdout:
        raise VideoIngestionError(f"contact-sheet generation failed: {completed.stderr.decode('utf-8', errors='replace').strip()}")
    return completed.stdout


def ingest_video_directory(
    input_directory: str | Path, output: str | Path, *, title: str | None = None,
    description: str = "", stream: str = "camera", recursive: bool = False,
    shard_target_size: int | str = "10GB", metadata: Metadata | None = None,
    ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe", ffmpeg_qscale: int = DEFAULT_FFMPEG_QSCALE,
    grayscale: bool = False, hash_sources: bool = True, source_file_boundary: bool = False,
    generate_previews: bool = True, preview_count: int = 12, preview_width: int = 512,
    require_previews: bool = False,
    progress: Callable[[str], None] | None = None,
) -> VideoIngestionResult:
    """Discover, probe, transcode, and archive all supported videos in a directory."""
    root = Path(input_directory).resolve()
    videos = discover_videos(root, recursive=recursive)
    if not videos:
        raise VideoIngestionError(f"no supported video files were found in {root}")
    ffmpeg_path = _executable(ffmpeg, "ffmpeg")
    ffprobe_path = _executable(ffprobe, "ffprobe")
    if not 2 <= ffmpeg_qscale <= 31:
        raise ValueError("ffmpeg_qscale must be between 2 and 31")
    if require_previews and not generate_previews:
        raise ValueError("require_previews cannot be used when preview generation is disabled")
    if generate_previews and preview_count < 1:
        raise ValueError("preview_count must be positive")
    if generate_previews and not 32 <= preview_width <= 8192:
        raise ValueError("preview_width must be between 32 and 8192")
    encoder_version = _ffmpeg_version(ffmpeg_path)
    timing_arguments, timing_mode = _passthrough_timing_arguments(ffmpeg_path)
    if progress and timing_mode == "vsync_0":
        progress(
            f"FFmpeg {encoder_version or 'unknown'} does not advertise -fps_mode; "
            "using compatible -vsync 0 frame passthrough"
        )
    storage = StorageFormat(codec="jpeg", quality=None, pixel_format="gray8" if grayscale else "yuvj444p",
                            bit_depth=8, encoder="ffmpeg mjpeg", encoder_version=encoder_version,
                            parameters={"qscale": ffmpeg_qscale, "grayscale": grayscale,
                                        "frame_sync_mode": timing_mode},
                            description=f"FFmpeg MJPEG qscale {ffmpeg_qscale}, {'grayscale' if grayscale else 'color'}")
    probes: list[VideoProbe] = []
    for index, video in enumerate(videos, 1):
        if progress:
            progress(f"[{index}/{len(videos)}] probing {video.relative_to(root)}")
        probes.append(probe_video(video, ffprobe=ffprobe_path))
    expected_total = sum(probe.frame_count for probe in probes)
    sample_count = min(preview_count, expected_total)
    selected = ({round(number * (expected_total - 1) / (sample_count - 1)) for number in range(sample_count)}
                if generate_previews and sample_count > 1 else ({0} if generate_previews else set()))
    total_frames = 0; preview_entries: list[dict[str, object]] = []; thumbnail_payloads: list[bytes] = []
    preview_warnings: list[str] = []
    with DatasetBuilder(output, title=title or Path(output).name, description=description,
                        shard_target_size=shard_target_size, source_file_boundary=source_file_boundary,
                        metadata=metadata) as builder:
        if metadata is not None:
            if title:
                builder.metadata.data.setdefault("dataset", {})["title"] = title
            if description:
                builder.metadata.data.setdefault("dataset", {})["description"] = description
        streams = builder.metadata.data.setdefault("streams", [])
        existing_stream = next((item for item in streams if item.get("name") == stream), None)
        stream_uuid = builder.register_stream(stream, stream_uuid=existing_stream.get("stream_uuid") if existing_stream else None)
        if existing_stream is None:
            streams.append({"stream_uuid": str(stream_uuid), "name": stream,
                            "storage_description": storage.description,
                            "timestamp": {"source": "unknown", "interpolated": False}})
        else:
            existing_stream.setdefault("stream_uuid", str(stream_uuid))
            existing_stream.setdefault("storage_description", storage.description)
            existing_stream.setdefault("timestamp", {"source": "unknown", "interpolated": False})
        next_frame_id = 0
        for index, (video, probe) in enumerate(zip(videos, probes), 1):
            if progress:
                progress(f"[{index}/{len(videos)}] hashing {video.relative_to(root)}")
            source_hash = hash_file(video) if hash_sources else None
            source = builder.register_source_file(
                path=video, original_relative_path=video.relative_to(root).as_posix(), sha256=source_hash,
                container=probe.container, codec=probe.codec, pixel_format=probe.pixel_format,
                width=probe.width, height=probe.height, frame_rate=probe.frame_rate,
                frame_count=probe.frame_count, start_timestamp=probe.creation_time,
            )
            if progress:
                progress(f"[{index}/{len(videos)}] transcoding {probe.frame_count:,} frames from {video.name}")
            with tempfile.TemporaryFile(mode="w+b") as errors:
                command = [ffmpeg_path, "-v", "error", "-xerror", "-err_detect", "explode", "-i", str(video),
                           "-map", "0:v:0", "-an", "-sn", "-dn", *timing_arguments,
                           "-c:v", "mjpeg", "-q:v", str(ffmpeg_qscale), "-pix_fmt", "gray" if grayscale else "yuvj444p",
                           "-f", "image2pipe", "pipe:1"]
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
                assert process.stdout is not None
                decoded = 0
                try:
                    for source_frame_number, payload in enumerate(_iter_mjpeg(process.stdout)):
                        frame_id = next_frame_id
                        builder.add_frame(stream=stream, source_file=source, frame_id=next_frame_id,
                                          source_frame_number=source_frame_number, encoded_bytes=payload,
                                          storage_format=storage, width=probe.width, height=probe.height,
                                          timestamp_source="unknown")
                        if frame_id in selected:
                            try:
                                thumbnail = _thumbnail(payload, ffmpeg=ffmpeg_path, width=preview_width)
                                relative_name = f"streams/{DatasetBuilder._slug(stream)}/frame_{frame_id:012d}.jpg"
                                builder.add_resource_bytes(
                                    thumbnail, kind="preview", relative_name=relative_name,
                                    description=f"Non-authoritative representative thumbnail for retained frame {frame_id}",
                                    attributes={"preview_kind": "representative_thumbnail", "stream_uuid": str(stream_uuid),
                                                "frame_id": frame_id, "source_uuid": str(source.source_uuid),
                                                "source_frame_number": source_frame_number,
                                                "selection_method": "evenly_spaced_retained_frame_ids",
                                                "preview_width_max": preview_width},
                                )
                                preview_entries.append({"relative_path": f"preview/{relative_name}",
                                                        "stream_uuid": str(stream_uuid), "frame_id": frame_id,
                                                        "source_uuid": str(source.source_uuid),
                                                        "source_frame_number": source_frame_number,
                                                        "timestamp_ns": None})
                                thumbnail_payloads.append(thumbnail)
                            except VideoIngestionError as exc:
                                if require_previews:
                                    raise
                                preview_warnings.append(str(exc))
                        next_frame_id += 1; decoded += 1; total_frames += 1
                        if progress and decoded % 10_000 == 0:
                            progress(f"[{index}/{len(videos)}] {video.name}: {decoded:,}/{probe.frame_count:,} frames")
                except BaseException:
                    process.kill(); process.wait(); raise
                return_code = process.wait()
                errors.seek(0); error_text = errors.read().decode("utf-8", errors="replace").strip()
            if return_code:
                raise VideoIngestionError(f"FFmpeg failed for {video}: {error_text or f'exit code {return_code}'}")
            if decoded != probe.frame_count:
                raise VideoIngestionError(f"frame-count mismatch for {video}: ffprobe expected {probe.frame_count}, FFmpeg produced {decoded}; the incomplete dataset was preserved for review")
            builder.history.append(operation="frames_transcoded", software="ffmpeg", software_version=encoder_version or "unknown",
                                   parameters={"codec": "mjpeg", "qscale": ffmpeg_qscale, "grayscale": grayscale,
                                               "source_file_boundary": source_file_boundary,
                                               "frame_sync_mode": timing_mode},
                                   inputs=[{"source_uuid": str(source.source_uuid)}],
                                   outputs=[{"stream_uuid": str(stream_uuid), "first_frame": next_frame_id - decoded,
                                             "last_frame": next_frame_id - 1, "frame_count": decoded}])
        if generate_previews and preview_entries:
            index_document = {"preview_schema_version": "1", "authoritative": False,
                              "selection_method": "evenly_spaced_retained_frame_ids",
                              "requested_count": preview_count, "thumbnail_width_max": preview_width,
                              "stream_uuid": str(stream_uuid), "frames": preview_entries}
            builder.add_resource_bytes((json.dumps(index_document, indent=2, sort_keys=True) + "\n").encode(),
                                       kind="preview", relative_name="index.json",
                                       description="Mapping from non-authoritative previews to authoritative frame records",
                                       attributes={"preview_kind": "preview_index", "stream_uuid": str(stream_uuid)})
            try:
                contact_sheet = _contact_sheet(thumbnail_payloads, ffmpeg=ffmpeg_path)
                builder.add_resource_bytes(contact_sheet, kind="preview", relative_name=f"streams/{DatasetBuilder._slug(stream)}/contact_sheet.jpg",
                                           description="Non-authoritative contact sheet of evenly spaced representative frames",
                                           attributes={"preview_kind": "contact_sheet", "stream_uuid": str(stream_uuid),
                                                       "selection_method": "evenly_spaced_retained_frame_ids",
                                                       "frame_ids": [entry["frame_id"] for entry in preview_entries]})
            except VideoIngestionError as exc:
                if require_previews:
                    raise
                preview_warnings.append(str(exc))
            builder.history.append(operation="previews_generated", parameters={"requested_count": preview_count,
                                   "thumbnail_width_max": preview_width, "selection_method": "evenly_spaced_retained_frame_ids"},
                                   outputs=[{"preview_count": len(preview_entries), "stream_uuid": str(stream_uuid)}],
                                   status="warning" if preview_warnings else "success",
                                   message="; ".join(preview_warnings) if preview_warnings else None)
        elif generate_previews and require_previews:
            raise VideoIngestionError("required previews could not be generated")
        elif generate_previews:
            builder.history.append(operation="previews_generated", parameters={"requested_count": preview_count,
                                   "thumbnail_width_max": preview_width, "selection_method": "evenly_spaced_retained_frame_ids"},
                                   outputs=[{"preview_count": 0, "stream_uuid": str(stream_uuid)}], status="warning",
                                   message="; ".join(preview_warnings) or "no previews were generated")
    return VideoIngestionResult(Dataset.open(output), len(videos), total_frames, len(preview_entries))
