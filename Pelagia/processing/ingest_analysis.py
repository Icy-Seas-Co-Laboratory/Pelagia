from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from ..domain import AssetKind, RawAssetManifest, normalize_collections
from .ingest import (
    discover_ingest_sources,
    is_supported_image_file,
    is_supported_video_file,
    list_image_frame_files,
)
from .frame_time import parse_filename_timestamp_utc


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class AnalyzedAsset:
    asset_id: str
    filename: str
    path: str
    kind: str
    size_bytes: int
    checksum: str
    checksum_status: str
    collections: list[str]
    media_count: int
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "asset_id": self.asset_id,
            "filename": self.filename,
            "path": self.path,
            "kind": self.kind,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
            "checksum_status": self.checksum_status,
            "collections": self.collections,
            "media_count": self.media_count,
            "metadata": self.metadata,
            "warnings": self.warnings,
        }

    def as_manifest(self) -> RawAssetManifest:
        return RawAssetManifest(
            asset_id=self.asset_id,
            filename=self.filename,
            path=self.path,
            kind=AssetKind(self.kind),
            size_bytes=self.size_bytes,
            checksum=self.checksum,
            collections=self.collections,
            media_count=self.media_count,
            metadata=self.metadata,
        )


def _checksum(path: Path, *, compute_checksum: bool) -> tuple[str, str]:
    stat = path.stat()
    if compute_checksum and path.is_file():
        return f"sha256:{sha256_file(path)}", "computed"
    return f"uncomputed:size={stat.st_size}:mtime_ns={stat.st_mtime_ns}", "deferred"


def _ffprobe_video(path: Path) -> tuple[dict[str, Any], list[str]]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return {}, ["ffprobe executable was not found."]
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration,codec_name,pix_fmt",
        "-show_entries",
        "format=duration,bit_rate,format_name",
        "-of",
        "json",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    except Exception as exc:
        return {}, [f"ffprobe failed: {exc}"]
    if result.returncode != 0:
        message = result.stderr.strip() or f"ffprobe exited with status {result.returncode}"
        return {}, [message]
    try:
        payload = json.loads(result.stdout or "{}")
    except json.JSONDecodeError as exc:
        return {}, [f"ffprobe returned invalid JSON: {exc}"]
    streams = payload.get("streams") or []
    stream = streams[0] if streams else {}
    fmt = payload.get("format") or {}
    metadata: dict[str, Any] = {
        "probe_source": "ffprobe",
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "format_name": fmt.get("format_name"),
        "bit_rate": _int_or_none(fmt.get("bit_rate")),
        "width": _int_or_none(stream.get("width")),
        "height": _int_or_none(stream.get("height")),
        "fps": _rate_to_float(stream.get("avg_frame_rate") or stream.get("r_frame_rate")),
        "duration_seconds": _float_or_none(stream.get("duration") or fmt.get("duration")),
        "source_frame_count": _int_or_none(stream.get("nb_frames")),
    }
    return {key: value for key, value in metadata.items() if value is not None}, []


def _opencv_video_probe(path: Path) -> tuple[dict[str, Any], list[str]]:
    video = cv2.VideoCapture(str(path))
    try:
        if not video.isOpened():
            return {}, ["OpenCV could not open the video."]
        fps = float(video.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        return {
            "opencv_fps": fps or None,
            "opencv_source_frame_count": frame_count or None,
            "opencv_width": width or None,
            "opencv_height": height or None,
        }, []
    finally:
        video.release()


def _rate_to_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value)
    if "/" in text:
        numerator, denominator = text.split("/", 1)
        try:
            denom = float(denominator)
            return None if denom == 0 else float(numerator) / denom
        except ValueError:
            return None
    return _float_or_none(text)


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _image_dimensions(path: Path) -> tuple[int | None, int | None, str | None]:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None, None, f"Could not read sample image {path.name!r}."
    height, width = image.shape[:2]
    return int(width), int(height), None


def analyze_video_asset(
    path: Path,
    *,
    collections: Any = None,
    compute_checksum: bool = False,
    metadata: dict[str, Any] | None = None,
) -> AnalyzedAsset:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    checksum, checksum_status = _checksum(resolved, compute_checksum=compute_checksum)
    probe_metadata, warnings = _ffprobe_video(resolved)
    opencv_metadata, opencv_warnings = _opencv_video_probe(resolved)
    warnings.extend(opencv_warnings)
    if opencv_metadata:
        probe_metadata.update({key: value for key, value in opencv_metadata.items() if value is not None})
    if "source_frame_count" not in probe_metadata and probe_metadata.get("opencv_source_frame_count"):
        probe_metadata["source_frame_count"] = probe_metadata["opencv_source_frame_count"]
    if "width" not in probe_metadata and probe_metadata.get("opencv_width"):
        probe_metadata["width"] = probe_metadata["opencv_width"]
    if "height" not in probe_metadata and probe_metadata.get("opencv_height"):
        probe_metadata["height"] = probe_metadata["opencv_height"]
    if "fps" not in probe_metadata and probe_metadata.get("opencv_fps"):
        probe_metadata["fps"] = probe_metadata["opencv_fps"]
    timestamp = parse_filename_timestamp_utc(resolved.name)
    asset_metadata = {
        **dict(metadata or {}),
        **probe_metadata,
        "analysis_endpoint": "POST /ingestion/analyze",
        "source_type": AssetKind.VIDEO.value,
        "checksum_status": checksum_status,
    }
    if timestamp is not None:
        asset_metadata.setdefault("source_timestamp_utc", timestamp.isoformat())
    return AnalyzedAsset(
        asset_id=str(uuid.uuid4()),
        filename=resolved.name,
        path=str(resolved),
        kind=AssetKind.VIDEO.value,
        size_bytes=stat.st_size,
        checksum=checksum,
        checksum_status=checksum_status,
        collections=normalize_collections(collections),
        media_count=int(asset_metadata.get("source_frame_count") or 0),
        metadata=asset_metadata,
        warnings=warnings,
    )


def analyze_image_sequence_asset(
    path: Path,
    *,
    recursive: bool = False,
    collections: Any = None,
    compute_checksum: bool = False,
    metadata: dict[str, Any] | None = None,
) -> AnalyzedAsset:
    resolved = path.expanduser().resolve()
    files = list_image_frame_files(resolved, recursive=recursive)
    size_bytes = sum(file.stat().st_size for file in files)
    checksum = f"uncomputed:size={size_bytes}:file_count={len(files)}"
    checksum_status = "deferred"
    if compute_checksum:
        digest = hashlib.sha256()
        for file in files:
            digest.update(str(file.relative_to(resolved)).encode("utf-8"))
            digest.update(file.read_bytes())
        checksum = f"sha256:{digest.hexdigest()}"
        checksum_status = "computed"
    warnings: list[str] = []
    width = height = None
    if files:
        width, height, warning = _image_dimensions(files[0])
        if warning:
            warnings.append(warning)
    asset_metadata = {
        **dict(metadata or {}),
        "analysis_endpoint": "POST /ingestion/analyze",
        "source_type": AssetKind.IMAGE_SEQUENCE.value,
        "source_folder": str(resolved),
        "recursive": bool(recursive),
        "checksum_status": checksum_status,
        "source_frame_count": len(files),
        "first_image_filename": None if not files else files[0].name,
        "last_image_filename": None if not files else files[-1].name,
        "width": width,
        "height": height,
    }
    return AnalyzedAsset(
        asset_id=str(uuid.uuid4()),
        filename=resolved.name,
        path=str(resolved),
        kind=AssetKind.IMAGE_SEQUENCE.value,
        size_bytes=size_bytes,
        checksum=checksum,
        checksum_status=checksum_status,
        collections=normalize_collections(collections),
        media_count=len(files),
        metadata={key: value for key, value in asset_metadata.items() if value is not None},
        warnings=warnings,
    )


def analyze_ingest_path(
    input_path: str | Path,
    *,
    kind: str = "auto",
    recursive: bool = False,
    collections: Any = None,
    compute_checksum: bool = False,
    metadata: dict[str, Any] | None = None,
) -> list[AnalyzedAsset]:
    root = Path(input_path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    normalized_kind = str(kind or "auto").lower()
    if normalized_kind not in {"auto", "video", "image_sequence"}:
        raise ValueError("kind must be one of: auto, video, image_sequence.")

    if normalized_kind == "video":
        if not root.is_file() or not is_supported_video_file(root):
            raise ValueError(f"Path is not a supported video file: {root}")
        return [analyze_video_asset(root, collections=collections, compute_checksum=compute_checksum, metadata=metadata)]

    if normalized_kind == "image_sequence":
        folder = root.parent if root.is_file() and is_supported_image_file(root) else root
        if not folder.is_dir():
            raise ValueError(f"Path is not an image sequence folder: {root}")
        return [
            analyze_image_sequence_asset(
                folder,
                recursive=recursive,
                collections=collections,
                compute_checksum=compute_checksum,
                metadata=metadata,
            )
        ]

    sources = discover_ingest_sources(root, recursive=recursive)
    assets: list[AnalyzedAsset] = []
    for source in sources:
        if source.kind == AssetKind.VIDEO.value:
            assets.append(
                analyze_video_asset(
                    source.path,
                    collections=collections,
                    compute_checksum=compute_checksum,
                    metadata=metadata,
                )
            )
        elif source.kind == AssetKind.IMAGE_SEQUENCE.value:
            assets.append(
                analyze_image_sequence_asset(
                    source.path,
                    recursive=source.recursive,
                    collections=collections,
                    compute_checksum=compute_checksum,
                    metadata=metadata,
                )
            )
    return assets
