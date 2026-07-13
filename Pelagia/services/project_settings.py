from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

from ..processing.codec_registry import STORAGE_ENCODINGS, normalize_image_encoding


ROI_STORAGE_ENCODINGS = frozenset({*STORAGE_ENCODINGS, "auto"})
PROJECT_SETTINGS_CACHE_SECONDS = 5.0


def normalize_frame_encoding(value: object) -> str:
    try:
        return normalize_image_encoding(value)
    except ValueError:
        raise ValueError("frame encoding must be one of: png, jpg, jxl, jxs, raw, zstd.")


def normalize_roi_encoding(value: object) -> str:
    if str(value).strip().lower() == "auto":
        return "auto"
    try:
        return normalize_image_encoding(value)
    except ValueError:
        raise ValueError("ROI encoding must be one of: png, jpg, jxl, jxs, raw, zstd, auto.")


def normalize_frame_quality(value: object) -> int:
    quality = int(value)
    if quality < 0 or quality > 100:
        raise ValueError("frame quality must be between 0 and 100.")
    return quality


@dataclass(frozen=True, slots=True)
class ProjectStorageSettings:
    frame_encoding: str
    frame_quality: int
    roi_encoding: str
    frame_encoding_source: str
    frame_quality_source: str
    roi_encoding_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": {"encoding": self.frame_encoding, "quality": self.frame_quality},
            "roi": {"encoding": self.roi_encoding},
            "sources": {
                "frame_encoding": self.frame_encoding_source,
                "frame_quality": self.frame_quality_source,
                "roi_encoding": self.roi_encoding_source,
            },
        }


@dataclass(frozen=True, slots=True)
class EffectiveProjectSettings:
    """Resolved project settings plus the raw persisted setting document."""

    project_id: str | None
    configured: dict[str, Any]
    storage: ProjectStorageSettings

    def as_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id,
            "configured": self.configured,
            "storage": self.storage.as_dict(),
        }


def storage_settings_payload(
    *,
    frame_encoding: object | None = None,
    frame_quality: object | None = None,
    roi_encoding: object | None = None,
) -> dict[str, Any]:
    """Return a canonical, validated partial project settings payload."""
    storage: dict[str, Any] = {}
    if frame_encoding is not None or frame_quality is not None:
        frame: dict[str, Any] = {}
        if frame_encoding is not None:
            frame["encoding"] = normalize_frame_encoding(frame_encoding)
        if frame_quality is not None:
            frame["quality"] = normalize_frame_quality(frame_quality)
        storage["frame"] = frame
    if roi_encoding is not None:
        storage["roi"] = {"encoding": normalize_roi_encoding(roi_encoding)}
    return {"storage": storage} if storage else {}


def merge_project_settings(existing: object, patch: object) -> dict[str, Any]:
    """Merge a validated storage patch without treating nested JSON as shallow."""
    result = dict(existing) if isinstance(existing, dict) else {}
    patch_value = dict(patch) if isinstance(patch, dict) else {}
    existing_storage = result.get("storage")
    storage = dict(existing_storage) if isinstance(existing_storage, dict) else {}
    patch_storage = patch_value.get("storage")
    if isinstance(patch_storage, dict):
        for section in ("frame", "roi"):
            values = patch_storage.get(section)
            if isinstance(values, dict):
                current = storage.get(section)
                merged = dict(current) if isinstance(current, dict) else {}
                merged.update(values)
                storage[section] = merged
        result["storage"] = storage
    return result


def _legacy_storage(project: dict[str, Any] | None) -> dict[str, Any]:
    metadata = project.get("metadata") if isinstance(project, dict) else None
    if not isinstance(metadata, dict):
        return {}
    processing = metadata.get("processing")
    legacy_frame = None
    if isinstance(processing, dict) and isinstance(processing.get("frame_storage"), dict):
        legacy_frame = processing["frame_storage"]
    frame_storage = metadata.get("frame_storage")
    if legacy_frame is None and isinstance(frame_storage, dict):
        legacy_frame = frame_storage
    if isinstance(legacy_frame, dict):
        frame: dict[str, Any] = {}
        if legacy_frame.get("image_encoding") is not None:
            frame["encoding"] = legacy_frame["image_encoding"]
        if legacy_frame.get("image_quality") is not None:
            frame["quality"] = legacy_frame["image_quality"]
        return {"frame": frame} if frame else {}
    encoding = metadata.get("frame_storage_image_encoding")
    return {"frame": {"encoding": encoding}} if encoding is not None else {}


def _configured_storage(project: dict[str, Any] | None) -> tuple[dict[str, Any], str]:
    settings = project.get("settings") if isinstance(project, dict) else None
    if isinstance(settings, dict) and isinstance(settings.get("storage"), dict):
        legacy = _legacy_storage(project)
        storage = dict(legacy)
        for section, values in settings["storage"].items():
            if not isinstance(values, dict):
                continue
            current = storage.get(section)
            merged = dict(current) if isinstance(current, dict) else {}
            merged.update(values)
            storage[section] = merged
        return storage, "project"
    legacy = _legacy_storage(project)
    return legacy, "legacy-project" if legacy else "global"


def project_settings_record(context, project_id: str | None, *, refresh: bool = False) -> dict[str, Any] | None:
    """Load and cache the project record for request and worker lifetime reuse."""
    if not project_id:
        return None
    cache = getattr(context, "_project_settings_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(context, "_project_settings_cache", cache)
        except (AttributeError, TypeError):
            cache = None
    cache_key = str(project_id)
    cached = cache.get(cache_key) if isinstance(cache, dict) else None
    if (
        not refresh
        and isinstance(cached, tuple)
        and len(cached) == 2
        and isinstance(cached[0], dict)
        and float(cached[1]) > monotonic()
    ):
        return cached[0]
    repository = getattr(context, "repository", None)
    project = repository.get_project(cache_key) if repository is not None and hasattr(repository, "get_project") else None
    if cache is not None and project is not None:
        cache[cache_key] = (project, monotonic() + PROJECT_SETTINGS_CACHE_SECONDS)
    return project


def invalidate_project_settings(context, project_id: str | None = None) -> None:
    cache = getattr(context, "_project_settings_cache", None)
    if not isinstance(cache, dict):
        return
    if project_id is None:
        cache.clear()
    else:
        cache.pop(str(project_id), None)


def resolve_project_settings(
    context,
    project_id: str | None,
    *,
    frame_encoding: object | None = None,
    frame_quality: object | None = None,
    roi_encoding: object | None = None,
) -> EffectiveProjectSettings:
    """Resolve persisted project settings with per-context caching and provenance."""
    project = project_settings_record(context, project_id)
    storage, project_source = _configured_storage(project)
    frame = storage.get("frame") if isinstance(storage.get("frame"), dict) else {}
    roi = storage.get("roi") if isinstance(storage.get("roi"), dict) else {}
    global_frame = context.config.processing.frame_storage
    global_roi = context.config.processing.roi_recording

    if frame_encoding is not None:
        resolved_frame_encoding, encoding_source = normalize_frame_encoding(frame_encoding), "override"
    elif frame.get("encoding") is not None:
        resolved_frame_encoding, encoding_source = normalize_frame_encoding(frame["encoding"]), project_source
    else:
        resolved_frame_encoding, encoding_source = global_frame.image_encoding, "global"

    if frame_quality is not None:
        resolved_frame_quality, quality_source = normalize_frame_quality(frame_quality), "override"
    elif frame.get("quality") is not None:
        resolved_frame_quality, quality_source = normalize_frame_quality(frame["quality"]), project_source
    else:
        resolved_frame_quality, quality_source = int(global_frame.image_quality), "global"

    if roi_encoding is not None:
        resolved_roi_encoding, roi_source = normalize_roi_encoding(roi_encoding), "override"
    elif roi.get("encoding") is not None:
        resolved_roi_encoding, roi_source = normalize_roi_encoding(roi["encoding"]), project_source
    else:
        resolved_roi_encoding, roi_source = str(global_roi.roi_encoding), "global"

    storage = ProjectStorageSettings(
        frame_encoding=resolved_frame_encoding,
        frame_quality=resolved_frame_quality,
        roi_encoding=resolved_roi_encoding,
        frame_encoding_source=encoding_source,
        frame_quality_source=quality_source,
        roi_encoding_source=roi_source,
    )
    settings = project.get("settings") if isinstance(project, dict) else None
    return EffectiveProjectSettings(
        project_id=None if project_id is None else str(project_id),
        configured=dict(settings) if isinstance(settings, dict) else {},
        storage=storage,
    )


def resolve_project_storage_settings(
    context,
    project_id: str | None,
    *,
    frame_encoding: object | None = None,
    frame_quality: object | None = None,
    roi_encoding: object | None = None,
) -> ProjectStorageSettings:
    """Compatibility helper for callers that only need storage defaults."""
    return resolve_project_settings(
        context,
        project_id,
        frame_encoding=frame_encoding,
        frame_quality=frame_quality,
        roi_encoding=roi_encoding,
    ).storage
