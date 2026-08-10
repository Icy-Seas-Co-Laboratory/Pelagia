from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Any

from ..processing.codec_registry import STORAGE_ENCODINGS, normalize_image_encoding


ROI_STORAGE_ENCODINGS = STORAGE_ENCODINGS
PROJECT_SETTINGS_CACHE_SECONDS = 5.0


def normalize_frame_encoding(value: object) -> str:
    try:
        return normalize_image_encoding(value)
    except ValueError:
        raise ValueError("frame encoding must be one of: png, jpg, jxl, jxs, raw, zstd.")


def normalize_roi_encoding(value: object) -> str:
    try:
        return normalize_image_encoding(value)
    except ValueError:
        raise ValueError("ROI encoding must be one of: png, jpg, jxl, jxs, raw, zstd.")


def normalize_mask_encoding(value: object) -> str:
    encoding = normalize_roi_encoding(value)
    if encoding not in {"zstd", "png", "raw"}:
        raise ValueError("mask encoding must be lossless: zstd, png, or raw.")
    return encoding


def normalize_frame_quality(value: object) -> int:
    quality = int(value)
    if quality < 0 or quality > 100:
        raise ValueError("frame quality must be between 0 and 100.")
    return quality


def normalize_roi_cutoff(value: object) -> int:
    cutoff = int(value)
    if cutoff < 1:
        raise ValueError("large ROI cutoff must be a positive pixel count.")
    return cutoff


def validate_allowed_storage_encodings(context, *encodings: str) -> None:
    """Reject project or request codecs disabled by deployment policy."""
    allowed = set(context.config.image_data_storage.allowed_encodings)
    disallowed = sorted({encoding for encoding in encodings if encoding not in allowed})
    if disallowed:
        raise ValueError(
            "Storage codec(s) disabled by image_data_storage.allowed_encodings: "
            + ", ".join(disallowed)
        )


@dataclass(frozen=True, slots=True)
class ProjectStorageSettings:
    frame_encoding: str
    frame_quality: int
    small_roi_encoding: str
    large_roi_encoding: str
    large_roi_min_pixels: int
    roi_quality: int
    mask_encoding: str
    frame_encoding_source: str
    frame_quality_source: str
    small_roi_encoding_source: str
    large_roi_encoding_source: str
    large_roi_min_pixels_source: str
    roi_quality_source: str
    mask_encoding_source: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame": {"encoding": self.frame_encoding, "quality": self.frame_quality},
            "roi": {
                "small_encoding": self.small_roi_encoding,
                "large_encoding": self.large_roi_encoding,
                "large_min_pixels": self.large_roi_min_pixels,
                "quality": self.roi_quality,
                "mask_encoding": self.mask_encoding,
            },
            "sources": {
                "frame_encoding": self.frame_encoding_source,
                "frame_quality": self.frame_quality_source,
                "small_roi_encoding": self.small_roi_encoding_source,
                "large_roi_encoding": self.large_roi_encoding_source,
                "large_roi_min_pixels": self.large_roi_min_pixels_source,
                "roi_quality": self.roi_quality_source,
                "mask_encoding": self.mask_encoding_source,
            },
        }

    def roi_policy_payload(self) -> dict[str, Any]:
        return {
            "small_roi_encoding": self.small_roi_encoding,
            "large_roi_encoding": self.large_roi_encoding,
            "large_roi_min_pixels": self.large_roi_min_pixels,
            "roi_quality": self.roi_quality,
            "mask_encoding": self.mask_encoding,
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
    small_roi_encoding: object | None = None,
    large_roi_encoding: object | None = None,
    large_roi_min_pixels: object | None = None,
    roi_quality: object | None = None,
    mask_encoding: object | None = None,
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
    if any(value is not None for value in (small_roi_encoding, large_roi_encoding, large_roi_min_pixels, roi_quality, mask_encoding)):
        roi: dict[str, Any] = {}
        if small_roi_encoding is not None:
            roi["small_encoding"] = normalize_roi_encoding(small_roi_encoding)
        if large_roi_encoding is not None:
            roi["large_encoding"] = normalize_roi_encoding(large_roi_encoding)
        if large_roi_min_pixels is not None:
            roi["large_min_pixels"] = normalize_roi_cutoff(large_roi_min_pixels)
        if roi_quality is not None:
            roi["quality"] = normalize_frame_quality(roi_quality)
        if mask_encoding is not None:
            roi["mask_encoding"] = normalize_mask_encoding(mask_encoding)
        storage["roi"] = roi
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
    small_roi_encoding: object | None = None,
    large_roi_encoding: object | None = None,
    large_roi_min_pixels: object | None = None,
    roi_quality: object | None = None,
    mask_encoding: object | None = None,
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

    def resolve_roi_value(override: object | None, key: str, fallback: object, normalizer):
        if override is not None:
            return normalizer(override), "override"
        if roi.get(key) is not None:
            return normalizer(roi[key]), project_source
        return normalizer(fallback), "global"

    resolved_small_encoding, small_source = resolve_roi_value(
        small_roi_encoding, "small_encoding", global_roi.small_roi_encoding, normalize_roi_encoding
    )
    resolved_large_encoding, large_source = resolve_roi_value(
        large_roi_encoding, "large_encoding", global_roi.large_roi_encoding, normalize_roi_encoding
    )
    resolved_large_min_pixels, cutoff_source = resolve_roi_value(
        large_roi_min_pixels, "large_min_pixels", global_roi.large_roi_min_pixels, normalize_roi_cutoff
    )
    resolved_roi_quality, roi_quality_source = resolve_roi_value(
        roi_quality, "quality", global_roi.roi_quality, normalize_frame_quality
    )
    resolved_mask_encoding, mask_source = resolve_roi_value(
        mask_encoding, "mask_encoding", global_roi.mask_encoding, normalize_mask_encoding
    )
    validate_allowed_storage_encodings(
        context,
        resolved_frame_encoding,
        resolved_small_encoding,
        resolved_large_encoding,
        resolved_mask_encoding,
    )

    storage = ProjectStorageSettings(
        frame_encoding=resolved_frame_encoding,
        frame_quality=resolved_frame_quality,
        small_roi_encoding=resolved_small_encoding,
        large_roi_encoding=resolved_large_encoding,
        large_roi_min_pixels=resolved_large_min_pixels,
        roi_quality=resolved_roi_quality,
        mask_encoding=resolved_mask_encoding,
        frame_encoding_source=encoding_source,
        frame_quality_source=quality_source,
        small_roi_encoding_source=small_source,
        large_roi_encoding_source=large_source,
        large_roi_min_pixels_source=cutoff_source,
        roi_quality_source=roi_quality_source,
        mask_encoding_source=mask_source,
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
    small_roi_encoding: object | None = None,
    large_roi_encoding: object | None = None,
    large_roi_min_pixels: object | None = None,
    roi_quality: object | None = None,
    mask_encoding: object | None = None,
) -> ProjectStorageSettings:
    """Compatibility helper for callers that only need storage defaults."""
    return resolve_project_settings(
        context,
        project_id,
        frame_encoding=frame_encoding,
        frame_quality=frame_quality,
        small_roi_encoding=small_roi_encoding,
        large_roi_encoding=large_roi_encoding,
        large_roi_min_pixels=large_roi_min_pixels,
        roi_quality=roi_quality,
        mask_encoding=mask_encoding,
    ).storage
