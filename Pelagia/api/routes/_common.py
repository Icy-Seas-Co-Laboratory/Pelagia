from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any

from fastapi import HTTPException, Request

from ...processing.thumbhash import thumbhash_to_base64
from ...services.context import AppContext
from ...utils.serialization import json_ready


def get_context(request: Request) -> AppContext:
    context = getattr(request.app.state, "context", None)
    if context is None:
        raise HTTPException(status_code=503, detail="Pelagia application context is not configured.")
    return context


def get_repository(request: Request):
    repository = get_context(request).repository
    if repository is None:
        raise HTTPException(status_code=503, detail="Postgres repository is not configured.")
    return repository


def get_kvstore(request: Request):
    kvstore = get_context(request).kvstore
    if kvstore is None:
        raise HTTPException(status_code=503, detail="KVStore is not configured.")
    return kvstore


def as_response(value: Any) -> Any:
    return json_ready(value)


def _as_dict(row: Any) -> dict[str, Any]:
    if row is None:
        return {}
    if is_dataclass(row):
        return asdict(row)
    return dict(row)


def without_payload_bytes(row: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    item = _as_dict(row)
    for field in fields:
        payload = item.pop(field, None)
        if payload is not None:
            item[f"{field}_bytes"] = len(payload)
    return as_response(item)


def page_metadata(*, limit: int | None, offset: int = 0, count: int = 0) -> dict[str, int | None]:
    resolved_offset = max(0, int(offset or 0))
    resolved_count = max(0, int(count or 0))
    if limit is None:
        resolved_limit = None
        next_offset = None
    else:
        resolved_limit = max(0, int(limit))
        next_offset = (
            resolved_offset + resolved_count
            if resolved_limit > 0 and resolved_count >= resolved_limit
            else None
        )
    return {
        "limit": resolved_limit,
        "offset": resolved_offset,
        "count": resolved_count,
        "next_offset": next_offset,
    }


def mark_frame_stage_status(
    repository,
    *,
    project_id: str | None,
    frame_ids: list[str],
    stage: str,
    status: str,
    job_id: str | None = None,
) -> None:
    if not project_id or not frame_ids:
        return
    resolved_frame_ids = list(dict.fromkeys(str(frame_id) for frame_id in frame_ids if frame_id))
    if not resolved_frame_ids:
        return
    updater = getattr(repository, "upsert_frame_stage_status", None)
    if callable(updater):
        updater(
            project_id=project_id,
            frame_ids=resolved_frame_ids,
            stage=stage,
            status=status,
            job_id=job_id,
        )


def refresh_frame_status_counts(
    repository,
    *,
    project_id: str | None,
    frame_ids: list[str],
    asset_id: str | None = None,
) -> None:
    if not project_id or not frame_ids:
        return
    resolved_frame_ids = list(dict.fromkeys(str(frame_id) for frame_id in frame_ids if frame_id))
    if not resolved_frame_ids:
        return
    refresh_counts = getattr(repository, "refresh_frame_status_counts", None)
    if callable(refresh_counts):
        refresh_counts(
            project_id=project_id,
            frame_ids=resolved_frame_ids,
            asset_id=asset_id,
        )


def touch_processing_status_snapshot(repository, *, project_id: str | None) -> None:
    if not project_id:
        return
    touch_snapshot = getattr(repository, "touch_processing_status_snapshot", None)
    if callable(touch_snapshot):
        touch_snapshot(project_id=project_id)


def frame_summary(row: Any) -> dict[str, Any]:
    item = _as_dict(row)
    flatfield_profile = item.pop("flatfield_profile", None)
    item["has_flatfield_profile"] = bool(flatfield_profile)
    item["flatfield_profile_length"] = len(flatfield_profile or [])
    preview_thumbhash = item.pop("preview_thumbhash", item.pop("frame_png", None))
    if preview_thumbhash is not None:
        item["preview_thumbhash_bytes"] = len(preview_thumbhash)
        item["preview_thumbhash_base64"] = thumbhash_to_base64(preview_thumbhash)
    preprocessed_preview_thumbhash = item.pop("preprocessed_preview_thumbhash", None)
    if preprocessed_preview_thumbhash is not None:
        item["preprocessed_preview_thumbhash_bytes"] = len(preprocessed_preview_thumbhash)
        item["preprocessed_preview_thumbhash_base64"] = thumbhash_to_base64(preprocessed_preview_thumbhash)
    item["has_preprocessed_payload"] = bool(
        item.get("preprocessed_payload_ref") or item.get("preprocessed_kvstore_hash")
    )
    item["has_background_payload"] = bool(
        item.get("background_payload_ref") or item.get("background_kvstore_hash")
    )
    return as_response(item)


def detection_summary(row: Any, *, include_payload: bool = False) -> dict[str, Any]:
    item = as_response(_as_dict(row)) if include_payload else without_payload_bytes(row, ("roi_payload", "mask_payload"))
    metadata = item.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    item["metadata"] = metadata
    bbox_values = [item.get("bbox_x"), item.get("bbox_y"), item.get("bbox_w"), item.get("bbox_h")]
    if all(value is not None for value in bbox_values):
        item["bbox"] = {
            "x": item.get("bbox_x"),
            "y": item.get("bbox_y"),
            "w": item.get("bbox_w"),
            "h": item.get("bbox_h"),
        }
    crop_bbox_values = [
        item.get("crop_bbox_x"),
        item.get("crop_bbox_y"),
        item.get("crop_bbox_w"),
        item.get("crop_bbox_h"),
    ]
    if all(value is not None for value in crop_bbox_values):
        item["crop_bbox"] = {
            "x": item.get("crop_bbox_x"),
            "y": item.get("crop_bbox_y"),
            "w": item.get("crop_bbox_w"),
            "h": item.get("crop_bbox_h"),
        }
    _add_refined_detection_contract(item)
    return as_response(item)


def _unique_strings(values: list[Any]) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if not text or text in seen:
            continue
        seen.add(text)
        result.append(text)
    return result


def _refinement_relationship(metadata: dict[str, Any]) -> str:
    has_split_parent = bool(metadata.get("split_from_candidate_detection_id"))
    consumed_ids = metadata.get("consumed_candidate_detection_ids") or []
    has_consumed = bool(consumed_ids)
    if has_split_parent and has_consumed:
        return "many_to_many"
    if has_split_parent:
        return "split_child"
    if has_consumed:
        return "merge_keeper"
    return "one_to_one"


def _add_refined_detection_contract(item: dict[str, Any]) -> None:
    metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
    refined_detection_id = item.get("refined_detection_id")
    if refined_detection_id is not None:
        item.setdefault("refined_roi_url", f"/refined-detections/{refined_detection_id}/roi")
        item.setdefault("refined_mask_url", f"/refined-detections/{refined_detection_id}/mask")

    candidate_detection_id = item.get("candidate_detection_id")
    if candidate_detection_id is None:
        return

    primary_candidate_detection_id = (
        metadata.get("split_from_candidate_detection_id")
        or metadata.get("primary_candidate_detection_id")
        or candidate_detection_id
    )
    consumed_ids = metadata.get("consumed_candidate_detection_ids") or []
    if not isinstance(consumed_ids, list):
        consumed_ids = [consumed_ids]
    candidate_detection_ids = _unique_strings([primary_candidate_detection_id, candidate_detection_id, *consumed_ids])

    item["primary_candidate_detection_id"] = str(primary_candidate_detection_id)
    item["candidate_detection_ids"] = candidate_detection_ids
    item["refinement_relationship"] = _refinement_relationship(metadata)
    if item.get("id") is not None:
        item.setdefault("refined_roi_url", f"/refined-detections/{item['id']}/roi")
        item.setdefault("refined_mask_url", f"/refined-detections/{item['id']}/mask")


def postgres_ping(repository) -> dict[str, Any]:
    with repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1 AS ok")
            row = cursor.fetchone()
    return {
        "healthy": bool(row and row.get("ok") == 1),
        "schema": repository.schema,
    }


def kvstore_status(kvstore, *, deep: bool = False) -> dict[str, Any]:
    try:
        status = kvstore.status(deep=deep)
    except TypeError:
        status = kvstore.status()
    status = _normalize_kvstore_status(status)
    return as_response(status)


def _normalize_kvstore_status(status: dict[str, Any]) -> dict[str, Any]:
    """Add stable byte-count aliases across legacy KVStore and KVStore2."""
    normalized = dict(status)
    total_file_bytes = _first_int(
        normalized,
        "total_file_bytes",
        "total_physical_file_bytes",
        "total_storage_file_bytes",
    )
    if total_file_bytes is None:
        index_bytes = _first_int(normalized, "total_index_file_bytes")
        blob_bytes = _first_int(normalized, "total_blob_file_bytes")
        sqlite_bytes = _first_int(normalized, "total_sqlite_file_bytes")
        if blob_bytes is not None:
            total_file_bytes = blob_bytes + (index_bytes or 0)
        elif sqlite_bytes is not None:
            total_file_bytes = sqlite_bytes

    if total_file_bytes is not None:
        normalized.setdefault("total_file_bytes", total_file_bytes)
        normalized.setdefault("total_physical_file_bytes", total_file_bytes)
        # Compatibility for clients that historically read the legacy KVStore field
        # as the single on-disk size indicator.
        normalized.setdefault("total_sqlite_file_bytes", total_file_bytes)

    if "total_storage_file_bytes" not in normalized and total_file_bytes is not None:
        normalized["total_storage_file_bytes"] = total_file_bytes
    return normalized


def _first_int(mapping: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = mapping.get(key)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None
