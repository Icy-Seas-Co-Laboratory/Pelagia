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


def frame_summary(row: Any) -> dict[str, Any]:
    item = _as_dict(row)
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


def detection_summary(row: Any) -> dict[str, Any]:
    item = without_payload_bytes(row, ("roi_payload", "mask_payload"))
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
    return as_response(item)


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
    return as_response(status)
