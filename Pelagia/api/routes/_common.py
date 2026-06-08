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
    return as_response(item)


def detection_summary(row: Any) -> dict[str, Any]:
    return without_payload_bytes(row, ("roi_payload", "mask_payload"))


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
