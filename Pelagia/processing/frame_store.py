import json
import time
from datetime import datetime
from typing import Any

import numpy as np

from ..domain import FrameRecord
from ..services.context import AppContext
from ..storage.blob_store import initialize_kvstore
from ..utils.serialization import json_ready
from ._logging import log_processing_event, processing_core_logger
from .frame_codec import decode_array_payload, encode_array_payload
from .frame_model import FrameData
from .thumbhash import compute_thumbhash


_DEFAULT_CONTEXT: AppContext | None = None
_CORE_LOGGER = processing_core_logger("frame_store")


def default_context() -> AppContext:
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = AppContext.from_config()
        if _DEFAULT_CONTEXT.kvstore is not None and not _DEFAULT_CONTEXT.kvstore.initialized:
            initialize_kvstore(_DEFAULT_CONTEXT.kvstore, _DEFAULT_CONTEXT.config.kvstore)
    return _DEFAULT_CONTEXT


def metadata_without_none(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}


def _active_project_id(ctx: AppContext) -> str | None:
    return getattr(ctx, "active_project_id", None)


def _kvstore_for_project(ctx: AppContext, project_id: str | None):
    if hasattr(ctx, "kvstore_for_project"):
        return ctx.kvstore_for_project(project_id)
    return ctx.kvstore


def _asset_project_id(ctx: AppContext, asset_id: str | None, *, fallback: str | None = None) -> str | None:
    if asset_id is None or ctx.repository is None or not hasattr(ctx.repository, "get_asset"):
        return fallback
    asset = ctx.repository.get_asset(str(asset_id), project_id=fallback) if fallback else ctx.repository.get_asset(str(asset_id))
    if asset is None:
        if fallback:
            raise KeyError(f"Asset {asset_id!r} was not found in project {fallback!r}.")
        return fallback
    return None if asset.get("project_id") is None else str(asset["project_id"])


def _frame_project_id(ctx: AppContext, frame_id: str, *, fallback: str | None = None) -> str | None:
    if ctx.repository is None or not hasattr(ctx.repository, "get_frame"):
        return fallback
    row = ctx.repository.get_frame(str(frame_id), project_id=fallback) if fallback else ctx.repository.get_frame(str(frame_id))
    if row is None:
        if fallback:
            raise KeyError(f"Frame {frame_id!r} was not found in project {fallback!r}.")
        return fallback
    return _asset_project_id(ctx, row.get("asset_id"), fallback=fallback)


def store_frame(frame: FrameData, context: AppContext | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    data = frame.read()
    if data is None:
        raise ValueError("Frame has no numpy data to store.")

    metadata = dict(frame.metadata or {})
    ctx = context or default_context()
    project_id = _active_project_id(ctx)
    run_id = getattr(frame, "run_id", None) or metadata.get("run_id")
    asset_id = getattr(frame, "asset_id", None) or metadata.get("asset_id")
    frame_index = metadata.get("frame_index")
    if frame_index is None:
        frame_index = frame.tileNumber if frame.tileNumber is not None else frame.frameNumber
    try:
        array = np.ascontiguousarray(data)
        if array.ndim < 2:
            raise ValueError("Frame data must have at least two dimensions.")
        frame.validate_geometry(array)

        project_id = _asset_project_id(ctx, str(asset_id) if asset_id else None, fallback=project_id)
        kvstore = _kvstore_for_project(ctx, project_id)
        if kvstore is None:
            raise RuntimeError("A KVStore is required to store frame data.")
        if ctx.repository is None:
            raise RuntimeError("A PostgresRepository is required to record frame metadata.")

        default_encoding = ctx.config.processing.frame_storage.image_encoding
        requested_encoding = metadata.get(
            "kvstore_encoding",
            metadata.get("array_encoding", metadata.get("kvstore_format", default_encoding)),
        )
        default_quality = ctx.config.processing.frame_storage.image_quality
        requested_quality = metadata.get(
            "kvstore_quality",
            metadata.get("array_quality", metadata.get("image_quality", default_quality)),
        )
        payload, kvstore_encoding, kvstore_format = encode_array_payload(
            array,
            requested_encoding,
            quality=int(requested_quality),
        )
        kvstore_key = kvstore.put_store(payload)
        preview_thumbhash = compute_thumbhash(array, max_dim=ctx.config.processing.thumbhash.max_dim)
        width, height = frame.get_size()
        source_frame_start, source_frame_end = frame.get_source_frame_range()
        captured_at = frame.timestamp if isinstance(frame.timestamp, datetime) else None

        if not run_id or not asset_id:
            raise ValueError("Frame metadata must include run_id and asset_id.")
        metadata.update(
            metadata_without_none(
                {
                    "kvstore_key": kvstore_key,
                    "kvstore_hash": kvstore_key,
                    "kvstore_encoding": kvstore_encoding,
                    "kvstore_format": kvstore_format,
                    "kvstore_quality": int(requested_quality),
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "width": width,
                    "height": height,
                    "bbox_x": frame.bbox_x,
                    "bbox_y": frame.bbox_y,
                    "parent_frame_id": frame.parent_frame_id,
                    "source_path": frame.sourcePath,
                    "filename": frame.filename,
                    "frame_number": frame.frameNumber,
                    "tile_number": frame.tileNumber,
                    "source_frame_start": source_frame_start,
                    "source_frame_end": source_frame_end,
                    "frame_type": frame.frameType,
                    "channel": frame.channel,
                    "timestamp": None if captured_at is not None else frame.timestamp,
                }
            )
        )
        frame_record = FrameRecord(
            run_id=str(run_id),
            asset_id=str(asset_id),
            frame_index=int(frame_index),
            captured_at=captured_at,
            width=width,
            height=height,
            bbox_x=frame.bbox_x,
            bbox_y=frame.bbox_y,
            parent_frame_id=frame.parent_frame_id,
            source_ref=frame.get_source_file_path(),
            kvstore_hash=kvstore_key,
            preview_thumbhash=preview_thumbhash,
            payload_ref=kvstore_key,
            payload_encoding=kvstore_encoding,
            payload_format=kvstore_format,
            payload_dtype=str(array.dtype),
            payload_shape=list(array.shape),
            metadata=metadata,
        )

        with ctx.repository.connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f"""
                    INSERT INTO {ctx.repository.schema}.frames
                    (run_id, asset_id, frame_index, captured_at, width, height,
                     bbox_x, bbox_y, parent_frame_id, source_ref, kvstore_hash, preview_thumbhash,
                     payload_ref, payload_encoding, payload_format, payload_dtype, payload_shape, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    ON CONFLICT (asset_id, frame_index) DO UPDATE SET
                        captured_at = EXCLUDED.captured_at,
                        width = EXCLUDED.width,
                        height = EXCLUDED.height,
                        bbox_x = EXCLUDED.bbox_x,
                        bbox_y = EXCLUDED.bbox_y,
                        parent_frame_id = EXCLUDED.parent_frame_id,
                        source_ref = EXCLUDED.source_ref,
                        kvstore_hash = EXCLUDED.kvstore_hash,
                        preview_thumbhash = EXCLUDED.preview_thumbhash,
                        payload_ref = EXCLUDED.payload_ref,
                        payload_encoding = EXCLUDED.payload_encoding,
                        payload_format = EXCLUDED.payload_format,
                        payload_dtype = EXCLUDED.payload_dtype,
                        payload_shape = EXCLUDED.payload_shape,
                        metadata = EXCLUDED.metadata
                    RETURNING *;
                    """,
                    (
                        frame_record.run_id,
                        frame_record.asset_id,
                        frame_record.frame_index,
                        frame_record.captured_at,
                        frame_record.width,
                        frame_record.height,
                        frame_record.bbox_x,
                        frame_record.bbox_y,
                        frame_record.parent_frame_id,
                        frame_record.source_ref,
                        frame_record.kvstore_hash,
                        frame_record.preview_thumbhash,
                        frame_record.payload_ref,
                        frame_record.payload_encoding,
                        frame_record.payload_format,
                        frame_record.payload_dtype,
                        json.dumps(json_ready(frame_record.payload_shape)),
                        json.dumps(json_ready(frame_record.metadata)),
                    ),
                )
                row = cursor.fetchone()
            connection.commit()
        _CORE_LOGGER.debug(
            "Stored frame run_id=%s asset_id=%s frame_index=%s shape=%s encoding=%s duration_ms=%.2f",
            run_id,
            asset_id,
            frame_index,
            list(array.shape),
            kvstore_encoding,
            (time.perf_counter() - started) * 1000,
        )
        return row
    except Exception as exc:
        duration_ms = (time.perf_counter() - started) * 1000
        _CORE_LOGGER.exception(
            "Frame storage failed run_id=%s asset_id=%s frame_index=%s",
            run_id,
            asset_id,
            frame_index,
        )
        log_processing_event(
            ctx,
            "error",
            "frame_store.store_failed",
            "Frame storage failed",
            run_id=None if run_id is None else str(run_id),
            asset_id=None if asset_id is None else str(asset_id),
            duration_ms=duration_ms,
            payload={
                "frame_index": frame_index,
                "frame_number": frame.frameNumber,
                "tile_number": frame.tileNumber,
                "error_type": type(exc).__name__,
                "error_message": str(exc),
            },
            logger="pelagia.processing.frame_store",
            core_logger=_CORE_LOGGER,
        )
        raise


def store_preprocessed_frame(
    frame_id: str,
    frame: FrameData,
    *,
    context: AppContext | None = None,
    encoding: str | None = None,
    quality: int | None = None,
) -> dict[str, Any]:
    data = frame.read()
    if data is None:
        raise ValueError("Preprocessed frame has no numpy data to store.")

    ctx = context or default_context()
    project_id = _frame_project_id(ctx, frame_id, fallback=_active_project_id(ctx))
    kvstore = _kvstore_for_project(ctx, project_id)
    if kvstore is None:
        raise RuntimeError("A KVStore is required to store preprocessed frame data.")
    if ctx.repository is None:
        raise RuntimeError("A PostgresRepository is required to record preprocessed frame metadata.")

    array = np.ascontiguousarray(data)
    requested_encoding = encoding or ctx.config.processing.frame_storage.image_encoding
    requested_quality = int(
        quality
        if quality is not None
        else dict(frame.metadata or {}).get("kvstore_quality", ctx.config.processing.frame_storage.image_quality)
    )
    payload, kvstore_encoding, kvstore_format = encode_array_payload(
        array,
        requested_encoding,
        quality=requested_quality,
    )
    kvstore_key = kvstore.put_store(payload)
    preview_thumbhash = compute_thumbhash(array, max_dim=ctx.config.processing.thumbhash.max_dim)
    metadata = metadata_without_none(
        {
            **dict(frame.metadata or {}),
            "kvstore_key": kvstore_key,
            "kvstore_hash": kvstore_key,
            "kvstore_encoding": kvstore_encoding,
            "kvstore_format": kvstore_format,
            "kvstore_quality": requested_quality,
            "dtype": str(array.dtype),
            "shape": list(array.shape),
            "frame_variant": "preprocessed",
        }
    )
    return ctx.repository.update_frame_preprocessed_payload(
        frame_id,
        project_id=project_id,
        kvstore_hash=kvstore_key,
        preview_thumbhash=preview_thumbhash,
        payload_ref=kvstore_key,
        payload_encoding=kvstore_encoding,
        payload_format=kvstore_format,
        payload_dtype=str(array.dtype),
        payload_shape=list(array.shape),
        metadata=metadata,
    )


def retrieve_frame(
    id: str,
    context: AppContext | None = None,
    *,
    payload_kind: str = "original",
) -> FrameData:
    started = time.perf_counter()
    ctx = context or default_context()
    project_id = _active_project_id(ctx)
    if ctx.kvstore is None:
        raise RuntimeError("A KVStore is required to retrieve frame data.")
    if ctx.repository is None:
        raise RuntimeError("A PostgresRepository is required to load frame metadata.")

    with ctx.repository.connect() as connection:
        with connection.cursor() as cursor:
            if project_id:
                cursor.execute(
                    f"""
                    SELECT frames.*, assets.project_id AS project_id
                    FROM {ctx.repository.schema}.frames frames
                    JOIN {ctx.repository.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE frames.id = %s AND assets.project_id = %s
                    """,
                    (id, project_id),
                )
            else:
                cursor.execute(
                    f"""
                    SELECT frames.*, assets.project_id AS project_id
                    FROM {ctx.repository.schema}.frames frames
                    JOIN {ctx.repository.schema}.raw_assets assets ON assets.id = frames.asset_id
                    WHERE frames.id = %s
                    """,
                    (id,),
                )
            row = cursor.fetchone()
    if row is None:
        raise KeyError(id)

    record = FrameRecord.from_row(row)
    project_id = None if row.get("project_id") is None else str(row["project_id"])
    kvstore = _kvstore_for_project(ctx, project_id)
    if kvstore is None:
        raise RuntimeError("A KVStore is required to retrieve frame data.")
    requested_kind = str(payload_kind or "original").lower()
    if requested_kind in {"preprocessed", "processed", "corrected"}:
        metadata = {**dict(record.metadata or {}), **dict(record.preprocessed_metadata or {})}
        kvstore_key = record.preprocessed_payload_ref or record.preprocessed_kvstore_hash
        if record.preprocessed_payload_encoding is not None:
            metadata["kvstore_encoding"] = record.preprocessed_payload_encoding
        if record.preprocessed_payload_format is not None:
            metadata["kvstore_format"] = record.preprocessed_payload_format
        if record.preprocessed_payload_dtype is not None:
            metadata["dtype"] = record.preprocessed_payload_dtype
        if record.preprocessed_payload_shape:
            metadata["shape"] = list(record.preprocessed_payload_shape)
        metadata["frame_variant"] = "preprocessed"
    elif requested_kind in {"original", "raw"}:
        metadata = dict(record.metadata or {})
        kvstore_key = record.payload_ref or metadata.get("kvstore_key") or record.kvstore_hash
        metadata["frame_variant"] = "original"
    else:
        raise ValueError("payload_kind must be one of: original, raw, preprocessed, processed, corrected.")
    if not kvstore_key:
        raise ValueError(f"Frame {id} does not include a {requested_kind} kvstore key.")

    frame_data = FrameData.from_record(record, metadata=metadata)
    array = decode_array_payload(kvstore.get_store(kvstore_key), frame_data.metadata)
    frame_data.update(array)
    background_key = record.background_payload_ref or record.background_kvstore_hash
    if background_key:
        background_metadata = {
            **dict(record.background_metadata or {}),
            "kvstore_key": background_key,
        }
        if record.background_payload_encoding is not None:
            background_metadata["kvstore_encoding"] = record.background_payload_encoding
        if record.background_payload_format is not None:
            background_metadata["kvstore_format"] = record.background_payload_format
        if record.background_payload_dtype is not None:
            background_metadata["dtype"] = record.background_payload_dtype
        if record.background_payload_shape:
            background_metadata["shape"] = list(record.background_payload_shape)
        frame_data.update_background(
            decode_array_payload(kvstore.get_store(background_key), background_metadata)
        )
        frame_data.metadata["background_payload_ref"] = background_key
        frame_data.metadata["background_payload_encoding"] = record.background_payload_encoding
        frame_data.metadata["background_payload_format"] = record.background_payload_format
        frame_data.metadata["background_payload_dtype"] = record.background_payload_dtype
        frame_data.metadata["background_payload_shape"] = list(record.background_payload_shape)
    _CORE_LOGGER.debug(
        "Retrieved frame id=%s run_id=%s asset_id=%s shape=%s duration_ms=%.2f",
        id,
        record.run_id,
        record.asset_id,
        list(array.shape),
        (time.perf_counter() - started) * 1000,
    )
    return frame_data
