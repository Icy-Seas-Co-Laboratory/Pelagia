import json
import os
from datetime import datetime
from typing import Any

import numpy as np

from ..services.context import AppContext
from ..utils.serialization import json_ready
from .frame_codec import decode_array_payload, encode_array_payload
from .frame_correction import flatfield_correction_for_framedata, metadata_bool
from .frame_model import Frame


_DEFAULT_CONTEXT: AppContext | None = None


def default_context() -> AppContext:
    global _DEFAULT_CONTEXT
    if _DEFAULT_CONTEXT is None:
        _DEFAULT_CONTEXT = AppContext.from_config()
        if _DEFAULT_CONTEXT.kvstore is not None and not _DEFAULT_CONTEXT.kvstore.initialized:
            config = _DEFAULT_CONTEXT.config.kvstore
            _DEFAULT_CONTEXT.kvstore.initialize(
                hash_algorithm=config.hash_algorithm,
                prefix_length=config.prefix_length,
                max_db_bytes=config.max_db_bytes,
                max_rows=config.max_rows,
            )
    return _DEFAULT_CONTEXT


def metadata_without_none(metadata: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metadata.items() if value is not None}


def store_frame(frame: Frame, context: AppContext | None = None) -> dict[str, Any]:
    data = frame.read()
    if data is None:
        raise ValueError("Frame has no numpy data to store.")

    metadata = dict(frame.metadata or {})
    if metadata_bool(metadata, "flatfield_correction"):
        flatfield_q = float(metadata.get("flatfield_q", 0.9))
        data = flatfield_correction_for_framedata(data, q=flatfield_q)
        metadata.update(
            {
                "flatfield_correction": True,
                "flatfield_q": flatfield_q,
            }
        )

    array = np.ascontiguousarray(data)
    if array.ndim < 2:
        raise ValueError("Frame data must have at least two dimensions.")
    frame.validate_geometry(array)

    ctx = context or default_context()
    if ctx.kvstore is None:
        raise RuntimeError("A KVStore is required to store frame data.")
    if ctx.repository is None:
        raise RuntimeError("A PostgresRepository is required to record frame metadata.")

    default_encoding = getattr(
        getattr(ctx.config, "image_data_storage", None),
        "encoding",
        "png",
    )
    requested_encoding = metadata.get(
        "kvstore_encoding",
        metadata.get("array_encoding", metadata.get("kvstore_format", default_encoding)),
    )
    payload, kvstore_encoding, kvstore_format = encode_array_payload(array, requested_encoding)
    kvstore_key = ctx.kvstore.put_store(payload)
    width, height = frame.get_size()
    source_frame_start, source_frame_end = frame.get_source_frame_range()
    captured_at = frame.timestamp if isinstance(frame.timestamp, datetime) else None

    run_id = getattr(frame, "run_id", None) or metadata.get("run_id")
    asset_id = getattr(frame, "asset_id", None) or metadata.get("asset_id")
    if not run_id or not asset_id:
        raise ValueError("Frame metadata must include run_id and asset_id.")

    frame_index = metadata.get("frame_index")
    if frame_index is None:
        frame_index = frame.tileNumber if frame.tileNumber is not None else frame.frameNumber

    metadata.update(
        metadata_without_none(
            {
                "kvstore_key": kvstore_key,
                "kvstore_encoding": kvstore_encoding,
                "kvstore_format": kvstore_format,
                "dtype": str(array.dtype),
                "shape": list(array.shape),
                "width": width,
                "height": height,
                "bbox_x": frame.bbox_x,
                "bbox_y": frame.bbox_y,
                "parent_frame_id": frame.parent_frame_id,
                "source_path": frame.sourcePath,
                "dest_path": frame.destPath,
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

    with ctx.repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"""
                INSERT INTO {ctx.repository.schema}.frames
                (run_id, asset_id, frame_index, captured_at, width, height, source_ref, frame_hash, frame_png, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (asset_id, frame_index) DO UPDATE SET
                    captured_at = EXCLUDED.captured_at,
                    width = EXCLUDED.width,
                    height = EXCLUDED.height,
                    source_ref = EXCLUDED.source_ref,
                    frame_hash = EXCLUDED.frame_hash,
                    frame_png = EXCLUDED.frame_png,
                    metadata = EXCLUDED.metadata
                RETURNING *;
                """,
                (
                    run_id,
                    asset_id,
                    int(frame_index),
                    captured_at,
                    width,
                    height,
                    frame.get_source_file_path(),
                    kvstore_key,
                    b"",
                    json.dumps(json_ready(metadata)),
                ),
            )
            row = cursor.fetchone()
        connection.commit()
    return row


def retrieve_frame(id: int, context: AppContext | None = None) -> Frame:
    ctx = context or default_context()
    if ctx.kvstore is None:
        raise RuntimeError("A KVStore is required to retrieve frame data.")
    if ctx.repository is None:
        raise RuntimeError("A PostgresRepository is required to load frame metadata.")

    with ctx.repository.connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"SELECT * FROM {ctx.repository.schema}.frames WHERE id = %s", (id,))
            row = cursor.fetchone()
    if row is None:
        raise KeyError(id)

    metadata = dict(row.get("metadata") or {})
    kvstore_key = metadata.get("kvstore_key") or row.get("frame_hash")
    if not kvstore_key:
        raise ValueError(f"Frame {id} does not include a kvstore key.")

    array = decode_array_payload(ctx.kvstore.get_store(kvstore_key), metadata)
    source_ref = row.get("source_ref") or ""
    source_path = metadata.get("source_path") or os.path.dirname(source_ref)
    filename = metadata.get("filename") or os.path.basename(source_ref)

    metadata.setdefault("frame_id", row.get("id"))
    metadata.setdefault("run_id", str(row.get("run_id")))
    metadata.setdefault("asset_id", str(row.get("asset_id")))
    metadata.setdefault("frame_index", row.get("frame_index"))

    return Frame(
        sourcePath=source_path,
        destPath=metadata.get("dest_path") or source_path,
        filename=filename,
        frameNumber=metadata.get("frame_number") or row["frame_index"],
        data=array,
        width=row.get("width") or metadata.get("width"),
        height=row.get("height") or metadata.get("height"),
        bbox_x=metadata.get("bbox_x", 0),
        bbox_y=metadata.get("bbox_y", 0),
        parent_frame_id=metadata.get("parent_frame_id"),
        tileNumber=metadata.get("tile_number"),
        sourceFrameStart=metadata.get("source_frame_start"),
        sourceFrameEnd=metadata.get("source_frame_end"),
        frameType=metadata.get("frame_type"),
        channel=metadata.get("channel"),
        timestamp=row.get("captured_at") or metadata.get("timestamp"),
        metadata=metadata,
    )


_default_context = default_context
_metadata_without_none = metadata_without_none
