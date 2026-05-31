import json
from datetime import datetime
from typing import Any

import numpy as np

from ..domain import FrameRecord
from ..services.context import AppContext
from ..utils.serialization import json_ready
from .frame_codec import decode_array_payload, encode_array_payload
from .frame_correction import flatfield_correction_for_framedata, metadata_bool
from .frame_model import FrameData


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


def store_frame(frame: FrameData, context: AppContext | None = None) -> dict[str, Any]:
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
        frame_hash=kvstore_key,
        frame_png=b"",
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
                 bbox_x, bbox_y, parent_frame_id, source_ref, frame_hash, frame_png,
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
                    frame_hash = EXCLUDED.frame_hash,
                    frame_png = EXCLUDED.frame_png,
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
                    frame_record.frame_hash,
                    frame_record.frame_png,
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
    return row


def retrieve_frame(id: int, context: AppContext | None = None) -> FrameData:
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

    record = FrameRecord.from_row(row)
    metadata = dict(record.metadata or {})
    kvstore_key = record.payload_ref or metadata.get("kvstore_key") or record.frame_hash
    if not kvstore_key:
        raise ValueError(f"Frame {id} does not include a kvstore key.")

    frame_data = FrameData.from_record(record)
    array = decode_array_payload(ctx.kvstore.get_store(kvstore_key), frame_data.metadata)
    frame_data.update(array)
    return frame_data
