import numpy as np

from ..domain import FrameRecord
from ..services.context import AppContext
from ..utils.serialization import json_ready
from .defaults import default_processing_config
from .frame_codec import encode_array_payload
from .frame_model import FrameData


def _flatfield_profile(array: np.ndarray, *, q: float, axis: int) -> np.ndarray:
    if axis == 0:
        return np.quantile(array, q=q, axis=0).astype(np.float32)
    if axis == 1:
        if array.ndim == 2:
            return np.quantile(array, q=q, axis=1).astype(np.float32)[:, None]
        return np.quantile(array, q=q, axis=1).astype(np.float32)[:, None, :]
    raise ValueError("flatfield axis must be 0 or 1.")


def _flatfield_correction(
    data: np.ndarray,
    *,
    q: float,
    axis: int,
    min_field_value: float = 1.0,
    max_field_value: float | None = None,
) -> np.ndarray:
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1.")

    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("flatfield correction requires at least two dimensions.")

    field = _bounded_field(
        _flatfield_profile(array, q=q, axis=axis),
        min_field_value=min_field_value,
        max_field_value=max_field_value,
    )
    return _divide_by_field(array, field)


def flatfield_correction(
    data: np.ndarray,
    q: float | None = None,
    axis: int | None = None,
    min_field_value: float | None = None,
    max_field_value: float | None = None,
) -> np.ndarray:
    """Apply quantile-profile flatfield correction to an image array."""
    defaults = default_processing_config().flatfield
    resolved_q = defaults.flatfield_q if q is None else q
    resolved_axis = defaults.flatfield_axis if axis is None else axis
    resolved_min_field_value = (
        defaults.flatfield_min_field_value if min_field_value is None else min_field_value
    )
    resolved_max_field_value = (
        defaults.flatfield_max_field_value if max_field_value is None else max_field_value
    )
    return _flatfield_correction(
        data,
        q=resolved_q,
        axis=resolved_axis,
        min_field_value=resolved_min_field_value,
        max_field_value=resolved_max_field_value,
    )


def _bounded_field(
    field: np.ndarray | int | float,
    *,
    min_field_value: float = 1.0,
    max_field_value: float | None = None,
) -> np.ndarray:
    resolved_min = float(min_field_value)
    if resolved_min < 0:
        raise ValueError("min_field_value must be >= 0.")
    if max_field_value is not None and float(max_field_value) < resolved_min:
        raise ValueError("max_field_value must be >= min_field_value.")

    bounded = np.asarray(field, dtype=np.float32)
    bounded = np.where(bounded < resolved_min, 0.0, bounded).astype(np.float32, copy=False)
    if max_field_value is not None:
        bounded = np.where(bounded > float(max_field_value), 255.0, bounded).astype(
            np.float32,
            copy=False,
        )
    return bounded


def _divide_by_field(data: np.ndarray, field: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    field_array = np.asarray(field, dtype=np.float32)
    output_shape = np.broadcast_shapes(array.shape, field_array.shape)
    corrected = np.ones(output_shape, dtype=np.float32)
    np.divide(
        array.astype(np.float32),
        field_array,
        out=corrected,
        where=field_array != 0,
    )
    corrected *= 255 # TODO Does this need to be hardcoded for 8bit grayscale?

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(array.dtype)

    return np.ascontiguousarray(corrected)


def divide_background(
    data: np.ndarray,
    *,
    background: np.ndarray | int | float,
    min_field_value: float = 1.0,
    max_field_value: float | None = None,
) -> np.ndarray:
    """Divide image data by a scalar/image background field and rescale."""
    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("background division requires at least two dimensions.")

    field = _bounded_field(
        background,
        min_field_value=min_field_value,
        max_field_value=max_field_value,
    )
    if field.ndim > 0 and field.shape[:2] != array.shape[:2]:
        raise ValueError(
            f"background shape {field.shape[:2]} does not match image shape {array.shape[:2]}."
        )
    return _divide_by_field(array, field)


def apply_flatfield_correction(
    frame_record: FrameRecord,
    *,
    frame: FrameData | None = None,
    q: float | None = None,
    axis: int | None = None,
    min_field_value: float | None = None,
    max_field_value: float | None = None,
    context: AppContext | None = None,
) -> FrameData:
    """
    Apply flatfield correction to a stored frame and return an in-memory frame container.

    ``frame_record`` carries the database identity and geometry. When ``frame`` is not
    supplied, the stored pixels are loaded from the configured frame store.
    """
    defaults = (
        context.config.processing.flatfield
        if context is not None and getattr(context, "config", None) is not None
        else default_processing_config().flatfield
    )
    resolved_q = defaults.flatfield_q if q is None else q
    resolved_axis = defaults.flatfield_axis if axis is None else axis
    resolved_min_field_value = (
        defaults.flatfield_min_field_value if min_field_value is None else min_field_value
    )
    resolved_max_field_value = (
        defaults.flatfield_max_field_value if max_field_value is None else max_field_value
    )

    source_frame = frame
    if source_frame is None:
        if frame_record.id is None:
            raise ValueError("FrameRecord must include id when frame data is not supplied.")
        from .frame_store import retrieve_frame

        source_frame = retrieve_frame(str(frame_record.id), context=context)

    data = source_frame.read()
    if data is None:
        raise ValueError("Frame has no image data to flatfield correct.")

    corrected = _flatfield_correction(
        data,
        q=resolved_q,
        axis=resolved_axis,
        min_field_value=resolved_min_field_value,
        max_field_value=resolved_max_field_value,
    )
    metadata = dict(source_frame.metadata or {})
    metadata.update(
        {
            "flatfield_correction": True,
            "flatfield_q": float(resolved_q),
            "flatfield_axis": int(resolved_axis),
            "flatfield_min_field_value": float(resolved_min_field_value),
            "flatfield_max_field_value": resolved_max_field_value,
            "flatfield_source_frame_id": frame_record.id,
        }
    )
    return FrameData(
        sourcePath=source_frame.sourcePath,
        filename=source_frame.filename,
        frameNumber=source_frame.frameNumber,
        data=corrected,
        mask=source_frame.mask,
        width=source_frame.width,
        height=source_frame.height,
        bbox_x=source_frame.bbox_x,
        bbox_y=source_frame.bbox_y,
        parent_frame_id=source_frame.parent_frame_id,
        bkg=source_frame.bkg,
        tileNumber=source_frame.tileNumber,
        sourceFrameStart=source_frame.sourceFrameStart,
        sourceFrameEnd=source_frame.sourceFrameEnd,
        frameType=source_frame.frameType,
        channel=source_frame.channel,
        timestamp=source_frame.timestamp,
        metadata=metadata,
        imageReadFlag=source_frame.imageReadFlag,
        cacheRead=source_frame.cacheRead,
    )


def generate_background_for_frames(
    frame_ids: list[str] | tuple[str, ...],
    *,
    context: AppContext | None = None,
    payload_kind: str = "original",
    encoding: str = "zstd",
) -> dict:
    """
    Build a mean background field from stored frames and assign it to those frames.

    The generated field is stored once in the KVStore and the resulting payload ref
    is written back to every frame row listed in ``frame_ids``.
    """
    resolved_frame_ids = [str(frame_id) for frame_id in frame_ids]
    if not resolved_frame_ids:
        raise ValueError("generate_background_for_frames requires at least one frame_id.")

    ctx = context
    if ctx is None:
        from .frame_store import default_context

        ctx = default_context()
    if ctx.kvstore is None:
        raise RuntimeError("A KVStore is required to store background data.")
    if ctx.repository is None:
        raise RuntimeError("A PostgresRepository is required to record background metadata.")

    from .frame_store import retrieve_frame

    accumulator: np.ndarray | None = None
    frame_shape: tuple[int, ...] | None = None
    for frame_id in resolved_frame_ids:
        frame = retrieve_frame(frame_id, context=ctx, payload_kind=payload_kind)
        data = frame.read()
        if data is None:
            raise ValueError(f"Frame {frame_id!r} has no image data.")
        array = np.asarray(data, dtype=np.float32)
        if array.ndim < 2:
            raise ValueError(f"Frame {frame_id!r} data must have at least two dimensions.")
        if frame_shape is None:
            frame_shape = tuple(array.shape)
            accumulator = np.zeros(frame_shape, dtype=np.float64)
        elif tuple(array.shape) != frame_shape:
            raise ValueError(
                f"Frame {frame_id!r} shape {tuple(array.shape)} does not match {frame_shape}."
            )
        accumulator += array

    if accumulator is None:
        raise ValueError("No frame data was loaded.")
    background = np.ascontiguousarray((accumulator / len(resolved_frame_ids)).astype(np.float32))
    payload, payload_encoding, payload_format = encode_array_payload(background, encoding)
    kvstore_key = ctx.kvstore.put_store(payload)
    metadata = {
        "frame_variant": "background",
        "background_method": "mean",
        "background_source_payload_kind": payload_kind,
        "background_source_frame_ids": resolved_frame_ids,
        "background_source_frame_count": len(resolved_frame_ids),
        "kvstore_key": kvstore_key,
        "kvstore_hash": kvstore_key,
        "kvstore_encoding": payload_encoding,
        "kvstore_format": payload_format,
        "dtype": str(background.dtype),
        "shape": list(background.shape),
    }
    rows = ctx.repository.update_frame_background_payloads(
        resolved_frame_ids,
        kvstore_hash=kvstore_key,
        payload_ref=kvstore_key,
        payload_encoding=payload_encoding,
        payload_format=payload_format,
        payload_dtype=str(background.dtype),
        payload_shape=list(background.shape),
        metadata=json_ready(metadata),
    )
    return {
        "background_payload_ref": kvstore_key,
        "background_payload_encoding": payload_encoding,
        "background_payload_format": payload_format,
        "background_payload_dtype": str(background.dtype),
        "background_payload_shape": list(background.shape),
        "frame_ids": resolved_frame_ids,
        "frame_count": len(resolved_frame_ids),
        "updated_frame_count": len(rows),
        "metadata": json_ready(metadata),
    }
