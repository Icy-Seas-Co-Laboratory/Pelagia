"""Apply flatfield/background correction and generate reusable backgrounds."""

from collections import Counter

import numpy as np

from ..domain import FrameRecord
from ..services.context import AppContext
from ..utils.serialization import json_ready
from .defaults import default_processing_config
from .frame_codec import encode_array_payload
from .frame_model import FrameData
from .timing import measure_phase


def _nominal_frame_geometry(frames: list[FrameRecord]) -> tuple[int, int]:
    geometries = Counter((frame.width, frame.height) for frame in frames)
    if not geometries:
        raise ValueError("Cannot determine nominal frame geometry without frames.")
    return max(
        geometries,
        key=lambda geometry: (geometries[geometry], geometry[0] * geometry[1]),
    )


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


def build_background_payload_for_frames(
    frame_ids: list[str] | tuple[str, ...],
    *,
    context: AppContext | None = None,
    payload_kind: str = "original",
    encoding: str = "zstd",
    quality: int | None = None,
) -> dict:
    """Build a mean background field from stored frames and store it in the KVStore."""
    resolved_frame_ids = [str(frame_id) for frame_id in frame_ids]
    if not resolved_frame_ids:
        raise ValueError("Background generation requires at least one frame_id.")

    ctx = context
    if ctx is None:
        from .frame_store import default_context

        ctx = default_context()
    if ctx.kvstore is None:
        raise RuntimeError("A KVStore is required to store background data.")
    if ctx.repository is None:
        raise RuntimeError("A PostgresRepository is required to record background metadata.")

    from .frame_store import frame_id_work_units, retrieve_frame

    accumulator: np.ndarray | None = None
    frame_shape: tuple[int, ...] | None = None
    bulk_loader = getattr(ctx.repository, "get_frame_records", None)
    for work_frame_ids in frame_id_work_units(resolved_frame_ids):
        records_by_id = {}
        if callable(bulk_loader):
            with measure_phase("background.frame_metadata_lookup"):
                records_by_id = {
                    str(record.id): record
                    for record in bulk_loader(
                        work_frame_ids,
                        project_id=getattr(ctx, "active_project_id", None),
                    )
                }
            missing = [frame_id for frame_id in work_frame_ids if frame_id not in records_by_id]
            if missing:
                raise KeyError(f"Frame {missing[0]!r} was not found.")
        for frame_id in work_frame_ids:
            retrieve_kwargs = {
                "context": ctx,
                "payload_kind": payload_kind,
            }
            if records_by_id:
                retrieve_kwargs["frame_record"] = records_by_id[frame_id]
            frame = retrieve_frame(frame_id, **retrieve_kwargs)
            with measure_phase("background.accumulate"):
                data = frame.read()
                if data is None:
                    raise ValueError(f"Frame {frame_id!r} has no image data.")
                array = np.asarray(data, dtype=np.float32)
                if array.ndim < 2:
                    raise ValueError(f"Frame {frame_id!r} data must have at least two dimensions.")
                if frame_shape is None:
                    frame_shape = tuple(array.shape)
                    # Float64 accumulation avoids drift when wide windows contain many frames.
                    accumulator = np.zeros(frame_shape, dtype=np.float64)
                elif tuple(array.shape) != frame_shape:
                    raise ValueError(
                        f"Frame {frame_id!r} shape {tuple(array.shape)} does not match {frame_shape}."
                    )
                accumulator += array

    if accumulator is None:
        raise ValueError("No frame data was loaded.")
    with measure_phase("background.finalize"):
        background = np.ascontiguousarray(
            np.clip(
                np.rint(accumulator / len(resolved_frame_ids)),
                0,
                np.iinfo(np.uint8).max,
            ).astype(np.uint8)
        )
    resolved_quality = ctx.config.processing.frame_storage.image_quality if quality is None else int(quality)
    with measure_phase("background.encode"):
        payload, payload_encoding, payload_format = encode_array_payload(
            background,
            encoding,
            quality=resolved_quality,
        )
    with measure_phase("background.kvstore_write"):
        kvstore_key = ctx.kvstore.put_store(payload)
    metadata = {
        "frame_variant": "background",
        "background_method": "mean",
        "background_source_payload_kind": payload_kind,
        "background_source_frame_ids": resolved_frame_ids,
        "background_source_frame_count": len(resolved_frame_ids),
        "background_layout": "nominal_frame",
        "kvstore_key": kvstore_key,
        "kvstore_hash": kvstore_key,
        "kvstore_encoding": payload_encoding,
        "kvstore_format": payload_format,
        "kvstore_quality": resolved_quality,
        "dtype": str(background.dtype),
        "shape": list(background.shape),
    }
    return {
        "background_payload_ref": kvstore_key,
        "background_payload_encoding": payload_encoding,
        "background_payload_format": payload_format,
        "background_payload_quality": resolved_quality,
        "background_payload_dtype": str(background.dtype),
        "background_payload_shape": list(background.shape),
        "background_method": "mean",
        "background_metadata": json_ready(metadata),
        "frame_ids": resolved_frame_ids,
        "frame_count": len(resolved_frame_ids),
        "metadata": json_ready(metadata),
    }


def generate_background_for_frames(
    frame_ids: list[str] | tuple[str, ...],
    *,
    context: AppContext | None = None,
    payload_kind: str = "original",
    encoding: str = "zstd",
    quality: int | None = None,
    metadata: dict | None = None,
) -> dict:
    """
    Build a mean background field from stored frames and assign it to those frames.

    The generated field is stored once in the KVStore and the resulting payload ref
    is written back to every frame row listed in ``frame_ids``.
    """
    result = build_background_payload_for_frames(
        frame_ids,
        context=context,
        payload_kind=payload_kind,
        encoding=encoding,
        quality=quality,
    )
    resolved_frame_ids = [str(frame_id) for frame_id in frame_ids]
    ctx = context
    if ctx is None:
        from .frame_store import default_context

        ctx = default_context()
    background_metadata = {**dict(result["background_metadata"] or {}), **dict(metadata or {})}
    with measure_phase("background.database_update"):
        rows = ctx.repository.update_frame_background_payloads(
            resolved_frame_ids,
            project_id=getattr(ctx, "active_project_id", None),
            kvstore_hash=str(result["background_payload_ref"]),
            payload_ref=str(result["background_payload_ref"]),
            payload_encoding=str(result["background_payload_encoding"]),
            payload_format=str(result["background_payload_format"]),
            payload_dtype=str(result["background_payload_dtype"]),
            payload_shape=list(result["background_payload_shape"] or []),
            metadata=background_metadata,
        )
    return {
        **result,
        "background_metadata": background_metadata,
        "updated_frame_count": len(rows),
    }


def ensure_asset_background_windows(
    frame_ids: list[str] | tuple[str, ...],
    *,
    context: AppContext,
    payload_kind: str = "original",
    encoding: str = "zstd",
    quality: int | None = None,
    window_stride: int | None = None,
    window_width: int | None = None,
) -> dict:
    """Ensure selected frames have an asset-wide, fixed-stride background field."""
    if context.repository is None:
        raise RuntimeError("A PostgresRepository is required to generate backgrounds.")
    defaults = context.config.processing.preprocessing
    stride = defaults.background_window_stride if window_stride is None else int(window_stride)
    width = defaults.background_window_width if window_width is None else int(window_width)
    if stride < 1 or width < 1 or stride % 2 == 0 or width % 2 == 0:
        raise ValueError("background_window_stride and background_window_width must be positive odd integers.")

    from .frame_store import frame_id_work_units

    targets_by_asset: dict[str, list[FrameRecord]] = {}
    selected_frame_ids = list(dict.fromkeys(str(value) for value in frame_ids))
    bulk_loader = getattr(context.repository, "get_frame_records", None)
    for work_frame_ids in frame_id_work_units(selected_frame_ids):
        with measure_phase("selection.frame_metadata_lookup"):
            if callable(bulk_loader):
                records = bulk_loader(work_frame_ids, project_id=context.active_project_id)
            else:
                records = [
                    context.repository.get_frame_record(
                        frame_id,
                        project_id=context.active_project_id,
                    )
                    for frame_id in work_frame_ids
                ]
        records_by_id = {
            str(record.id): record
            for record in records
            if record is not None
        }
        missing = [frame_id for frame_id in work_frame_ids if frame_id not in records_by_id]
        if missing:
            raise KeyError(f"Frame {missing[0]!r} was not found.")
        for frame_id in work_frame_ids:
            record = records_by_id[frame_id]
            targets_by_asset.setdefault(record.asset_id, []).append(record)

    windows: list[dict] = []
    application_half_width = stride // 2
    source_half_width = width // 2
    for asset_id, targets in targets_by_asset.items():
        with measure_phase("background.asset_frame_lookup"):
            rows = context.repository.list_frames(
                asset_id,
                project_id=context.active_project_id,
                limit=None,
            )
        frames = [FrameRecord.from_row(row) for row in rows]
        nominal_width, nominal_height = _nominal_frame_geometry(frames)
        by_center: dict[int, list[FrameRecord]] = {}
        for target in targets:
            # Fixed boundaries let neighboring jobs reuse the same background payload.
            center = ((target.frame_index + application_half_width) // stride) * stride
            by_center.setdefault(center, []).append(target)
        for center, window_targets in by_center.items():
            first_index = center - source_half_width
            last_index = center + source_half_width
            sources = [
                frame
                for frame in frames
                if first_index <= frame.frame_index <= last_index
                and frame.width == nominal_width
                and frame.height == nominal_height
            ]
            if not sources:
                raise ValueError(
                    "No nominal-sized frames for background window "
                    f"centered at {center} with geometry {nominal_width}x{nominal_height}."
                )
            current_metadata = window_targets[0].background_metadata
            if (
                window_targets[0].background_payload_ref
                and current_metadata.get("background_window_center") == center
                and current_metadata.get("background_window_stride") == stride
                and current_metadata.get("background_window_width") == width
                and current_metadata.get("background_source_payload_kind") == payload_kind
                and current_metadata.get("background_nominal_width") == nominal_width
                and current_metadata.get("background_nominal_height") == nominal_height
            ):
                reused_background = window_targets[0]
                missing_target_ids = [
                    str(target.id)
                    for target in window_targets
                    if target.id is not None
                    and target.background_payload_ref != reused_background.background_payload_ref
                ]
                if missing_target_ids:
                    context.repository.update_frame_background_payloads(
                        missing_target_ids,
                        project_id=context.active_project_id,
                        kvstore_hash=str(
                            reused_background.background_kvstore_hash
                            or reused_background.background_payload_ref
                        ),
                        payload_ref=str(reused_background.background_payload_ref),
                        payload_encoding=str(reused_background.background_payload_encoding),
                        payload_format=str(reused_background.background_payload_format),
                        payload_dtype=str(reused_background.background_payload_dtype),
                        payload_shape=list(reused_background.background_payload_shape or []),
                        metadata=current_metadata,
                    )
                windows.append(
                    {
                        "asset_id": asset_id,
                        "center": center,
                        "nominal_width": nominal_width,
                        "nominal_height": nominal_height,
                        "reused": True,
                        "frame_ids": [str(frame.id) for frame in sources],
                    }
                )
                continue
            source_ids = [str(frame.id) for frame in sources if frame.id is not None]
            result = generate_background_for_frames(
                source_ids,
                context=context,
                payload_kind=payload_kind,
                encoding=encoding,
                quality=quality,
                metadata={
                    "background_window_center": center,
                    "background_window_start": first_index,
                    "background_window_end": last_index,
                    "background_window_stride": stride,
                    "background_window_width": width,
                    "background_application_start": center - application_half_width,
                    "background_application_end": center + application_half_width,
                    "background_nominal_width": nominal_width,
                    "background_nominal_height": nominal_height,
                },
            )
            source_id_set = set(source_ids)
            additional_target_ids = [
                str(target.id)
                for target in window_targets
                if target.id is not None and str(target.id) not in source_id_set
            ]
            if additional_target_ids:
                context.repository.update_frame_background_payloads(
                    additional_target_ids,
                    project_id=context.active_project_id,
                    kvstore_hash=str(result["background_payload_ref"]),
                    payload_ref=str(result["background_payload_ref"]),
                    payload_encoding=str(result["background_payload_encoding"]),
                    payload_format=str(result["background_payload_format"]),
                    payload_dtype=str(result["background_payload_dtype"]),
                    payload_shape=list(result["background_payload_shape"] or []),
                    metadata=dict(result["background_metadata"] or {}),
                )
            windows.append(
                {
                    "asset_id": asset_id,
                    "center": center,
                    "nominal_width": nominal_width,
                    "nominal_height": nominal_height,
                    "reused": False,
                    **result,
                }
            )
    return {"window_stride": stride, "window_width": width, "windows": windows}
