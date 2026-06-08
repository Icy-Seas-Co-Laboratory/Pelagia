import numpy as np

from ..domain import FrameRecord
from ..services.context import AppContext
from .defaults import default_processing_config
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
) -> np.ndarray:
    if not 0 < q < 1:
        raise ValueError("q must be between 0 and 1.")

    array = np.asarray(data)
    if array.ndim < 2:
        raise ValueError("flatfield correction requires at least two dimensions.")

    field = np.maximum(_flatfield_profile(array, q=q, axis=axis), min_field_value)
    #scale = float(np.max(field)) if np.any(field > 0) else 1.0
    corrected = array.astype(np.float32) / field * 255

    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        corrected = np.clip(corrected, info.min, info.max).astype(array.dtype)

    return np.ascontiguousarray(corrected)


def flatfield_correction_for_framedata(
    data: np.ndarray,
    q: float | None = None,
    min_field_value: float | None = None,
) -> np.ndarray:
    defaults = default_processing_config().flatfield
    resolved_q = defaults.flatfield_q if q is None else q

    array = np.asarray(data)
    if array.ndim != 2:
        raise ValueError("flatfield correction currently expects a 2D grayscale frame.")

    return _flatfield_correction(
        array,
        q=resolved_q,
        axis=0,
        min_field_value=1.0,
    )


def flatfield_global_correction_for_framedata(
    data: np.ndarray,
    q: float | None = None,
    axis: int | None = None,
) -> np.ndarray:
    defaults = default_processing_config().flatfield
    resolved_q = defaults.flatfield_q if q is None else q
    resolved_axis = defaults.flatfield_axis if axis is None else axis
    return _flatfield_correction(
        data,
        q=resolved_q,
        axis=resolved_axis,
        min_field_value=1.0,
    )


def apply_flatfield_correction(
    frame_record: FrameRecord,
    *,
    frame: FrameData | None = None,
    q: float | None = None,
    axis: int | None = None,
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

    source_frame = frame
    if source_frame is None:
        if frame_record.id is None:
            raise ValueError("FrameRecord must include id when frame data is not supplied.")
        from .frame_store import retrieve_frame

        source_frame = retrieve_frame(str(frame_record.id), context=context)

    data = source_frame.read()
    if data is None:
        raise ValueError("Frame has no image data to flatfield correct.")

    corrected = flatfield_global_correction_for_framedata(
        data,
        q=resolved_q,
        axis=resolved_axis,
    )
    metadata = dict(source_frame.metadata or {})
    metadata.update(
        {
            "flatfield_correction": True,
            "flatfield_q": float(resolved_q),
            "flatfield_axis": int(resolved_axis),
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
