"""Public frame-processing facade.

Implementation lives in focused sibling modules; this module preserves the
historical import path for callers.
"""

from .frame_codec import (
    _decode_array_payload,
    _encode_array_payload,
    decode_array_payload,
    encode_array_payload,
)
from .frame_correction import (
    _flatfield_correction_for_framedata,
    _metadata_bool,
    flatfield_correction_for_framedata,
    metadata_bool,
)
from .frame_model import Frame
from .frame_store import (
    _default_context,
    _metadata_without_none,
    default_context,
    metadata_without_none,
    retrieve_frame,
    store_frame,
)
from .frame_time import (
    _parse_filename_timestamp_utc,
    _timestamp_for_frame,
    parse_filename_timestamp_utc,
    timestamp_for_frame,
)
from .video_ingest import ingest_video_file


__all__ = [
    "Frame",
    "decode_array_payload",
    "default_context",
    "encode_array_payload",
    "flatfield_correction_for_framedata",
    "ingest_video_file",
    "metadata_bool",
    "metadata_without_none",
    "parse_filename_timestamp_utc",
    "retrieve_frame",
    "store_frame",
    "timestamp_for_frame",
]
