"""Portable archival interchange for scientific image datasets."""

from .builder import DatasetBuilder
from .constants import FORMAT_VERSION, LIBRARY_VERSION, SCHEMA_VERSION
from .dataset import Dataset
from .exceptions import CompatibilityError, DatasetStateError, FormatError, FrameNotFoundError, IntegrityError, InterchangeError, UnsafePathError
from .history import History
from .ingestion import VideoIngestionError, VideoIngestionResult, VideoProbe, discover_videos, ingest_video_directory, probe_video
from .manifest import Manifest
from .metadata import Metadata
from .models import Frame, FrameRecord, FrameStatus, HashRecord, SourceFile, StorageFormat
from .shard import Shard, ShardReader, ShardWriter
from .validation import Validator, VerificationResult

__version__ = LIBRARY_VERSION
__all__ = ["CompatibilityError", "Dataset", "DatasetBuilder", "DatasetStateError", "FORMAT_VERSION",
           "FormatError", "Frame", "FrameNotFoundError", "FrameRecord", "FrameStatus", "HashRecord", "History",
           "IntegrityError", "InterchangeError", "LIBRARY_VERSION", "Manifest", "Metadata", "SCHEMA_VERSION", "Shard",
           "ShardReader", "ShardWriter", "SourceFile", "StorageFormat", "UnsafePathError", "Validator", "VerificationResult"]
__all__ += ["VideoIngestionError", "VideoIngestionResult", "VideoProbe", "discover_videos", "ingest_video_directory", "probe_video"]
