import os
from dataclasses import dataclass, field
from typing import Any

import cv2


@dataclass
class FrameData:
    """
    Runtime container for one image-like object plus source/output metadata.

    ``width`` and ``height`` describe this object's own image data. ``bbox_x``
    and ``bbox_y`` describe its origin in a parent image coordinate system, so
    a full frame starts at (0, 0) and an ROI can preserve its source position.

    The constructor keeps source metadata close to the runtime image payload so
    processing stages can pass frame, ROI, and mask data through one container.
    """
    sourcePath: str
    filename: str
    frameNumber: int
    data: object = None
    mask: object = None
    width: int | None = None
    height: int | None = None
    bbox_x: int = 0
    bbox_y: int = 0
    parent_frame_id: str | None = None
    bkg: object = None
    tileNumber: int = None
    sourceFrameStart: int = None
    sourceFrameEnd: int = None
    frameType: str = None
    channel: int = None
    timestamp: object = None
    metadata: dict = field(default_factory=dict)
    imageReadFlag: int = cv2.IMREAD_GRAYSCALE
    cacheRead: bool = True

    @classmethod
    def from_record(
        cls,
        record: Any,
        *,
        data: object = None,
        mask: object = None,
        metadata: dict | None = None,
    ) -> "FrameData":
        """Build runtime frame data from a stored frame row model."""
        resolved_metadata = dict(getattr(record, "metadata", {}) or {})
        resolved_metadata.update(metadata or {})

        if getattr(record, "payload_encoding", None) is not None:
            resolved_metadata.setdefault("kvstore_encoding", record.payload_encoding)
        if getattr(record, "payload_format", None) is not None:
            resolved_metadata.setdefault("kvstore_format", record.payload_format)
        if getattr(record, "payload_dtype", None) is not None:
            resolved_metadata.setdefault("dtype", record.payload_dtype)
        if getattr(record, "payload_shape", None):
            resolved_metadata.setdefault("shape", list(record.payload_shape))

        resolved_metadata.setdefault("frame_id", getattr(record, "id", None))
        resolved_metadata.setdefault("run_id", getattr(record, "run_id", None))
        resolved_metadata.setdefault("asset_id", getattr(record, "asset_id", None))
        resolved_metadata.setdefault("frame_index", getattr(record, "frame_index", None))

        source_ref = getattr(record, "source_ref", None) or ""
        source_path = resolved_metadata.get("source_path") or os.path.dirname(source_ref)
        filename = resolved_metadata.get("filename") or os.path.basename(source_ref)

        return cls(
            sourcePath=source_path,
            filename=filename,
            frameNumber=resolved_metadata.get("frame_number") or record.frame_index,
            data=data,
            mask=mask,
            width=record.width,
            height=record.height,
            bbox_x=record.bbox_x,
            bbox_y=record.bbox_y,
            parent_frame_id=record.parent_frame_id,
            tileNumber=resolved_metadata.get("tile_number"),
            sourceFrameStart=resolved_metadata.get("source_frame_start"),
            sourceFrameEnd=resolved_metadata.get("source_frame_end"),
            frameType=resolved_metadata.get("frame_type"),
            channel=resolved_metadata.get("channel"),
            timestamp=getattr(record, "captured_at", None) or resolved_metadata.get("timestamp"),
            metadata=resolved_metadata,
        )

    def __post_init__(self):
        self.sourcePath = os.fspath(self.sourcePath)
        self.filename = os.fspath(self.filename)
        self.bbox_x = 0 if self.bbox_x is None else int(self.bbox_x)
        self.bbox_y = 0 if self.bbox_y is None else int(self.bbox_y)
        if self.width is not None:
            self.width = int(self.width)
        if self.height is not None:
            self.height = int(self.height)
        self.infer_geometry()
        self.validate_mask()

    def infer_geometry(self, frame=None):
        """Populate missing width/height from in-memory image data."""
        image = self.data if frame is None else frame
        if image is None:
            image = self.mask
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 2:
            return None

        inferred_height = int(shape[0])
        inferred_width = int(shape[1])
        if self.width is None:
            self.width = inferred_width
        if self.height is None:
            self.height = inferred_height
        return (self.width, self.height)

    def validate_geometry(self, frame=None):
        """Ensure explicit geometry matches in-memory image data."""
        image = self.data if frame is None else frame
        shape = getattr(image, "shape", None)
        if shape is None or len(shape) < 2:
            return

        inferred_height = int(shape[0])
        inferred_width = int(shape[1])
        if self.width is not None and self.width != inferred_width:
            raise ValueError(
                f"Frame width {self.width} does not match image width {inferred_width}."
            )
        if self.height is not None and self.height != inferred_height:
            raise ValueError(
                f"Frame height {self.height} does not match image height {inferred_height}."
            )
        self.width = inferred_width
        self.height = inferred_height

    def validate_mask(self, mask=None):
        """Ensure an ROI mask matches this frame's image geometry."""
        image_mask = self.mask if mask is None else mask
        shape = getattr(image_mask, "shape", None)
        if shape is None:
            return
        if len(shape) < 2:
            raise ValueError("Frame mask must have at least two dimensions.")

        self.infer_geometry()
        mask_height = int(shape[0])
        mask_width = int(shape[1])
        if self.width is not None and self.width != mask_width:
            raise ValueError(
                f"Frame mask width {mask_width} does not match frame width {self.width}."
            )
        if self.height is not None and self.height != mask_height:
            raise ValueError(
                f"Frame mask height {mask_height} does not match frame height {self.height}."
            )

    def get_size(self):
        self.infer_geometry()
        return (self.width, self.height)

    def get_bbox(self):
        self.infer_geometry()
        return (self.bbox_x, self.bbox_y, self.width, self.height)

    def get_bounds(self):
        self.infer_geometry()
        if self.width is None or self.height is None:
            return (self.bbox_x, self.bbox_y, None, None)
        return (
            self.bbox_x,
            self.bbox_y,
            self.bbox_x + self.width,
            self.bbox_y + self.height,
        )

    def get_frame_number(self):
        return self.frameNumber

    def get_source_path(self):
        return self.sourcePath

    def get_filename(self):
        return self.filename

    def get_background(self):
        return self.bkg

    def get_tile_number(self):
        return self.tileNumber

    def get_frame_type(self):
        return self.frameType

    def get_channel(self):
        return self.channel

    def get_timestamp(self):
        return self.timestamp

    def get_metadata(self, key=None, default=None):
        if key is None:
            return self.metadata
        return self.metadata.get(key, default)

    def get_source_file_path(self):
        if os.path.isabs(self.filename):
            return self.filename
        if self.sourcePath.endswith(os.path.sep) or os.path.isdir(self.sourcePath):
            return os.path.join(self.sourcePath, self.filename)
        return self.sourcePath + self.filename

    def get_source_frame_range(self):
        return (self.sourceFrameStart, self.sourceFrameEnd)

    def get_mask(self):
        return self.mask

    def has_background(self):
        return self.bkg is not None

    def is_loaded(self):
        return self.data is not None

    def read(self):
        if self.data is not None:
            return self.data

        source_file_path = self.get_source_file_path()
        image = cv2.imread(source_file_path, self.imageReadFlag)
        self.infer_geometry(image)
        if self.cacheRead:
            self.data = image
        return image

    def update(self, newframe):
        self.data = newframe
        self.width = None
        self.height = None
        self.infer_geometry()
        self.validate_mask()

    def update_mask(self, mask):
        self.mask = mask
        self.validate_mask()

    def clear_data(self):
        self.data = None

    def update_background(self, background):
        self.bkg = background

    def update_metadata(self, **metadata):
        self.metadata.update(metadata)

    def shape(self):
        frame = self.read()
        if frame is None:
            return None
        return frame.shape

    def dtype(self):
        frame = self.read()
        if frame is None:
            return None
        return frame.dtype

    @property
    def source_path(self):
        return self.sourcePath

    @property
    def frame_number(self):
        return self.frameNumber

    @property
    def background(self):
        return self.bkg

    @property
    def roi_mask(self):
        return self.mask

    @property
    def size(self):
        return self.get_size()

    @property
    def bbox(self):
        return self.get_bbox()

    @property
    def bounds(self):
        return self.get_bounds()
