import os
from dataclasses import dataclass, field

import cv2


@dataclass
class Frame:
    """
    Container for one frame or tiled frame plus source/output metadata.

    The constructor keeps the historical camelCase field names so existing
    segmentation calls remain compatible.
    """
    sourcePath: str
    destPath: str
    filename: str
    frameNumber: int
    data: object = None
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

    def __post_init__(self):
        self.sourcePath = os.fspath(self.sourcePath)
        self.destPath = os.fspath(self.destPath)
        self.filename = os.fspath(self.filename)

    def get_frame_number(self):
        return self.frameNumber

    def get_source_path(self):
        return self.sourcePath

    def get_dest_path(self):
        return self.destPath

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

    def has_background(self):
        return self.bkg is not None

    def is_loaded(self):
        return self.data is not None

    def read(self):
        if self.data is not None:
            return self.data

        source_file_path = self.get_source_file_path()
        image = cv2.imread(source_file_path, self.imageReadFlag)
        if self.cacheRead:
            self.data = image
        return image

    def update(self, newframe):
        self.data = newframe

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
    def dest_path(self):
        return self.destPath

    @property
    def frame_number(self):
        return self.frameNumber

    @property
    def background(self):
        return self.bkg
