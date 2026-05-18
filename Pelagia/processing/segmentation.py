from __future__ import annotations

from ..domain import DetectionRecord
from .frames import Frame


def segment_frame(frame: Frame) -> list[DetectionRecord]:
    """Template segmentation routine for converting one frame into detections."""
    raise NotImplementedError("Implement project-specific segmentation here.")
