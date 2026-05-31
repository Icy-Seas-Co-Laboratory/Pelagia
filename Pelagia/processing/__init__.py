"""Data processing routines that transform inputs into derived outputs."""

from .frame_model import FrameData
from .pipelines import ProcessingResult, ProcessingRoutine

__all__ = ["FrameData", "ProcessingResult", "ProcessingRoutine"]
