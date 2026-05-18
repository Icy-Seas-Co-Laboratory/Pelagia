"""Data processing routines that transform inputs into derived outputs."""

from .frames import Frame
from .pipelines import ProcessingResult, ProcessingRoutine

__all__ = ["Frame", "ProcessingResult", "ProcessingRoutine"]
