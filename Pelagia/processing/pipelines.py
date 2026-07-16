"""Shared result and callable protocols for processing components."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(slots=True)
class ProcessingResult:
    """Standard return shape for processing routines."""

    outputs: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)


class ProcessingRoutine(Protocol):
    """Callable protocol for data-in/data-out processing components."""

    def __call__(self, payload: dict[str, Any]) -> ProcessingResult:
        ...
