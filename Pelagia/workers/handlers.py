from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..domain import PipelineStage
from ..services.context import AppContext


JobHandler = Callable[[dict[str, Any], AppContext], dict[str, Any]]


class HandlerRegistry:
    """Maps pipeline stages to processing functions."""

    def __init__(self) -> None:
        self._handlers: dict[PipelineStage, JobHandler] = {}

    def register(self, stage: PipelineStage, handler: JobHandler) -> None:
        """Register a callable for a pipeline stage."""
        self._handlers[stage] = handler

    def handle(self, job: dict[str, Any], context: AppContext) -> dict[str, Any]:
        """Dispatch a leased job to its registered handler."""
        stage = PipelineStage(job["stage"])
        if stage not in self._handlers:
            raise KeyError(f"No worker handler registered for stage {stage.value!r}.")
        return self._handlers[stage](job, context)
