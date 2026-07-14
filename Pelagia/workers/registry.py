"""Declarative worker-stage registration."""

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
        self._handlers[stage] = handler

    def handle(self, job: dict[str, Any], context: AppContext) -> dict[str, Any]:
        stage = PipelineStage(job["stage"])
        if stage not in self._handlers:
            raise KeyError(f"No worker handler registered for stage {stage.value!r}.")
        return self._handlers[stage](job, context)


def default_stage_handlers() -> tuple[tuple[PipelineStage, JobHandler], ...]:
    """Return the built-in stages in one inspectable registration table."""
    from .stages import background, extract, preprocess, refinement, segmentation

    return (
        (PipelineStage.EXTRACT_FRAMES, extract.handle),
        (PipelineStage.PREPROCESS_FRAMES, preprocess.handle),
        (PipelineStage.BACKGROUND_FRAMES, background.handle),
        (PipelineStage.SEGMENT, segmentation.handle),
        (PipelineStage.ROI_REFINEMENT, refinement.handle),
    )


def default_handler_registry() -> HandlerRegistry:
    registry = HandlerRegistry()
    for stage, handler in default_stage_handlers():
        registry.register(stage, handler)
    return registry
