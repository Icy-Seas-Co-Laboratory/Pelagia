"""Shared worker failure and progress boundaries."""

from __future__ import annotations

from typing import Any

from ..services.context import AppContext


def mark_job_frame_stage_failed(job: dict[str, Any], context: AppContext) -> None:
    """Mark frame-stage status after an unhandled stage exception."""
    # Retain the existing status implementation while stage code is moved out.
    from .handlers import mark_job_frame_stage_failed as legacy_marker

    legacy_marker(job, context)
