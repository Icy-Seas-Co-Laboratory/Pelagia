from __future__ import annotations

from typing import Any

from ...services.context import AppContext


def handle(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    from ..handlers import extract_frames_handler

    return extract_frames_handler(job, context)
