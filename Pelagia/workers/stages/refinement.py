from __future__ import annotations

from typing import Any

from ...services.context import AppContext


def handle(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    from ..handlers import roi_refinement_handler

    return roi_refinement_handler(job, context)
