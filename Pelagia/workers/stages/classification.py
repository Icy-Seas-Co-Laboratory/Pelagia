from __future__ import annotations

from typing import Any

from ...services.context import AppContext


def handle(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    from ..handlers import classification_handler

    return classification_handler(job, context)
