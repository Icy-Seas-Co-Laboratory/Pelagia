from __future__ import annotations

from typing import Any

from ...services.context import AppContext


def handle(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    from ...services.storage_reencode import reencode_project_storage

    return reencode_project_storage(job, context)
