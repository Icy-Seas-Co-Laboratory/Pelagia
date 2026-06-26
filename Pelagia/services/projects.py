from __future__ import annotations

from typing import Any

from .context import AppContext


def initialize_project_kvstore(context: AppContext, project: dict[str, Any]) -> dict[str, Any]:
    """Ensure the physical KVStore for a newly created project exists."""
    project_id = project.get("id")
    if project_id is None:
        return {"initialized": False, "root_path": None}
    store = context.kvstore_for_project(str(project_id), initialize=True)
    if store is None:
        return {"initialized": False, "root_path": None}
    return {
        "initialized": bool(store.initialized),
        "root_path": str(store.root_path),
    }
