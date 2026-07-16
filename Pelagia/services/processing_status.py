from __future__ import annotations

from typing import Any


class ProcessingStatusService:
    """Build frame-status views while keeping snapshot writes project-scoped."""

    def __init__(self, repository):
        self.frames = getattr(repository, "frames", repository)

    @staticmethod
    def _has_filters(filters: dict[str, Any]) -> bool:
        return any(value not in (None, [], "") for value in filters.values())

    @staticmethod
    def _snapshot_is_fresh(snapshot: dict[str, Any] | None) -> bool:
        if not snapshot or not snapshot.get("summary") or not snapshot.get("generated_at"):
            return False
        return snapshot["generated_at"] >= snapshot["updated_at"]

    @staticmethod
    def _empty_snapshot(project_id: str) -> dict[str, Any]:
        return {
            "project_id": project_id,
            "session_id": None,
            "status_version": 0,
            "summary": {},
        }

    def summary(self, *, project_id: str, filters: dict[str, Any]) -> dict[str, Any]:
        snapshot = self.frames.get_processing_status_snapshot(project_id=project_id)
        if not self._has_filters(filters) and self._snapshot_is_fresh(snapshot):
            return {"summary": snapshot["summary"], "snapshot": snapshot}

        summary = self.frames.get_frame_status_summary(project_id=project_id, **filters)
        if not self._has_filters(filters):
            snapshot = self.frames.get_or_create_processing_status_snapshot(
                project_id=project_id,
                summary=summary,
            )
        return {
            "summary": summary,
            "snapshot": snapshot or self._empty_snapshot(project_id),
        }

    def facets(self, *, project_id: str, filters: dict[str, Any]) -> dict[str, Any]:
        result = self.frames.get_frame_status_facets(project_id=project_id, **filters)
        if not self._has_filters(filters):
            snapshot = self.frames.get_or_create_processing_status_snapshot(
                project_id=project_id,
                summary=result["summary"],
            )
        else:
            snapshot = self.frames.get_processing_status_snapshot(project_id=project_id)
        return {**result, "snapshot": snapshot or self._empty_snapshot(project_id)}
