"""Scoped views over the transaction-owning PostgreSQL repository.

The repository still owns connections and transaction boundaries.  These views
make each use-case dependency explicit while allowing the SQL implementation to
move out of the legacy facade incrementally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .postgres import PostgresRepository


class RepositoryScope:
    """Expose only the repository operations owned by one application area."""

    operations: frozenset[str] = frozenset()

    def __init__(self, repository: "PostgresRepository") -> None:
        self._repository = repository

    def __getattr__(self, name: str) -> Any:
        if name not in self.operations:
            raise AttributeError(
                f"{type(self).__name__} does not provide the {name!r} operation."
            )
        return getattr(self._repository, name)


class IdentityRepository(RepositoryScope):
    operations = frozenset(
        {
            "create_user", "get_user", "get_user_by_username", "list_users",
            "deactivate_user", "reset_user_password", "delete_user",
            "verify_user_password", "create_project", "list_projects",
            "list_user_projects", "get_project", "get_project_by_key",
            "update_project", "deactivate_project", "add_project_member",
            "get_project_membership", "create_session", "get_session",
            "revoke_session", "revoke_user_sessions", "ensure_default_project",
        }
    )


class CatalogRepository(RepositoryScope):
    operations = frozenset(
        {
            "list_runs", "get_run", "register_planned_run", "list_assets",
            "list_collections", "get_asset", "count_frames", "register_model",
            "list_models", "get_model", "get_model_by_key",
            "replace_classification_results", "cancel_run", "reconcile_run",
        }
    )


class FrameRepository(RepositoryScope):
    operations = frozenset(
        {
            "replace_frames", "list_frames", "get_frame", "get_frame_by_asset_index",
            "list_frame_records", "get_frame_record", "get_frame_records", "create_live_frame_copy",
            "list_live_frame_copies", "count_frame_payload_references",
            "delete_live_frame_copy", "update_frame_preprocessed_payload",
            "update_frame_preprocessed_payloads",
            "update_frame_background_payloads", "update_frame_background_payload_assignments",
            "upsert_refined_detections",
            "replace_detections", "replace_frame_detections", "list_detections",
            "get_detection", "get_refined_detection_for_candidate",
            "get_refined_detection", "list_detection_records",
            "list_asset_detection_stats", "list_asset_processing_state",
            "list_frame_processing_state", "ensure_frame_status_rows",
            "upsert_frame_stage_status", "refresh_frame_status_counts",
            "rebuild_frame_status", "touch_processing_status_snapshot",
            "list_frame_status", "list_frame_status_ids", "get_frame_status_summary",
            "get_frame_status_facets", "get_processing_status_snapshot",
            "get_or_create_processing_status_snapshot",
        }
    )


class JobRepository(RepositoryScope):
    operations = frozenset(
        {
            "list_jobs", "summarize_jobs", "get_job", "create_job",
            "update_job_payload", "update_job_progress", "append_job_event",
            "append_log", "list_job_events", "list_logs", "set_job_priority",
            "pause_job", "finalize_paused_job", "resume_job", "get_status_summary",
            "list_worker_sessions", "touch_worker", "get_worker_session",
            "request_worker_shutdown", "heartbeat", "requeue_expired_jobs",
            "claim_jobs", "complete_job", "record_failure", "fail_job", "retry_job",
            "cancel_jobs", "delete_jobs",
        }
    )
