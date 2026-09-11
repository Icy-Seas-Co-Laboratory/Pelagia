"""Worker entry point for reproducible export bundles.

Product extraction deliberately lives in ``services.exports.products`` so the
worker only owns job lifecycle/progress plumbing.
"""

from __future__ import annotations

from typing import Any

from ...domain import PipelineStage
from ...services.context import AppContext
from ..progress import JobProgressReporter


def handle(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    if context.repository is None:
        raise RuntimeError("Export bundles require PostgreSQL.")
    payload = dict(job.get("payload") or {})
    export_id = str(payload.get("export_id") or "")
    if not export_id:
        raise ValueError("Export bundle job is missing export_id.")
    artifact = context.repository.get_export_artifact(export_id, project_id=str(job["project_id"]))
    if artifact is None:
        raise ValueError("Export artifact was not found in the selected project.")
    reporter = JobProgressReporter(
        job, context, stage=PipelineStage.EXPORT_BUNDLE.value, unit="input_record", total=0,
        emit_every=1_000,
    )
    reporter.start("Preparing export bundle")
    project_id = str(job["project_id"])
    attempt_number = max(1, int(job.get("attempt_count") or 1))
    begin_attempt = getattr(context.repository, "begin_export_attempt", None)
    if callable(begin_attempt):
        begin_attempt(export_id, project_id=project_id, job_id=str(job.get("id") or ""), attempt_number=attempt_number)
    else:
        context.repository.update_export_artifact(export_id, project_id=project_id, status="working")

    def product_event(product: str, status: str, detail: dict[str, Any]) -> None:
        update = getattr(context.repository, "update_export_attempt_product", None)
        if callable(update):
            update(export_id, project_id=project_id, attempt_number=attempt_number, product=product, status=status, detail=detail)
    try:
        # Submission only persists a small request and queues this job.  The
        # potentially large selection/fingerprint pass belongs to the worker.
        from ...services.exports.snapshot import prepare_export_snapshot
        def snapshot_progress(completed: int, message: str) -> None:
            reporter.update(
                completed, current={"phase": "freezing_inputs", "estimated": False},
                message=message,
            )

        snapshot, estimate = prepare_export_snapshot(
            context.repository, project_id=project_id, request=dict(artifact.get("request") or {}),
            config=None, progress_callback=snapshot_progress,
        )
        artifact = context.repository.update_export_artifact(
            export_id, project_id=project_id, input_snapshot=snapshot, estimate=estimate,
            progress={"phase": "snapshot_complete", "estimate": estimate},
        ) or artifact
        from ...services.exports.products import estimate_export_work_units
        reporter = JobProgressReporter(
            job, context, stage=PipelineStage.EXPORT_BUNDLE.value, unit="estimated_work_unit",
            total=estimate_export_work_units(artifact), emit_every=1_000,
        )
        reporter.start("Export inputs frozen; writing bundle")
        # Kept as an import here to make worker registration cheap and to allow
        # product modules to depend on optional image codecs only at execution.
        from ...services.exports.products import build_export_bundle

        result = build_export_bundle(job=job, context=context, artifact=artifact, reporter=reporter, on_product_event=product_event)
        context.repository.update_export_artifact(
            export_id, project_id=str(job["project_id"]), status="succeeded",
            manifest=result["manifest"], artifact_path=result["path"],
            artifact_sha256=result["sha256"], size_bytes=result["size_bytes"],
            snapshot_at=result.get("snapshot_at"),
        )
        finish_attempt = getattr(context.repository, "finish_export_attempt", None)
        if callable(finish_attempt):
            finish_attempt(export_id, project_id=project_id, attempt_number=attempt_number, status="succeeded")
        reporter.finish(message="Export bundle published")
        return {"operation": "export_bundle", "export_id": export_id, **result}
    except Exception as exc:
        retry_pending = attempt_number < max(1, int(job.get("max_attempts") or 1))
        finish_attempt = getattr(context.repository, "finish_export_attempt", None)
        if callable(finish_attempt):
            finish_attempt(export_id, project_id=project_id, attempt_number=attempt_number,
                           status="failed", error=str(exc), retry_pending=retry_pending)
        else:
            context.repository.update_export_artifact(
                export_id, project_id=project_id, status="queued" if retry_pending else "failed",
                failure_message="" if retry_pending else str(exc)[:4000],
            )
        raise
