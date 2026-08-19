from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain import PipelineStage
from ...services.context import AppContext
from ...services.registry_transfer import export_sqlite_workspace, load_sqlite_workspace
from ...services.registry_generation import generate_and_load_registry_dataset
from ..progress import JobProgressReporter


def _payload(job: dict[str, Any]) -> dict[str, Any]:
    value = job.get("payload") or {}
    if not isinstance(value, dict):
        raise ValueError("Registry transfer job payload must be an object")
    return value


def handle_load(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    if context.repository is None:
        raise RuntimeError("Registry load requires PostgreSQL")
    payload = _payload(job)
    source = Path(str(payload["source_path"])).expanduser().resolve()
    reporter = JobProgressReporter(job, context, stage=PipelineStage.REGISTRY_LOAD.value, unit="phase", total=8, emit_every=1)
    reporter.start(f"Loading Registry dataset {source.name}")
    result = load_sqlite_workspace(
        context.repository, source, project_id=str(job["project_id"]),
        owner_username=str(payload["owner_username"]),
        progress_callback=lambda completed, phase: reporter.update(completed, current={"phase": phase}, message=phase),
    )
    reporter.finish(message=f"Loaded Registry dataset {source.name}")
    return {"operation": "registry_load", **result}


def handle_export(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    if context.repository is None:
        raise RuntimeError("Registry export requires PostgreSQL")
    payload = _payload(job)
    destination = Path(str(payload["destination_path"])).expanduser().resolve()
    reporter = JobProgressReporter(job, context, stage=PipelineStage.REGISTRY_EXPORT.value, unit="phase", total=8, emit_every=1)
    reporter.start(f"Exporting Registry dataset to {destination.name}")
    result = export_sqlite_workspace(
        context.repository, str(payload["workspace_id"]), destination,
        project_id=str(job["project_id"]), owner_username=str(payload["owner_username"]),
        replace_source=bool(payload.get("replace_source")),
        operation_id=str(job["id"]),
        progress_callback=lambda completed, phase: reporter.update(completed, current={"phase": phase}, message=phase),
    )
    reporter.finish(message=f"Exported Registry dataset to {destination.name}")
    return {"operation": "registry_export", **result}


def handle_generate(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    if context.repository is None:
        raise RuntimeError("Registry dataset generation requires PostgreSQL")
    payload = _payload(job)
    destination = Path(str(payload["destination_path"])).expanduser().resolve()
    selected_count = max(1, int(payload.get("selected_count") or 1))
    reporter = JobProgressReporter(
        job, context, stage=PipelineStage.REGISTRY_GENERATE.value,
        unit="roi", total=selected_count, emit_every=1,
    )
    reporter.start(f"Generating Registry dataset {destination.name}")
    result = generate_and_load_registry_dataset(
        context.repository, destination,
        project_id=str(job["project_id"]), owner_username=str(payload["owner_username"]),
        name=str(payload["name"]), selection=dict(payload.get("selection") or {}),
        subsample_ratio=int(payload.get("subsample_ratio") or 1),
        dataset_id=str(payload["dataset_id"]), revision_id=str(payload["revision_id"]),
        progress_callback=lambda completed, total, message: reporter.update(
            completed, current={"phase": "write_dataset"}, message=message
        ),
    )
    reporter.finish(message=f"Generated and loaded Registry dataset {destination.name}")
    return {"operation": "registry_generate", **result}
