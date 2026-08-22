from __future__ import annotations

from pathlib import Path
from typing import Any

from ...domain import PipelineStage
from ...services.context import AppContext
from ...services.telemetry import TelemetryColumn, TelemetryCsvSpec, TelemetryIngestionService
from ..progress import JobProgressReporter


def _allowed_source_path(context: AppContext, source_path: str) -> Path:
    resolved = Path(source_path).expanduser().resolve()
    browser = context.config.file_browser
    roots = [browser.root_path_import_dir, *browser.allowed_root_paths]
    resolved_roots = [Path(root).expanduser().resolve() for root in roots]
    if resolved_roots:
        for root in resolved_roots:
            try:
                resolved.relative_to(root)
                return resolved
            except ValueError:
                continue
        raise ValueError("Telemetry source path is outside the configured import roots.")
    return resolved


def handle(job: dict[str, Any], context: AppContext) -> dict[str, Any]:
    if context.repository is None:
        raise RuntimeError("Telemetry import requires PostgreSQL")
    payload = dict(job.get("payload") or {})
    path = _allowed_source_path(context, str(payload.get("path") or ""))
    streams = [TelemetryColumn(**dict(item)) for item in payload.get("streams") or ()]
    spec = TelemetryCsvSpec(
        timestamp_column=str(payload.get("timestamp_column") or ""),
        streams=streams,
        timestamp_format=str(payload.get("timestamp_format") or "iso8601"),
        source_timezone=str(payload.get("source_timezone") or "UTC"),
        delimiter=str(payload.get("delimiter") or ","),
        parser_name=str(payload.get("parser_name") or "pelagia.delimited"),
        parser_version=str(payload.get("parser_version") or "1"),
        metadata=dict(payload.get("metadata") or {}),
    )
    reporter = JobProgressReporter(
        job, context, stage=PipelineStage.TELEMETRY_IMPORT.value,
        unit="source", total=1, emit_every=1,
    )
    reporter.start(f"Importing telemetry source {path.name}")
    result = TelemetryIngestionService(
        context.repository, context.kvstore_for_project(str(job["project_id"]))
    ).import_csv(
        path,
        project_id=str(job["project_id"]),
        run_id=str(job["run_id"]),
        spec=spec,
        collections=payload.get("collections"),
    )
    reporter.finish(message=f"Imported telemetry source {path.name}")
    return {"operation": "telemetry_import", **result}
