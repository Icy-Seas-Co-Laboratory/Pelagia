from __future__ import annotations

import csv
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

try:
    from fastapi import APIRouter, HTTPException, Query, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore


if APIRouter is not None:
    from ..auth import require_project_write, scoped_project_id
    from ...domain import PipelineStage
    from ...services.job_commands import TelemetryImportCommand
    from ...services.pipeline import PipelineService
    from ...services.telemetry import (
        TelemetryResolver,
        infer_timestamp_format,
        iter_parsed_csv_rows,
        normalize_observed_at,
    )
    from ...services.telemetry_units import DEFAULT_TELEMETRY_UNIT_REGISTRY
    from ._common import as_response, get_context, get_repository

    router = APIRouter(tags=["telemetry"])

    class TelemetryColumnRequest(BaseModel):
        column: str
        stream_key: str
        sensor_key: str
        parameter_key: str
        native_unit: str
        canonical_unit: str
        display_name: str | None = None
        standard_name: str | None = None
        sensor_display_name: str | None = None
        manufacturer: str | None = None
        model: str | None = None
        serial_number: str | None = None
        qc_column: str | None = None
        scale: float = 1.0
        offset: float = 0.0
        interpolation: Literal["linear", "nearest", "previous", "none"] = "none"
        max_gap_seconds: float | None = None
        sampling_rate_hz: float | None = None
        priority: int = 100
        is_default: bool = False
        metadata: dict[str, Any] = Field(default_factory=dict)
        sensor_metadata: dict[str, Any] = Field(default_factory=dict)
        parameter_metadata: dict[str, Any] = Field(default_factory=dict)

    class TelemetryImportRequest(BaseModel):
        path: str
        timestamp_column: str
        streams: list[TelemetryColumnRequest]
        timestamp_format: str = "iso8601"
        source_timezone: str = "UTC"
        delimiter: str = ","
        parser_name: str = "pelagia.delimited"
        parser_version: str = "1"
        collections: list[str] | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class TelemetryAnalyzeRequest(BaseModel):
        path: str
        timestamp_column: str | None = None
        timestamp_format: str = "auto"
        source_timezone: str = "UTC"
        delimiter: str = ","
        sample_limit: int = Field(default=240, ge=20, le=2_000)

    def _analyze_csv(path: Path, body: TelemetryAnalyzeRequest) -> dict[str, Any]:
        if not path.is_file():
            raise ValueError("Telemetry source path must be a file.")
        if len(body.delimiter) != 1:
            raise ValueError("delimiter must be one character.")
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source, delimiter=body.delimiter)
            columns = list(reader.fieldnames or [])
            first_row = next(reader, None)
        if not columns:
            raise ValueError("Telemetry CSV has no header row.")
        timestamp_column = body.timestamp_column or next(
            (column for column in columns if column.casefold() in {"time", "timestamp", "datetime", "date"}),
            columns[0],
        )
        if timestamp_column not in columns:
            raise ValueError(f"Timestamp column {timestamp_column!r} was not found.")
        format_name = infer_timestamp_format(
            (first_row or {}).get(timestamp_column), body.timestamp_format,
        )
        preview_rows: list[dict[str, Any]] = []
        stats: dict[str, dict[str, Any]] = {
            column: {"column": column, "missing": 0, "numeric": 0, "invalid_numeric": 0, "sample": []}
            for column in columns
        }
        row_count = 0
        invalid_count = 0
        invalid_timestamps: list[dict[str, Any]] = []
        previous: datetime | None = None
        range_start: datetime | None = None
        range_end: datetime | None = None
        duplicate_count = 0
        non_monotonic_count = 0
        intervals: list[float] = []
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            for row_number, row, observed_at, parse_error in iter_parsed_csv_rows(
                source,
                timestamp_column=timestamp_column,
                timestamp_format=format_name,
                source_timezone=body.source_timezone,
                delimiter=body.delimiter,
                strict=False,
            ):
                row_count += 1
                if parse_error is not None or observed_at is None:
                    invalid_count += 1
                    if len(invalid_timestamps) < 20:
                        invalid_timestamps.append({
                            "row": row_number,
                            "value": row.get(timestamp_column),
                            "message": str(parse_error),
                        })
                else:
                    if range_start is None or observed_at < range_start:
                        range_start = observed_at
                    if range_end is None or observed_at > range_end:
                        range_end = observed_at
                    if previous is not None:
                        delta = (observed_at - previous).total_seconds()
                        if observed_at == previous:
                            duplicate_count += 1
                        elif observed_at < previous:
                            non_monotonic_count += 1
                        elif len(intervals) < 10_000:
                            intervals.append(delta)
                    previous = observed_at
                for column in columns:
                    value = str(row.get(column) or "").strip()
                    entry = stats[column]
                    if not value:
                        entry["missing"] += 1
                    else:
                        try:
                            number = float(value)
                            if math.isfinite(number):
                                entry["numeric"] += 1
                                if len(entry["sample"]) < 3:
                                    entry["sample"].append(number)
                            else:
                                entry["invalid_numeric"] += 1
                        except ValueError:
                            entry["invalid_numeric"] += 1
                if len(preview_rows) < body.sample_limit:
                    preview_rows.append({
                        "row": row_number,
                        "timestamp": observed_at.isoformat() if observed_at else None,
                        "values": {column: row.get(column) for column in columns if column != timestamp_column},
                    })
        return {
                "path": str(path),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
                "columns": columns,
                "timestamp_column": timestamp_column,
                "timestamp_format": format_name,
                "source_timezone": body.source_timezone,
                "row_count": row_count,
                "time_range": {
                    "start": range_start.isoformat() if range_start else None,
                    "end": range_end.isoformat() if range_end else None,
                    "duration_seconds": (range_end - range_start).total_seconds() if range_start and range_end else None,
                },
                "sampling": {
                    "median_interval_seconds": sorted(intervals)[len(intervals) // 2] if intervals else None,
                    "min_interval_seconds": min(intervals) if intervals else None,
                    "max_interval_seconds": max(intervals) if intervals else None,
                },
                "timestamp_diagnostics": {
                    "invalid_count": invalid_count,
                    "invalid_examples": invalid_timestamps,
                    "duplicate_count": duplicate_count,
                    "non_monotonic_count": non_monotonic_count,
                    "valid": bool(range_start) and invalid_count == 0 and duplicate_count == 0 and non_monotonic_count == 0,
                },
                "column_stats": list(stats.values()),
                "preview_rows": preview_rows,
            }

    class TimelineEventTypeRequest(BaseModel):
        event_type_key: str
        display_name: str | None = None
        description: str | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class TimelineEventRequest(BaseModel):
        event_type_id: str
        source_id: str | None = None
        start_at: datetime
        end_at: datetime | None = None
        value: str | None = None
        metadata: dict[str, Any] = Field(default_factory=dict)

    class TimelineEventUpdateRequest(BaseModel):
        event_type_id: str | None = None
        source_id: str | None = None
        start_at: datetime | None = None
        end_at: datetime | None = None
        value: str | None = None
        metadata: dict[str, Any] | None = None

    def _normalize_event_timestamp(value: datetime | None, field_name: str) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None or value.utcoffset() is None:
            raise HTTPException(status_code=422, detail=f"{field_name} must include a UTC offset.")
        try:
            return normalize_observed_at(value)
        except (OverflowError, ValueError) as exc:
            raise HTTPException(status_code=422, detail=f"Invalid {field_name}: {exc}") from exc

    def _validate_event_interval(start_at: datetime, end_at: datetime | None) -> None:
        if end_at is not None and end_at < start_at:
            raise HTTPException(status_code=422, detail="end_at must not precede start_at.")

    @router.get("/telemetry/parameters")
    def list_parameters(request: Request) -> dict[str, Any]:
        rows = get_repository(request).telemetry.list_telemetry_parameters(
            project_id=scoped_project_id(request)
        )
        return {"parameters": as_response(rows)}

    @router.get("/telemetry/catalog")
    def telemetry_catalog(request: Request) -> dict[str, Any]:
        rows = get_repository(request).telemetry.list_telemetry_parameters(
            project_id=scoped_project_id(request)
        )
        return {
            "unit_registry": {
                "name": "pelagia.telemetry_units",
                "version": DEFAULT_TELEMETRY_UNIT_REGISTRY.version,
                "units": DEFAULT_TELEMETRY_UNIT_REGISTRY.catalog(),
            },
            "parameters": as_response(rows),
            "sensors": as_response(get_repository(request).telemetry.list_telemetry_sensors(
                project_id=scoped_project_id(request)
            )),
            "interpolation_methods": ["none", "linear", "nearest", "previous"],
        }

    @router.post("/telemetry/analyze")
    def analyze_telemetry(request: Request, body: TelemetryAnalyzeRequest) -> dict[str, Any]:
        from .ingestion import _resolve_allowed_import_path
        try:
            path = _resolve_allowed_import_path(request, body.path)
            return as_response(_analyze_csv(path, body))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except OSError as exc:
            raise HTTPException(status_code=422, detail=f"Unable to read telemetry source: {exc}") from exc

    @router.get("/telemetry/sensors")
    def list_sensors(request: Request) -> dict[str, Any]:
        rows = get_repository(request).telemetry.list_telemetry_sensors(
            project_id=scoped_project_id(request)
        )
        return {"sensors": as_response(rows)}

    @router.get("/runs/{run_id}/telemetry/sources")
    def list_sources(request: Request, run_id: str) -> dict[str, Any]:
        rows = get_repository(request).telemetry.list_telemetry_sources(
            project_id=scoped_project_id(request), run_id=run_id,
        )
        return {"sources": as_response(rows)}

    @router.get("/runs/{run_id}/telemetry/streams")
    def list_streams(request: Request, run_id: str) -> dict[str, Any]:
        rows = get_repository(request).telemetry.list_telemetry_streams(
            project_id=scoped_project_id(request), run_id=run_id,
        )
        return {"streams": as_response(rows)}

    @router.post("/runs/{run_id}/telemetry/import", status_code=202)
    def import_telemetry(request: Request, run_id: str, body: TelemetryImportRequest) -> dict[str, Any]:
        auth = require_project_write(request)
        from .ingestion import _resolve_allowed_import_path

        try:
            source_path = _resolve_allowed_import_path(request, body.path)
            if not source_path.is_file():
                raise ValueError("Telemetry source path must be a file.")
            if get_repository(request).get_run(run_id, project_id=str(auth.project_id)) is None:
                raise KeyError(f"Run {run_id!r} was not found in the active project.")
            for stream in body.streams:
                DEFAULT_TELEMETRY_UNIT_REGISTRY.validate_affine_conversion(
                    stream.native_unit,
                    stream.canonical_unit,
                    scale=stream.scale,
                    offset=stream.offset,
                )
            payload = TelemetryImportCommand.from_payload(
                {
                    **body.model_dump(),
                    "path": str(source_path),
                    "streams": [item.model_dump() for item in body.streams],
                }
            ).to_payload()
            job = PipelineService(get_context(request)).queue(
                PipelineStage.TELEMETRY_IMPORT,
                project_id=str(auth.project_id),
                run_id=run_id,
                payload=payload,
                summary=f"telemetry_import queued for {source_path.name}",
                submitted_by_user_id=auth.user_id,
                submitted_by_username=auth.username,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"job": as_response(job)}

    @router.get("/runs/{run_id}/telemetry/lookup")
    def lookup_telemetry(
        request: Request, run_id: str, observed_at: datetime,
        parameters: list[str] | None = Query(default=None),
    ) -> dict[str, Any]:
        target = _normalize_event_timestamp(observed_at, "observed_at")
        assert target is not None
        return as_response(
            TelemetryResolver(get_repository(request)).at(
                project_id=scoped_project_id(request), run_id=run_id,
                observed_at=target, parameters=parameters,
            )
        )

    @router.get("/runs/{run_id}/telemetry/observations")
    def list_observations(
        request: Request,
        run_id: str,
        stream_id: str,
        start_at: datetime | None = Query(default=None),
        end_at: datetime | None = Query(default=None),
        limit: int = Query(default=10_000, ge=1, le=100_000),
        offset: int = Query(default=0, ge=0),
    ) -> dict[str, Any]:
        range_start = _normalize_event_timestamp(start_at, "start_at")
        range_end = _normalize_event_timestamp(end_at, "end_at")
        if range_start is not None and range_end is not None:
            _validate_event_interval(range_start, range_end)
        try:
            rows = get_repository(request).telemetry.list_telemetry_observations(
                project_id=scoped_project_id(request), run_id=run_id, stream_id=stream_id,
                start_at=range_start, end_at=range_end, limit=limit, offset=offset,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {
            "stream_id": stream_id,
            "observations": as_response(rows),
            "limit": limit,
            "offset": offset,
            "next_offset": offset + limit if len(rows) == limit else None,
        }

    @router.get("/timeline-event-types")
    def list_event_types(request: Request) -> dict[str, Any]:
        rows = get_repository(request).telemetry.list_timeline_event_types(
            project_id=scoped_project_id(request)
        )
        return {"event_types": as_response(rows)}

    @router.post("/timeline-event-types")
    def create_event_type(request: Request, body: TimelineEventTypeRequest) -> dict[str, Any]:
        auth = require_project_write(request)
        if not body.event_type_key.strip():
            raise HTTPException(status_code=422, detail="event_type_key must not be blank.")
        try:
            row = get_repository(request).telemetry.create_timeline_event_type(
                project_id=str(auth.project_id), **body.model_dump(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"event_type": as_response(row)}

    @router.post("/runs/{run_id}/events")
    def create_event(request: Request, run_id: str, body: TimelineEventRequest) -> dict[str, Any]:
        auth = require_project_write(request)
        start_at = _normalize_event_timestamp(body.start_at, "start_at")
        assert start_at is not None
        end_at = _normalize_event_timestamp(body.end_at, "end_at")
        _validate_event_interval(start_at, end_at)
        try:
            row = get_repository(request).telemetry.create_timeline_event(
                project_id=str(auth.project_id), run_id=run_id,
                event_type_id=body.event_type_id, start_at=start_at, end_at=end_at,
                source_id=body.source_id, value=body.value, created_by=auth.user_id,
                metadata=body.metadata,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        return {"event": as_response(row)}

    @router.get("/runs/{run_id}/events")
    def list_events(
        request: Request,
        run_id: str,
        observed_at: datetime | None = Query(default=None),
        start_at: datetime | None = Query(default=None),
        end_at: datetime | None = Query(default=None),
        event_type_id: str | None = Query(default=None),
        source_id: str | None = Query(default=None),
    ) -> dict[str, Any]:
        target = _normalize_event_timestamp(observed_at, "observed_at")
        range_start = _normalize_event_timestamp(start_at, "start_at")
        range_end = _normalize_event_timestamp(end_at, "end_at")
        if target is not None and (range_start is not None or range_end is not None):
            raise HTTPException(status_code=422, detail="observed_at cannot be combined with start_at or end_at.")
        if range_start is not None and range_end is not None:
            _validate_event_interval(range_start, range_end)
        project_id = scoped_project_id(request)
        if target is not None:
            rows = get_repository(request).telemetry.list_timeline_events_at(
                project_id=project_id, run_id=run_id, observed_at=target,
            )
            return {"events": as_response(rows), "observed_at": as_response(target)}
        rows = get_repository(request).telemetry.list_timeline_events(
            project_id=project_id,
            run_id=run_id,
            start_at=range_start,
            end_at=range_end,
            event_type_id=event_type_id,
            source_id=source_id,
        )
        return {"events": as_response(rows)}

    @router.get("/runs/{run_id}/events/{event_id}")
    def get_event(request: Request, run_id: str, event_id: str) -> dict[str, Any]:
        row = get_repository(request).telemetry.get_timeline_event(
            project_id=scoped_project_id(request), run_id=run_id, event_id=event_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"Event {event_id!r} was not found.")
        return {"event": as_response(row)}

    @router.patch("/runs/{run_id}/events/{event_id}")
    def update_event(
        request: Request, run_id: str, event_id: str, body: TimelineEventUpdateRequest,
    ) -> dict[str, Any]:
        auth = require_project_write(request)
        changes = body.model_dump(exclude_unset=True)
        if not changes:
            raise HTTPException(status_code=422, detail="Provide at least one event field to update.")
        repository = get_repository(request).telemetry
        current = repository.get_timeline_event(
            project_id=str(auth.project_id), run_id=run_id, event_id=event_id,
        )
        if current is None:
            raise HTTPException(status_code=404, detail=f"Event {event_id!r} was not found.")
        if "start_at" in changes:
            changes["start_at"] = _normalize_event_timestamp(changes["start_at"], "start_at")
        if "end_at" in changes:
            changes["end_at"] = _normalize_event_timestamp(changes["end_at"], "end_at")
        effective_start = changes.get("start_at", current["start_at"])
        effective_end = changes.get("end_at", current["end_at"])
        assert effective_start is not None
        _validate_event_interval(effective_start, effective_end)
        try:
            row = repository.update_timeline_event(
                project_id=str(auth.project_id), run_id=run_id, event_id=event_id, updates=changes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        if row is None:
            raise HTTPException(status_code=404, detail=f"Event {event_id!r} was not found.")
        return {"event": as_response(row)}

    @router.delete("/runs/{run_id}/events/{event_id}")
    def delete_event(request: Request, run_id: str, event_id: str) -> dict[str, Any]:
        auth = require_project_write(request)
        row = get_repository(request).telemetry.delete_timeline_event(
            project_id=str(auth.project_id), run_id=run_id, event_id=event_id,
        )
        if row is None:
            raise HTTPException(status_code=404, detail=f"Event {event_id!r} was not found.")
        return {"status": "deleted", "event": as_response(row)}
else:
    router = None
