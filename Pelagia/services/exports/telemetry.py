"""Portable telemetry export product.

This module deliberately has no HTTP, job, or artifact-store knowledge.  The
export worker supplies a project-scoped repository, a destination underneath a
bundle root, and an optional project KVStore.  The result is a manifest-ready
description of the files written below ``products/telemetry``.

The JSON representation is the canonical interchange representation.  SQLite
is a convenient, typed tabular view of the same selected telemetry source.
XLSX is intentionally left to the shared tabular writer: the core package has
no spreadsheet dependency and must not fake an XLSX file with CSV content.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...utils.serialization import json_ready
from .xlsx import write_xlsx


TELEMETRY_PRODUCT_SCHEMA_VERSION = "1.0"
_FORMATS = frozenset({"json", "sqlite", "xlsx"})


def write_telemetry_bundle(
    repository: Any,
    *,
    project_id: str,
    selection: Mapping[str, Any],
    output_root: Path,
    file_format: str = "json",
    kvstore: Any | None = None,
    progress_callback: Callable[[Mapping[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Write a telemetry product and return manifest-ready metadata.

    ``selection`` accepts ``source_ids`` and/or ``run_ids``.  Selected source
    IDs are always constrained to ``project_id`` by the repository's telemetry
    list methods.  A source is emitted once even when it matches both filters.
    """
    normalized_format = str(file_format).lower()
    if normalized_format not in _FORMATS:
        raise ValueError(
            f"Unsupported telemetry export format {file_format!r}. "
            "Use 'json', 'sqlite', or 'xlsx'."
        )
    telemetry = getattr(repository, "telemetry", repository)
    source_ids = _selection_ids(selection, "source_ids")
    run_ids = _selection_ids(selection, "run_ids")
    sources = _selected_sources(
        telemetry, project_id=project_id, source_ids=source_ids, run_ids=run_ids,
    )
    product_root = Path(output_root) / "products" / "telemetry"
    product_root.mkdir(parents=True, exist_ok=True)

    all_parameters = telemetry.list_telemetry_parameters(project_id=project_id)
    all_sensors = telemetry.list_telemetry_sensors(project_id=project_id)
    selected_source_ids = {str(source["id"]) for source in sources}
    selected_run_ids = {str(source["run_id"]) for source in sources}
    streams_by_source: dict[str, list[dict[str, Any]]] = {}
    selected_streams: list[dict[str, Any]] = []
    for run_id in sorted(selected_run_ids):
        for stream in telemetry.list_telemetry_streams(project_id=project_id, run_id=run_id):
            if str(stream["source_id"]) in selected_source_ids:
                streams_by_source.setdefault(str(stream["source_id"]), []).append(stream)
                selected_streams.append(stream)

    selected_parameter_ids = {str(stream["parameter_id"]) for stream in selected_streams}
    selected_sensor_ids = {str(stream["sensor_id"]) for stream in selected_streams}
    parameters = [row for row in all_parameters if str(row["id"]) in selected_parameter_ids]
    sensors = [row for row in all_sensors if str(row["id"]) in selected_sensor_ids]
    _write_json(product_root / "catalog" / "parameters.json", _public_rows(parameters))
    _write_json(product_root / "catalog" / "sensors.json", _public_rows(sensors))
    _write_json(product_root / "catalog" / "streams.json", _public_rows(selected_streams))

    files: list[dict[str, Any]] = []
    source_results: list[dict[str, Any]] = []
    observation_count = 0
    for ordinal, source in enumerate(sources, start=1):
        source_id, run_id = str(source["id"]), str(source["run_id"])
        source_root = product_root / "runs" / run_id / "sources" / source_id
        source_root.mkdir(parents=True, exist_ok=True)
        source_streams = streams_by_source.get(source_id, [])
        source_parameters = _related_rows(parameters, source_streams, "parameter_id")
        source_sensors = _related_rows(sensors, source_streams, "sensor_id")
        observations = _observations(telemetry, project_id, run_id, source_streams)
        observation_count += len(observations)

        profile = _import_profile(source, source_streams)
        _write_json(source_root / "import-profile.json", profile)
        profile_path = _relative(output_root, source_root / "import-profile.json")
        files.append(_file_entry(output_root, source_root / "import-profile.json"))

        source_payload_path = _write_source_payload(source, source_root, kvstore)
        if source_payload_path is not None:
            files.append(_file_entry(output_root, source_payload_path))

        if normalized_format == "json":
            observations_path = source_root / "observations.ndjson"
            _write_ndjson(observations_path, observations)
            files.append(_file_entry(output_root, observations_path))
        elif normalized_format == "sqlite":
            database_path = source_root / "telemetry.sqlite"
            _write_sqlite(
                database_path,
                source=source,
                streams=source_streams,
                parameters=source_parameters,
                sensors=source_sensors,
                observations=observations,
            )
            files.append(_file_entry(output_root, database_path))
        else:
            workbook_path = source_root / "telemetry.xlsx"
            _write_xlsx(
                workbook_path,
                source=source,
                streams=source_streams,
                parameters=source_parameters,
                sensors=source_sensors,
                observations=observations,
            )
            files.append(_file_entry(output_root, workbook_path))

        source_results.append({
            "source_id": source_id,
            "run_id": run_id,
            "import_profile": profile_path,
            "stream_count": len(source_streams),
            "observation_count": len(observations),
            "source_payload": None if source_payload_path is None else _relative(output_root, source_payload_path),
        })
        _progress(progress_callback, product="telemetry", completed=ordinal, total=len(sources))

    context_files = _write_timeline_context(
        telemetry, project_id=project_id, run_ids=selected_run_ids, output_root=output_root, product_root=product_root,
    )
    files.extend(context_files)
    for catalog_file in ("parameters.json", "sensors.json", "streams.json"):
        files.append(_file_entry(output_root, product_root / "catalog" / catalog_file))
    return {
        "product": "telemetry",
        "schema_version": TELEMETRY_PRODUCT_SCHEMA_VERSION,
        "format": normalized_format,
        "source_count": len(sources),
        "stream_count": len(selected_streams),
        "observation_count": observation_count,
        "sources": source_results,
        "files": sorted(files, key=lambda item: item["path"]),
    }


class TelemetryBundleWriter:
    """Small adapter for workers that prefer an object-oriented writer."""

    def __init__(self, repository: Any, kvstore: Any | None = None):
        self.repository = repository
        self.kvstore = kvstore

    def write(self, **kwargs: Any) -> dict[str, Any]:
        return write_telemetry_bundle(self.repository, kvstore=self.kvstore, **kwargs)


def _selection_ids(selection: Mapping[str, Any], key: str) -> set[str]:
    value = selection.get(key, ())
    if value is None:
        return set()
    if isinstance(value, (str, bytes)):
        value = (value,)
    if not isinstance(value, Sequence):
        raise ValueError(f"Telemetry selection {key!r} must be a sequence of UUIDs.")
    return {str(item) for item in value if str(item).strip()}


def _selected_sources(
    telemetry: Any, *, project_id: str, source_ids: set[str], run_ids: set[str],
) -> list[dict[str, Any]]:
    # Query by run when supplied so repositories can retain their indexed path;
    # otherwise one project-scoped query is enough.
    candidates: list[dict[str, Any]] = []
    if run_ids:
        for run_id in sorted(run_ids):
            candidates.extend(telemetry.list_telemetry_sources(project_id=project_id, run_id=run_id))
    else:
        candidates = telemetry.list_telemetry_sources(project_id=project_id)
    result = {
        str(row["id"]): dict(row)
        for row in candidates
        if (not source_ids or str(row["id"]) in source_ids)
    }
    missing = source_ids - set(result)
    if missing:
        raise ValueError("One or more telemetry sources were not found in the selected project.")
    return sorted(result.values(), key=lambda row: (str(row["run_id"]), str(row["id"])))


def _observations(telemetry: Any, project_id: str, run_id: str, streams: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for stream in sorted(streams, key=lambda item: str(item["public_id"])):
        for observation in telemetry.list_telemetry_observations(
            project_id=project_id, run_id=run_id, stream_id=int(stream["id"]),
        ):
            output.append({
                "stream_id": str(stream["public_id"]),
                "observed_at": observation["observed_at"],
                "value": observation["value"],
                "qc_flag": observation.get("qc_flag"),
            })
    return output


def _import_profile(source: Mapping[str, Any], streams: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metadata = dict(source.get("metadata") or {})
    return {
        "schema_version": "1.0",
        "source_id": str(source["id"]),
        "run_id": str(source["run_id"]),
        "format": source.get("format"),
        "parser_name": source.get("parser_name"),
        "parser_version": source.get("parser_version"),
        "original_filename": source.get("filename"),
        "source_checksum_sha256": source.get("checksum"),
        "source_size_bytes": source.get("size_bytes"),
        "source_payload_path": "source/original.bin",
        "timestamp_column": metadata.get("timestamp_column"),
        "timestamp_format": metadata.get("timestamp_format"),
        "source_timezone": metadata.get("source_timezone"),
        "canonical_timezone": metadata.get("canonical_timezone", "UTC"),
        "source_metadata": metadata,
        "streams": [
            {
                "stream_id": str(row["public_id"]),
                "stream_key": row["stream_key"],
                "sensor_id": str(row["sensor_id"]),
                "parameter_id": str(row["parameter_id"]),
                "native_unit": row["native_unit"],
                "canonical_unit": row.get("canonical_unit"),
                "metadata": row.get("metadata") or {},
                "qc_scheme": row.get("qc_scheme"),
                "interpolation": row.get("interpolation"),
                "max_gap": row.get("max_gap"),
                "sampling_rate_hz": row.get("sampling_rate_hz"),
                "priority": row.get("priority"),
                "is_default": row.get("is_default"),
            }
            for row in sorted(streams, key=lambda item: str(item["public_id"]))
        ],
    }


def _write_source_payload(source: Mapping[str, Any], source_root: Path, kvstore: Any | None) -> Path | None:
    key = str(source.get("source_payload_key") or "").strip()
    if not key:
        return None
    if kvstore is None:
        raise ValueError("Telemetry export requires the project's KVStore to include source bytes.")
    payload = kvstore.get_store(key)
    expected = source.get("checksum")
    actual = hashlib.sha256(payload).hexdigest()
    if expected and actual != expected:
        raise ValueError(f"Telemetry source {source['id']} payload checksum does not match its recorded checksum.")
    destination = source_root / "source" / "original.bin"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def _write_timeline_context(
    telemetry: Any, *, project_id: str, run_ids: set[str], output_root: Path, product_root: Path,
) -> list[dict[str, Any]]:
    types_path = product_root / "context" / "timeline-event-types.json"
    _write_json(types_path, _public_rows(telemetry.list_timeline_event_types(project_id=project_id)))
    entries = [_file_entry(output_root, types_path)]
    for run_id in sorted(run_ids):
        events = telemetry.list_timeline_events(project_id=project_id, run_id=run_id)
        path = product_root / "runs" / run_id / "timeline-events.json"
        _write_json(path, _public_rows(events))
        entries.append(_file_entry(output_root, path))
    return entries


def _write_sqlite(
    path: Path, *, source: Mapping[str, Any], streams: Sequence[Mapping[str, Any]],
    parameters: Sequence[Mapping[str, Any]], sensors: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]],
) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript("""
            CREATE TABLE source (id TEXT PRIMARY KEY, run_id TEXT NOT NULL, format TEXT, parser_name TEXT,
                parser_version TEXT, filename TEXT, checksum_sha256 TEXT, size_bytes INTEGER, metadata_json TEXT NOT NULL);
            CREATE TABLE parameters (id TEXT PRIMARY KEY, parameter_key TEXT, display_name TEXT, definition TEXT,
                standard_name TEXT, canonical_unit TEXT, metadata_json TEXT NOT NULL);
            CREATE TABLE sensors (id TEXT PRIMARY KEY, sensor_key TEXT, display_name TEXT, manufacturer TEXT,
                model TEXT, serial_number TEXT, metadata_json TEXT NOT NULL);
            CREATE TABLE streams (id TEXT PRIMARY KEY, source_id TEXT NOT NULL, sensor_id TEXT NOT NULL,
                parameter_id TEXT NOT NULL, stream_key TEXT, native_unit TEXT, canonical_unit TEXT,
                sampling_rate_hz REAL, interpolation TEXT, max_gap TEXT, priority INTEGER, is_default INTEGER,
                qc_scheme TEXT, metadata_json TEXT NOT NULL);
            CREATE TABLE observations (stream_id TEXT NOT NULL, observed_at TEXT NOT NULL, value REAL NOT NULL,
                qc_flag INTEGER, PRIMARY KEY (stream_id, observed_at));
        """)
        connection.execute(
            "INSERT INTO source VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (str(source["id"]), str(source["run_id"]), source.get("format"), source.get("parser_name"),
             source.get("parser_version"), source.get("filename"), source.get("checksum"), source.get("size_bytes"),
             _json(source.get("metadata") or {})),
        )
        connection.executemany(
            "INSERT INTO parameters VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(str(row["id"]), row.get("parameter_key"), row.get("display_name"), row.get("definition"),
              row.get("standard_name"), row.get("canonical_unit"), _json(row.get("metadata") or {})) for row in parameters],
        )
        connection.executemany(
            "INSERT INTO sensors VALUES (?, ?, ?, ?, ?, ?, ?)",
            [(str(row["id"]), row.get("sensor_key"), row.get("display_name"), row.get("manufacturer"), row.get("model"),
              row.get("serial_number"), _json(row.get("metadata") or {})) for row in sensors],
        )
        connection.executemany(
            "INSERT INTO streams VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(str(row["public_id"]), str(row["source_id"]), str(row["sensor_id"]), str(row["parameter_id"]),
              row.get("stream_key"), row.get("native_unit"), row.get("canonical_unit"), row.get("sampling_rate_hz"),
              row.get("interpolation"), _ready_scalar(row.get("max_gap")), row.get("priority"), int(bool(row.get("is_default"))),
              row.get("qc_scheme"), _json(row.get("metadata") or {})) for row in streams],
        )
        connection.executemany(
            "INSERT INTO observations VALUES (?, ?, ?, ?)",
            [(row["stream_id"], _ready_scalar(row["observed_at"]), row["value"], row.get("qc_flag")) for row in observations],
        )


def _write_xlsx(
    path: Path, *, source: Mapping[str, Any], streams: Sequence[Mapping[str, Any]],
    parameters: Sequence[Mapping[str, Any]], sensors: Sequence[Mapping[str, Any]], observations: Sequence[Mapping[str, Any]],
) -> None:
    """Write a bounded, analyst-friendly view; JSON remains the import contract."""
    source_row = _public_rows([source])[0]
    source_row.pop("source_payload_key", None)
    source_row.pop("path", None)
    path.write_bytes(write_xlsx({
        "Source": [source_row],
        "Parameters": _public_rows(parameters),
        "Sensors": _public_rows(sensors),
        "Streams": _public_rows(streams),
        "Observations": _public_rows(observations),
    }))


def _related_rows(rows: Sequence[Mapping[str, Any]], streams: Sequence[Mapping[str, Any]], field: str) -> list[dict[str, Any]]:
    wanted = {str(stream[field]) for stream in streams}
    return [dict(row) for row in rows if str(row["id"]) in wanted]


def _public_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_json(value) + "\n", encoding="utf-8")


def _write_ndjson(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(_json(row) + "\n")


def _json(value: Any) -> str:
    return json.dumps(json_ready(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ready_scalar(value: Any) -> Any:
    return json_ready(value)


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _file_entry(root: Path, path: Path) -> dict[str, Any]:
    payload = path.read_bytes()
    return {"path": _relative(root, path), "size_bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def _progress(callback: Callable[[Mapping[str, Any]], None] | None, **payload: Any) -> None:
    if callback is not None:
        callback(payload)
