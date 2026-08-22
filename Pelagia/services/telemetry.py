from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np

from ..domain import normalize_collections
from ..storage.blob_store import BlobStore
from ..storage.postgres import PostgresRepository
from ..utils.serialization import json_ready
from .telemetry_units import (
    DEFAULT_TELEMETRY_UNIT_REGISTRY,
    AffineUnitConversion,
)


INTERPOLATION_METHODS = frozenset({"linear", "nearest", "previous", "none"})


@dataclass(frozen=True, slots=True)
class TelemetryColumn:
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
    interpolation: str = "none"
    max_gap_seconds: float | None = None
    sampling_rate_hz: float | None = None
    priority: int = 100
    is_default: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)
    sensor_metadata: Mapping[str, Any] = field(default_factory=dict)
    parameter_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TelemetryCsvSpec:
    timestamp_column: str
    streams: Sequence[TelemetryColumn]
    timestamp_format: str = "iso8601"
    source_timezone: str = "UTC"
    delimiter: str = ","
    parser_name: str = "pelagia.delimited"
    parser_version: str = "1"
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TelemetryRangeFilter:
    """A canonical telemetry value range used by ROI selection queries."""

    parameter_key: str
    min_value: float | None = None
    max_value: float | None = None


def parse_telemetry_filters(values: Sequence[str] | None) -> list[TelemetryRangeFilter]:
    """Parse repeated JSON query values into validated telemetry ranges."""
    parsed: list[TelemetryRangeFilter] = []
    for raw in values or ():
        try:
            payload = json.loads(raw)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("Telemetry filters must be JSON objects.") from exc
        if not isinstance(payload, Mapping):
            raise ValueError("Each telemetry filter must be a JSON object.")
        parameter_key = str(payload.get("parameter_key") or payload.get("parameter") or "").strip()
        if not parameter_key:
            raise ValueError("Telemetry filters require a parameter_key.")
        min_value = _finite_filter_bound(payload.get("min_value", payload.get("min")))
        max_value = _finite_filter_bound(payload.get("max_value", payload.get("max")))
        if min_value is None and max_value is None:
            raise ValueError(f"Telemetry filter {parameter_key!r} requires a minimum or maximum value.")
        if min_value is not None and max_value is not None and min_value > max_value:
            raise ValueError(f"Telemetry filter {parameter_key!r} has a minimum above its maximum.")
        parsed.append(TelemetryRangeFilter(parameter_key, min_value, max_value))
    return parsed


def _finite_filter_bound(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Telemetry filter bounds must be finite numbers.") from exc
    if not math.isfinite(parsed):
        raise ValueError("Telemetry filter bounds must be finite numbers.")
    return parsed


def infer_timestamp_format(value: object, requested_format: str = "auto") -> str:
    """Resolve the UI-friendly auto format to the parser format."""
    if requested_format != "auto":
        return requested_format
    try:
        float(str(value).strip())
    except (TypeError, ValueError):
        return "iso8601"
    return "unix_seconds"


def iter_parsed_csv_rows(
    source: io.TextIOBase,
    *,
    timestamp_column: str,
    timestamp_format: str,
    source_timezone: str,
    delimiter: str,
    strict: bool = True,
):
    """Yield ``(row_number, row, timestamp, error)`` using import semantics.

    The analyzer uses ``strict=False`` to collect bounded diagnostics; ingestion
    uses the default strict mode and raises on the first invalid timestamp.
    """
    reader = csv.DictReader(source, delimiter=delimiter)
    fieldnames = list(reader.fieldnames or [])
    if not fieldnames:
        raise ValueError("Telemetry CSV has no header row.")
    if timestamp_column not in fieldnames:
        raise ValueError(f"Timestamp column {timestamp_column!r} was not found.")
    for row_number, row in enumerate(reader, start=2):
        try:
            observed_at = normalize_observed_at(
                row.get(timestamp_column),
                timestamp_format=infer_timestamp_format(row.get(timestamp_column), timestamp_format),
                source_timezone=source_timezone,
            )
        except (TypeError, ValueError, OverflowError, ZoneInfoNotFoundError) as exc:
            if strict:
                raise ValueError(f"Invalid timestamp on row {row_number}: {exc}") from exc
            yield row_number, row, None, exc
            continue
        yield row_number, row, observed_at, None


def normalize_observed_at(
    value: str | int | float | datetime,
    *,
    timestamp_format: str = "iso8601",
    source_timezone: str = "UTC",
) -> datetime:
    """Normalize a source timestamp to an aware UTC datetime.

    PostgreSQL ``timestamptz`` remains canonical. It provides more precision
    than the millisecond-level requirement without a second integer timebase.
    """
    if isinstance(value, datetime):
        observed_at = value
    elif timestamp_format == "unix_seconds":
        observed_at = datetime.fromtimestamp(float(value), tz=timezone.utc)
    elif timestamp_format == "unix_milliseconds":
        observed_at = datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc)
    elif timestamp_format == "iso8601":
        text = str(value).strip()
        if text.endswith("Z"):
            text = f"{text[:-1]}+00:00"
        observed_at = datetime.fromisoformat(text)
    else:
        observed_at = datetime.strptime(str(value), timestamp_format)

    if observed_at.tzinfo is None:
        zone = ZoneInfo(source_timezone)
        fold_zero = observed_at.replace(tzinfo=zone, fold=0)
        fold_one = observed_at.replace(tzinfo=zone, fold=1)
        valid_zero = fold_zero.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == observed_at
        valid_one = fold_one.astimezone(timezone.utc).astimezone(zone).replace(tzinfo=None) == observed_at
        if not valid_zero and not valid_one:
            raise ValueError(
                f"Timestamp {observed_at.isoformat()} does not exist in timezone {source_timezone!r}."
            )
        if valid_zero and valid_one and fold_zero.utcoffset() != fold_one.utcoffset():
            raise ValueError(
                f"Timestamp {observed_at.isoformat()} is ambiguous in timezone {source_timezone!r}; "
                "supply an explicit UTC offset."
            )
        observed_at = fold_zero if valid_zero else fold_one
    return observed_at.astimezone(timezone.utc)


def _import_key(*, checksum: str, spec: TelemetryCsvSpec, collections: Sequence[str]) -> str:
    """Identify one reproducible interpretation of immutable source bytes."""
    spec_payload = asdict(spec)
    spec_payload["streams"] = sorted(
        spec_payload["streams"], key=lambda item: str(item["stream_key"]),
    )
    payload = {
        "source_sha256": checksum,
        "format": "delimited",
        "collections": sorted(collections),
        "spec": spec_payload,
    }
    canonical = json.dumps(
        json_ready(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class TelemetryIngestionService:
    def __init__(self, repository: PostgresRepository, blob_store: BlobStore | None = None):
        self.repository = repository
        self.telemetry = getattr(repository, "telemetry", repository)
        self.blob_store = blob_store

    def import_csv(
        self,
        path: Path,
        *,
        project_id: str,
        run_id: str,
        spec: TelemetryCsvSpec,
        collections: Iterable[str] | None = None,
    ) -> dict[str, Any]:
        resolved = path.expanduser().resolve()
        if not resolved.is_file():
            raise ValueError(f"Telemetry source {resolved} is not a file.")
        if not spec.streams:
            raise ValueError("At least one telemetry stream mapping is required.")
        if len(spec.delimiter) != 1:
            raise ValueError("CSV delimiter must be one character.")
        if self.blob_store is None:
            raise ValueError("Telemetry import requires the selected project's KVStore.")

        # Persist the exact source bytes before parsing. KVStore content addressing
        # makes this write retry-safe; database publication happens only after the
        # immutable snapshot has been completely validated.
        with resolved.open("rb") as source_file:
            put_stream = getattr(self.blob_store, "put_stream", None)
            if callable(put_stream):
                source_payload_key = str(put_stream(source_file))
            else:
                source_payload_key = str(self.blob_store.put_store(source_file.read()))
        source_payload = self.blob_store.get_store(source_payload_key)
        expected_checksum = hashlib.sha256(source_payload).hexdigest()
        source_size_bytes = len(source_payload)
        resolved_collections = normalize_collections(collections)
        import_key = _import_key(
            checksum=expected_checksum, spec=spec, collections=resolved_collections,
        )

        get_existing = getattr(self.telemetry, "get_telemetry_import", None)
        if callable(get_existing):
            existing_import = get_existing(
                project_id=project_id, run_id=run_id, import_key=import_key,
            )
            if existing_import is not None:
                return existing_import

        parameter_defaults: dict[str, list[TelemetryColumn]] = {}
        unit_conversions: dict[str, AffineUnitConversion] = {}
        stream_keys: set[str] = set()
        for stream in spec.streams:
            for field_name in (
                "column", "stream_key", "sensor_key", "parameter_key",
                "native_unit", "canonical_unit",
            ):
                if not str(getattr(stream, field_name)).strip():
                    raise ValueError(f"Telemetry stream {field_name.replace('_', ' ')} must not be blank.")
            if stream.stream_key in stream_keys:
                raise ValueError(f"Duplicate telemetry stream key {stream.stream_key!r}.")
            stream_keys.add(stream.stream_key)
            if stream.interpolation not in INTERPOLATION_METHODS:
                raise ValueError(f"Unsupported interpolation method {stream.interpolation!r}.")
            if stream.sampling_rate_hz is not None and (
                not math.isfinite(float(stream.sampling_rate_hz)) or float(stream.sampling_rate_hz) <= 0
            ):
                raise ValueError(f"Stream {stream.stream_key!r} requires a finite positive sampling rate.")
            if stream.max_gap_seconds is not None and (
                not math.isfinite(float(stream.max_gap_seconds)) or float(stream.max_gap_seconds) <= 0
            ):
                raise ValueError(f"Stream {stream.stream_key!r} requires a finite positive maximum gap.")
            if stream.interpolation != "none" and (
                stream.max_gap_seconds is None or stream.max_gap_seconds <= 0
            ):
                raise ValueError(
                    f"Stream {stream.stream_key!r} requires a positive maximum gap."
                )
            try:
                unit_conversions[stream.stream_key] = (
                    DEFAULT_TELEMETRY_UNIT_REGISTRY.validate_affine_conversion(
                        stream.native_unit,
                        stream.canonical_unit,
                        scale=stream.scale,
                        offset=stream.offset,
                    )
                )
            except ValueError as exc:
                raise ValueError(f"Stream {stream.stream_key!r}: {exc}") from exc
            parameter_defaults.setdefault(stream.parameter_key, []).append(stream)
        for parameter_key, candidates in parameter_defaults.items():
            default_count = sum(candidate.is_default for candidate in candidates)
            if default_count > 1:
                raise ValueError(
                    f"Parameter {parameter_key!r} cannot define more than one default stream."
                )

        existing_streams = self.telemetry.list_telemetry_streams(
            project_id=project_id,
            run_id=run_id,
            parameter_keys=list(parameter_defaults),
        )
        existing_defaults = {
            str(item["parameter_key"]) for item in existing_streams if item.get("is_default")
        }
        for parameter_key, candidates in parameter_defaults.items():
            if parameter_key in existing_defaults and any(candidate.is_default for candidate in candidates):
                raise ValueError(
                    f"Parameter {parameter_key!r} already has a default stream for this run."
                )
            if (
                parameter_key not in existing_defaults
                and len(candidates) > 1
                and not any(candidate.is_default for candidate in candidates)
            ):
                raise ValueError(
                    f"Parameter {parameter_key!r} has multiple streams; exactly one must be default."
                )

        def iter_observations():
            with io.TextIOWrapper(
                io.BytesIO(source_payload), encoding="utf-8-sig", newline="",
            ) as source_file:
                fieldnames = set(next(csv.reader(source_file, delimiter=spec.delimiter), []))
                required = {spec.timestamp_column}
                required.update(stream.column for stream in spec.streams)
                required.update(stream.qc_column for stream in spec.streams if stream.qc_column)
                missing = sorted(required - fieldnames)
                if missing:
                    raise ValueError(f"Telemetry file is missing columns: {', '.join(missing)}.")

                source_file.seek(0)
                for row_number, row, observed_at, _error in iter_parsed_csv_rows(
                    source_file,
                    timestamp_column=spec.timestamp_column,
                    timestamp_format=spec.timestamp_format,
                    source_timezone=spec.source_timezone,
                    delimiter=spec.delimiter,
                ):
                    for stream in spec.streams:
                        raw_value = row.get(stream.column)
                        if raw_value is None or not str(raw_value).strip():
                            continue
                        try:
                            value = float(raw_value) * float(stream.scale) + float(stream.offset)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid value for {stream.column!r} on row {row_number}."
                            ) from exc
                        if not math.isfinite(value):
                            raise ValueError(
                                f"Non-finite value for {stream.column!r} on row {row_number}."
                            )
                        qc_flag = None
                        if stream.qc_column and str(row.get(stream.qc_column) or "").strip():
                            try:
                                qc_flag = int(row[stream.qc_column])
                            except ValueError as exc:
                                raise ValueError(
                                    f"Invalid QC flag for {stream.qc_column!r} on row {row_number}."
                                ) from exc
                            if qc_flag < -32768 or qc_flag > 32767:
                                raise ValueError(
                                    f"QC flag for {stream.qc_column!r} on row {row_number} is outside int16 range."
                                )
                        yield stream.stream_key, observed_at, value, qc_flag

        last_at: dict[str, datetime] = {}
        sampled_deltas: dict[str, list[float]] = {stream.stream_key: [] for stream in spec.streams}
        counts: dict[str, int] = {stream.stream_key: 0 for stream in spec.streams}
        for stream_key, observed_at, _value, _qc_flag in iter_observations():
            previous_at = last_at.get(stream_key)
            if previous_at is not None:
                if observed_at == previous_at:
                    raise ValueError(
                        f"Stream {stream_key!r} contains duplicate timestamp {observed_at.isoformat()}."
                    )
                if observed_at < previous_at:
                    raise ValueError(
                        f"Stream {stream_key!r} is not ordered by timestamp at {observed_at.isoformat()}."
                    )
                if len(sampled_deltas[stream_key]) < 10_000:
                    sampled_deltas[stream_key].append((observed_at - previous_at).total_seconds())
            last_at[stream_key] = observed_at
            counts[stream_key] += 1

        parameters: dict[str, dict[str, Any]] = {}
        sensors: dict[str, dict[str, Any]] = {}
        stream_payloads = []
        for stream in spec.streams:
            unit_conversion = unit_conversions[stream.stream_key]
            existing = parameters.get(stream.parameter_key)
            if existing and existing["canonical_unit"] != unit_conversion.canonical_unit:
                raise ValueError(
                    f"Parameter {stream.parameter_key!r} has conflicting canonical units."
                )
            parameters[stream.parameter_key] = {
                "parameter_key": stream.parameter_key,
                "display_name": stream.display_name,
                "standard_name": stream.standard_name,
                "canonical_unit": unit_conversion.canonical_unit,
                "metadata": dict(stream.parameter_metadata),
            }
            sensor_payload = {
                "sensor_key": stream.sensor_key,
                "display_name": stream.sensor_display_name,
                "manufacturer": stream.manufacturer,
                "model": stream.model,
                "serial_number": stream.serial_number,
                "metadata": dict(stream.sensor_metadata),
            }
            existing_sensor = sensors.get(stream.sensor_key)
            if existing_sensor is not None:
                for field_name in ("manufacturer", "model", "serial_number"):
                    if (
                        existing_sensor.get(field_name) and sensor_payload.get(field_name)
                        and existing_sensor[field_name] != sensor_payload[field_name]
                    ):
                        raise ValueError(
                            f"Sensor {stream.sensor_key!r} has conflicting {field_name.replace('_', ' ')}."
                        )
                sensor_payload = {
                    field_name: sensor_payload.get(field_name) or existing_sensor.get(field_name)
                    for field_name in sensor_payload
                }
                sensor_payload["metadata"] = {
                    **dict(existing_sensor.get("metadata") or {}),
                    **dict(stream.sensor_metadata),
                }
            sensors[stream.sensor_key] = sensor_payload
            deltas = sampled_deltas[stream.stream_key]
            estimated_rate = None if not deltas else 1.0 / float(np.median(np.asarray(deltas)))
            stream_payloads.append(
                {
                    "stream_key": stream.stream_key,
                    "sensor_key": stream.sensor_key,
                    "parameter_key": stream.parameter_key,
                    "native_unit": unit_conversion.native_unit,
                    "sampling_rate_hz": stream.sampling_rate_hz or estimated_rate,
                    "interpolation": stream.interpolation,
                    "max_gap_seconds": stream.max_gap_seconds,
                    "priority": stream.priority,
                    "is_default": stream.is_default or (
                        len(parameter_defaults[stream.parameter_key]) == 1
                        and stream.parameter_key not in existing_defaults
                    ),
                    "metadata": {
                        **dict(stream.metadata),
                        "source_column": stream.column,
                        "conversion": {
                            "scale": float(stream.scale), "offset": float(stream.offset),
                        },
                        "unit_provenance": {
                            "registry": "pelagia.telemetry_units",
                            "registry_version": DEFAULT_TELEMETRY_UNIT_REGISTRY.version,
                            "declared_native_unit": stream.native_unit,
                            "declared_canonical_unit": stream.canonical_unit,
                            "native_unit": unit_conversion.native_unit,
                            "canonical_unit": unit_conversion.canonical_unit,
                            "scale": float(stream.scale),
                            "offset": float(stream.offset),
                        },
                        "observation_count": counts[stream.stream_key],
                    },
                }
            )

        return self.telemetry.ingest_telemetry(
            project_id=project_id,
            run_id=run_id,
            asset={
                "id": str(uuid.uuid4()),
                "filename": resolved.name,
                "path": str(resolved),
                "checksum": expected_checksum,
                "size_bytes": source_size_bytes,
                "collections": resolved_collections,
                "metadata": {
                    "telemetry": True,
                    "source_payload_key": source_payload_key,
                },
            },
            source={
                "format": "delimited",
                "parser_name": spec.parser_name,
                "parser_version": spec.parser_version,
                "import_key": import_key,
                "source_payload_key": source_payload_key,
                "metadata": {
                    **dict(spec.metadata),
                    "timestamp_column": spec.timestamp_column,
                    "timestamp_format": spec.timestamp_format,
                    "source_timezone": spec.source_timezone,
                    "canonical_timezone": "UTC",
                    "unit_registry": "pelagia.telemetry_units",
                    "unit_registry_version": DEFAULT_TELEMETRY_UNIT_REGISTRY.version,
                },
            },
            parameters=list(parameters.values()),
            sensors=list(sensors.values()),
            streams=stream_payloads,
            observations=iter_observations(),
        )


class TelemetryResolver:
    def __init__(self, repository: PostgresRepository):
        self.repository = repository
        self.telemetry = getattr(repository, "telemetry", repository)

    @staticmethod
    def _excluded(stream: Mapping[str, Any], observation: Mapping[str, Any] | None) -> bool:
        if observation is None:
            return False
        excluded = set((stream.get("metadata") or {}).get("excluded_qc_flags") or [])
        return observation.get("qc_flag") in excluded

    @staticmethod
    def _gap_seconds(left: datetime, right: datetime) -> float:
        return abs((right - left).total_seconds())

    def resolve_stream(
        self, stream: Mapping[str, Any], observed_at: datetime,
    ) -> dict[str, Any]:
        target = normalize_observed_at(observed_at)
        excluded_qc_flags = tuple((stream.get("metadata") or {}).get("excluded_qc_flags") or ())
        around = self.telemetry.telemetry_observations_around(
            project_id=str(stream["project_id"]), stream_id=int(stream["id"]), observed_at=target,
            excluded_qc_flags=excluded_qc_flags,
        )
        previous = around["previous"]
        following = around["next"]
        # ``previous``/``next`` retain the raw bracket so an excluded exact
        # observation can be reported as such.  Repeated lookup methods use
        # the nearest QC-valid bracket when the repository can provide it.
        previous_valid = around.get("previous_valid", previous)
        following_valid = around.get("next_valid", following)
        base = {
            "parameter": stream["parameter_key"],
            "unit": stream["canonical_unit"],
            "stream_id": str(stream["public_id"]),
            "stream_key": stream["stream_key"],
            "sensor_key": stream["sensor_key"],
            "observed_at": target,
            "method": stream["interpolation"],
        }
        if previous is not None and previous["observed_at"] == target:
            if self._excluded(stream, previous):
                return {**base, "value": None, "missing_reason": "qc_excluded"}
            return {
                **base, "value": previous["value"], "method": "exact",
                "source_observed_at": [previous["observed_at"]], "qc_flags": [previous["qc_flag"]],
            }
        method = stream["interpolation"]
        if method == "none":
            return {**base, "value": None, "missing_reason": "interpolation_disabled"}

        max_gap = stream.get("max_gap")
        max_gap_seconds = None if max_gap is None else max_gap.total_seconds()
        candidates = [row for row in (previous_valid, following_valid)
                      if row is not None and not self._excluded(stream, row)]
        if not candidates:
            reason = "qc_excluded" if previous is not None or following is not None else "outside_stream_range"
            return {**base, "value": None, "missing_reason": reason}

        if method in {"nearest", "previous"}:
            chosen = previous_valid if method == "previous" else min(
                candidates, key=lambda row: self._gap_seconds(row["observed_at"], target)
            )
            if chosen is None or self._excluded(stream, chosen):
                return {**base, "value": None, "missing_reason": "outside_stream_range"}
            gap = self._gap_seconds(chosen["observed_at"], target)
            if max_gap_seconds is not None and gap > max_gap_seconds:
                return {**base, "value": None, "missing_reason": "gap_exceeded", "gap_seconds": gap}
            return {
                **base, "value": chosen["value"], "source_observed_at": [chosen["observed_at"]],
                "qc_flags": [chosen["qc_flag"]], "gap_seconds": gap,
            }

        if previous is None or following is None:
            return {**base, "value": None, "missing_reason": "outside_stream_range"}
        if self._excluded(stream, previous) or self._excluded(stream, following):
            return {**base, "value": None, "missing_reason": "qc_excluded"}
        bracket_gap = self._gap_seconds(previous["observed_at"], following["observed_at"])
        if max_gap_seconds is not None and bracket_gap > max_gap_seconds:
            return {
                **base, "value": None, "missing_reason": "gap_exceeded",
                "gap_seconds": bracket_gap,
            }
        fraction = (target - previous["observed_at"]).total_seconds() / bracket_gap
        left = float(previous["value"])
        right = float(following["value"])
        circular_period = (stream.get("metadata") or {}).get("circular_period")
        if circular_period is not None:
            period = float(circular_period)
            minimum = float((stream.get("metadata") or {}).get("circular_minimum", -period / 2.0))
            delta = ((right - left + period / 2.0) % period) - period / 2.0
            value = ((left + fraction * delta - minimum) % period) + minimum
        else:
            value = left + fraction * (right - left)
        return {
            **base, "value": value,
            "source_observed_at": [previous["observed_at"], following["observed_at"]],
            "qc_flags": [previous["qc_flag"], following["qc_flag"]],
            "gap_seconds": bracket_gap,
        }

    def at(
        self, *, project_id: str, run_id: str, observed_at: datetime,
        parameters: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        streams = self.telemetry.list_telemetry_streams(
            project_id=project_id, run_id=run_id, parameter_keys=parameters,
        )
        selected: dict[str, Mapping[str, Any]] = {}
        for stream in streams:
            key = stream["parameter_key"]
            if stream["is_default"] or key not in selected:
                selected[key] = stream
        requested = list(parameters or selected)
        values = {}
        for key in requested:
            stream = selected.get(key)
            values[key] = (
                {"parameter": key, "value": None, "missing_reason": "stream_not_found"}
                if stream is None else self.resolve_stream(stream, observed_at)
            )
        return {"run_id": run_id, "observed_at": normalize_observed_at(observed_at), "telemetry": values}

    def align_timestamps(
        self, *, project_id: str, run_id: str, observed_at: Sequence[datetime],
        parameters: Sequence[str] | None = None, chunk_size: int = 100_000,
    ) -> list[dict[str, Any]]:
        return [
            row
            for chunk in self.iter_aligned_timestamps(
                project_id=project_id, run_id=run_id, observed_at=observed_at,
                parameters=parameters, chunk_size=chunk_size,
            )
            for row in chunk
        ]

    def iter_aligned_timestamps(
        self, *, project_id: str, run_id: str, observed_at: Sequence[datetime],
        parameters: Sequence[str] | None = None, chunk_size: int = 100_000,
        max_window_seconds: float = 3600.0,
    ):
        """Yield bounded, vectorized wide-alignment chunks.

        Scalar lookup remains the provenance-rich interactive contract. This
        path applies the same selection, interpolation, QC, and gap rules while
        keeping memory proportional to one requested chunk and its brackets.
        """
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive.")
        if max_window_seconds <= 0:
            raise ValueError("max_window_seconds must be positive.")
        streams = self.telemetry.list_telemetry_streams(
            project_id=project_id, run_id=run_id, parameter_keys=parameters,
        )
        selected: dict[str, Mapping[str, Any]] = {}
        for stream in streams:
            key = stream["parameter_key"]
            if stream["is_default"] or key not in selected:
                selected[key] = stream
        requested = list(parameters or selected)

        def target_chunks():
            current_chunk: list[datetime] = []
            chunk_min: datetime | None = None
            chunk_max: datetime | None = None
            for value in observed_at:
                target = normalize_observed_at(value)
                candidate_min = target if chunk_min is None else min(chunk_min, target)
                candidate_max = target if chunk_max is None else max(chunk_max, target)
                if current_chunk and (
                    len(current_chunk) >= chunk_size
                    or (candidate_max - candidate_min).total_seconds() > max_window_seconds
                ):
                    yield current_chunk
                    current_chunk = []
                    chunk_min = target
                    chunk_max = target
                else:
                    chunk_min = candidate_min
                    chunk_max = candidate_max
                current_chunk.append(target)
            if current_chunk:
                yield current_chunk

        for targets in target_chunks():
            rows = [
                {"run_id": run_id, "observed_at": target, "telemetry": {}}
                for target in targets
            ]
            if not targets:
                yield rows
                continue
            target_seconds = np.asarray([target.timestamp() for target in targets], dtype=np.float64)
            for parameter_key in requested:
                stream = selected.get(parameter_key)
                if stream is None:
                    for row in rows:
                        row["telemetry"][parameter_key] = {
                            "value": None, "missing_reason": "stream_not_found"
                        }
                    continue
                max_gap = stream.get("max_gap")
                margin = 0.0 if max_gap is None else max_gap.total_seconds()
                observations = self.telemetry.list_telemetry_observations(
                    project_id=project_id,
                    stream_id=int(stream["id"]),
                    start_at=min(targets) - timedelta(seconds=margin),
                    end_at=max(targets) + timedelta(seconds=margin),
                )
                excluded = set((stream.get("metadata") or {}).get("excluded_qc_flags") or [])
                if not observations:
                    for row in rows:
                        row["telemetry"][parameter_key] = {
                            "value": None, "missing_reason": "outside_stream_range"
                        }
                    continue
                source_seconds = np.asarray(
                    [item["observed_at"].timestamp() for item in observations], dtype=np.float64
                )
                source_values = np.asarray([item["value"] for item in observations], dtype=np.float64)
                source_valid = np.asarray(
                    [item.get("qc_flag") not in excluded for item in observations], dtype=np.bool_
                )
                positions = np.searchsorted(source_seconds, target_seconds, side="left")
                valid_indexes = np.flatnonzero(source_valid)
                for index, (target, position) in enumerate(zip(target_seconds, positions)):
                    previous_index = int(position) - 1
                    next_index = int(position)
                    exact = next_index < len(source_seconds) and source_seconds[next_index] == target
                    if exact:
                        result = (
                            {"value": float(source_values[next_index]), "method": "exact"}
                            if source_valid[next_index]
                            else {"value": None, "missing_reason": "qc_excluded"}
                        )
                    else:
                        method = stream["interpolation"]
                        if method == "none":
                            result = {"value": None, "missing_reason": "interpolation_disabled"}
                        elif method == "previous":
                            valid_before = valid_indexes[valid_indexes < next_index]
                            result = self._bulk_selected_value(
                                source_seconds, source_values, source_valid,
                                int(valid_before[-1]) if len(valid_before) else -1,
                                target, margin, method,
                            )
                        elif method == "nearest":
                            insertion = int(np.searchsorted(valid_indexes, next_index, side="left"))
                            candidates = [candidate for candidate in (
                                valid_indexes[insertion - 1] if insertion else -1,
                                valid_indexes[insertion] if insertion < len(valid_indexes) else -1,
                            ) if candidate >= 0]
                            if not candidates:
                                result = {"value": None, "missing_reason": "qc_excluded"}
                            else:
                                chosen = min(
                                    candidates,
                                    key=lambda candidate: abs(source_seconds[candidate] - target),
                                )
                                result = self._bulk_selected_value(
                                    source_seconds, source_values, source_valid, chosen,
                                    target, margin, method,
                                )
                        elif previous_index < 0 or next_index >= len(source_seconds):
                            result = {"value": None, "missing_reason": "outside_stream_range"}
                        else:
                            bracket_gap = source_seconds[next_index] - source_seconds[previous_index]
                            if not source_valid[previous_index] or not source_valid[next_index]:
                                result = {"value": None, "missing_reason": "qc_excluded"}
                            elif max_gap is not None and bracket_gap > margin:
                                result = {"value": None, "missing_reason": "gap_exceeded"}
                            else:
                                fraction = (target - source_seconds[previous_index]) / bracket_gap
                                left = source_values[previous_index]
                                right = source_values[next_index]
                                circular_period = (stream.get("metadata") or {}).get("circular_period")
                                if circular_period is not None:
                                    period = float(circular_period)
                                    minimum = float(
                                        (stream.get("metadata") or {}).get(
                                            "circular_minimum", -period / 2.0
                                        )
                                    )
                                    delta = ((right - left + period / 2.0) % period) - period / 2.0
                                    value = ((left + fraction * delta - minimum) % period) + minimum
                                else:
                                    value = left + fraction * (right - left)
                                result = {"value": float(value), "method": "linear"}
                    rows[index]["telemetry"][parameter_key] = {
                        "unit": stream["canonical_unit"], "stream_id": str(stream["public_id"]),
                        **result,
                    }
            yield rows

    @staticmethod
    def _bulk_selected_value(
        source_seconds: np.ndarray, source_values: np.ndarray, source_valid: np.ndarray,
        source_index: int,
        target: float, max_gap_seconds: float, method: str,
    ) -> dict[str, Any]:
        if source_index < 0 or source_index >= len(source_seconds):
            return {"value": None, "missing_reason": "outside_stream_range"}
        if not source_valid[source_index]:
            return {"value": None, "missing_reason": "outside_stream_range"}
        gap = abs(float(source_seconds[source_index] - target))
        if max_gap_seconds and gap > max_gap_seconds:
            return {"value": None, "missing_reason": "gap_exceeded"}
        return {"value": float(source_values[source_index]), "method": method}


def frame_context(
    repository: PostgresRepository,
    *,
    project_id: str,
    frame: Mapping[str, Any],
    parameters: Sequence[str] | None = None,
    include_telemetry: bool = True,
    include_events: bool = True,
) -> dict[str, Any]:
    captured_at = frame.get("captured_at")
    if not include_telemetry:
        telemetry: dict[str, Any] = {}
    elif captured_at is None:
        telemetry = {
            key: {"parameter": key, "value": None, "missing_reason": "frame_timestamp_missing"}
            for key in parameters or []
        }
    else:
        telemetry = TelemetryResolver(repository).at(
            project_id=project_id,
            run_id=str(frame["run_id"]),
            observed_at=captured_at,
            parameters=parameters,
        )["telemetry"]

    if not include_events or captured_at is None:
        events: list[dict[str, Any]] = []
    else:
        event_repository = getattr(repository, "telemetry", repository)
        events = event_repository.list_timeline_events_at(
            project_id=project_id, run_id=str(frame["run_id"]), observed_at=captured_at,
        )
    return {"telemetry": telemetry, "events": events}
