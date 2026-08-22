from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from Pelagia.services.telemetry import (
    TelemetryColumn,
    TelemetryCsvSpec,
    TelemetryIngestionService,
    TelemetryResolver,
    normalize_observed_at,
    parse_telemetry_filters,
)


class MemoryBlobStore:
    def __init__(self):
        self.payloads = {}

    def put_store(self, payload):
        payload = bytes(payload)
        key = hashlib.sha256(payload).hexdigest()
        self.payloads[key] = payload
        return key

    def get_store(self, key):
        return self.payloads[key]


class CapturingRepository:
    def __init__(self):
        self.telemetry = self
        self.payload = None

    def ingest_telemetry(self, **payload):
        materialized = list(payload.pop("observations"))
        by_stream = {item["stream_key"]: item for item in payload["streams"]}
        for stream_key, observed_at, value, qc_flag in materialized:
            by_stream[stream_key].setdefault("observations", []).append((observed_at, value, qc_flag))
        self.payload = payload
        return payload

    def list_telemetry_streams(self, **_kwargs):
        return []


def test_normalize_observed_at_supports_iso_and_epoch_milliseconds():
    expected = datetime(2026, 8, 21, 12, 30, 1, 125000, tzinfo=timezone.utc)

    assert normalize_observed_at("2026-08-21T04:30:01.125-08:00") == expected
    assert normalize_observed_at(
        expected.timestamp() * 1000, timestamp_format="unix_milliseconds"
    ) == expected


def test_parse_telemetry_filters_accepts_open_and_closed_ranges():
    filters = parse_telemetry_filters([
        '{"parameter_key":"temperature","min_value":2.5}',
        '{"parameter_key":"pressure","max_value":1010}',
        '{"parameter":"oxygen","min":1,"max":4}',
    ])

    assert [(item.parameter_key, item.min_value, item.max_value) for item in filters] == [
        ("temperature", 2.5, None),
        ("pressure", None, 1010.0),
        ("oxygen", 1.0, 4.0),
    ]


@pytest.mark.parametrize(
    "value",
    [
        '{"parameter_key":"temperature"}',
        '{"parameter_key":"temperature","min_value":3,"max_value":2}',
        '{"parameter_key":"temperature","min_value":"not-a-number"}',
        '[]',
    ],
)
def test_parse_telemetry_filters_rejects_invalid_ranges(value):
    with pytest.raises(ValueError):
        parse_telemetry_filters([value])


@pytest.mark.parametrize(
    "value, message",
    [
        ("2026-11-01T01:30:00", "ambiguous"),
        ("2026-03-08T02:30:00", "does not exist"),
    ],
)
def test_normalize_observed_at_rejects_unsafe_local_dst_times(value, message):
    with pytest.raises(ValueError, match=message):
        normalize_observed_at(value, source_timezone="America/Anchorage")


def test_csv_ingestion_standardizes_values_and_preserves_provenance(tmp_path):
    source = tmp_path / "ship.csv"
    source.write_text(
        "time,temp,temp_qc\n"
        "2026-08-21T00:00:00Z,32.0,1\n"
        "2026-08-21T00:00:01Z,33.8,1\n",
        encoding="utf-8",
    )
    repository = CapturingRepository()

    store = MemoryBlobStore()
    result = TelemetryIngestionService(repository, store).import_csv(
        source,
        project_id="project-1",
        run_id="run-1",
        spec=TelemetryCsvSpec(
            timestamp_column="time",
            streams=[
                TelemetryColumn(
                    column="temp",
                    qc_column="temp_qc",
                    stream_key="sbe45.temperature",
                    sensor_key="sbe45",
                    parameter_key="temperature",
                    native_unit="degF",
                    canonical_unit="degC",
                    scale=5 / 9,
                    offset=-32 * 5 / 9,
                    interpolation="linear",
                    max_gap_seconds=5,
                )
            ],
        ),
    )

    stream = result["streams"][0]
    assert stream["is_default"] is True
    assert stream["sampling_rate_hz"] == pytest.approx(1.0)
    assert [row[1] for row in stream["observations"]] == pytest.approx([0.0, 1.0])
    assert stream["native_unit"] == "degF"
    assert stream["metadata"]["conversion"] == {
        "scale": pytest.approx(5 / 9), "offset": pytest.approx(-32 * 5 / 9),
    }
    assert stream["metadata"]["unit_provenance"] == {
        "registry": "pelagia.telemetry_units",
        "registry_version": "1",
        "declared_native_unit": "degF",
        "declared_canonical_unit": "degC",
        "native_unit": "degF",
        "canonical_unit": "degC",
        "scale": pytest.approx(5 / 9),
        "offset": pytest.approx(-32 * 5 / 9),
    }
    assert result["source"]["metadata"]["canonical_timezone"] == "UTC"
    assert result["source"]["metadata"]["unit_registry_version"] == "1"
    assert result["asset"]["checksum"]
    assert store.get_store(result["source"]["source_payload_key"]) == source.read_bytes()


def test_csv_ingestion_rejects_duplicate_stream_timestamps(tmp_path):
    source = tmp_path / "duplicate.csv"
    source.write_text(
        "time,temp\n2026-08-21T00:00:00Z,1\n2026-08-21T00:00:00Z,2\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate timestamp"):
        TelemetryIngestionService(CapturingRepository(), MemoryBlobStore()).import_csv(
            source,
            project_id="project-1",
            run_id="run-1",
            spec=TelemetryCsvSpec(
                timestamp_column="time",
                streams=[
                    TelemetryColumn(
                        column="temp", stream_key="temp", sensor_key="sensor",
                        parameter_key="temperature", native_unit="degC", canonical_unit="degC",
                    )
                ],
            ),
        )


@pytest.mark.parametrize(
    "native_unit, canonical_unit, scale, offset, message",
    [
        ("degC", "m", 1.0, 0.0, "Cannot convert telemetry unit"),
        ("degF", "degC", 1.0, 0.0, "must use scale"),
        ("mystery-unit", "degC", 1.0, 0.0, "Unsupported telemetry native unit"),
    ],
)
def test_csv_ingestion_rejects_invalid_unit_mappings(
    tmp_path, native_unit, canonical_unit, scale, offset, message,
):
    source = tmp_path / "units.csv"
    source.write_text("time,value\n2026-08-21T00:00:00Z,1\n", encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        TelemetryIngestionService(CapturingRepository(), MemoryBlobStore()).import_csv(
            source,
            project_id="project-1",
            run_id="run-1",
            spec=TelemetryCsvSpec(
                timestamp_column="time",
                streams=[
                    TelemetryColumn(
                        column="value", stream_key="value", sensor_key="sensor",
                        parameter_key="value", native_unit=native_unit,
                        canonical_unit=canonical_unit, scale=scale, offset=offset,
                    )
                ],
            ),
        )


def test_csv_ingestion_normalizes_registered_unit_aliases(tmp_path):
    source = tmp_path / "aliases.csv"
    source.write_text("time,temp\n2026-08-21T00:00:00Z,20\n", encoding="utf-8")

    result = TelemetryIngestionService(CapturingRepository(), MemoryBlobStore()).import_csv(
        source,
        project_id="project-1",
        run_id="run-1",
        spec=TelemetryCsvSpec(
            timestamp_column="time",
            streams=[
                TelemetryColumn(
                    column="temp", stream_key="temperature", sensor_key="sensor",
                    parameter_key="temperature", native_unit="°C", canonical_unit="K",
                    scale=1.0, offset=273.15,
                )
            ],
        ),
    )

    assert result["parameters"][0]["canonical_unit"] == "K"
    assert result["streams"][0]["native_unit"] == "degC"
    assert result["streams"][0]["metadata"]["unit_provenance"]["declared_native_unit"] == "°C"


def test_csv_ingestion_uses_durable_snapshot_if_source_changes_before_copy(tmp_path):
    source = tmp_path / "mutable.csv"
    source.write_text("time,temp\n2026-08-21T00:00:00Z,1\n", encoding="utf-8")

    class MutatingRepository(CapturingRepository):
        def ingest_telemetry(self, **payload):
            source.write_text("time,temp\n2026-08-21T00:00:00Z,2\n", encoding="utf-8")
            return super().ingest_telemetry(**payload)

    result = TelemetryIngestionService(MutatingRepository(), MemoryBlobStore()).import_csv(
        source,
        project_id="project-1",
        run_id="run-1",
        spec=TelemetryCsvSpec(
            timestamp_column="time",
            streams=[
                TelemetryColumn(
                    column="temp", stream_key="temp", sensor_key="sensor",
                    parameter_key="temperature", native_unit="degC", canonical_unit="degC",
                )
            ],
        ),
    )
    assert result["streams"][0]["observations"][0][1] == 1.0


class LookupRepository:
    def __init__(self, stream, previous, following):
        self.telemetry = self
        self.stream = stream
        self.previous = previous
        self.following = following

    def list_telemetry_streams(self, **_kwargs):
        return [self.stream]

    def telemetry_observations_around(self, **_kwargs):
        return {"previous": self.previous, "next": self.following}

    def list_telemetry_observations(self, **_kwargs):
        return [row for row in (self.previous, self.following) if row is not None]


class ObservationRepository(LookupRepository):
    def __init__(self, stream, observations):
        self.telemetry = self
        self.stream = stream
        self.observations = observations

    def telemetry_observations_around(self, **_kwargs):
        target = _kwargs["observed_at"]
        before = [row for row in self.observations if row["observed_at"] <= target]
        after = [row for row in self.observations if row["observed_at"] > target]
        excluded = set(_kwargs.get("excluded_qc_flags") or ())
        valid_before = [row for row in before if row.get("qc_flag") not in excluded]
        valid_after = [row for row in after if row.get("qc_flag") not in excluded]
        return {
            "previous": max(before, key=lambda row: row["observed_at"], default=None),
            "next": min(after, key=lambda row: row["observed_at"], default=None),
            "previous_valid": max(valid_before, key=lambda row: row["observed_at"], default=None),
            "next_valid": min(valid_after, key=lambda row: row["observed_at"], default=None),
        }

    def list_telemetry_observations(self, **_kwargs):
        return list(self.observations)


def _stream(**overrides):
    return {
        "id": 1,
        "public_id": "stream-public",
        "project_id": "project-1",
        "run_id": "run-1",
        "parameter_key": "temperature",
        "canonical_unit": "degC",
        "stream_key": "sensor.temperature",
        "sensor_key": "sensor",
        "interpolation": "linear",
        "max_gap": timedelta(seconds=5),
        "is_default": True,
        "metadata": {},
        **overrides,
    }


def test_linear_lookup_and_maximum_gap_semantics():
    start = datetime(2026, 8, 21, tzinfo=timezone.utc)
    previous = {"observed_at": start, "value": 10.0, "qc_flag": 0}
    following = {"observed_at": start + timedelta(seconds=2), "value": 14.0, "qc_flag": 0}
    resolver = TelemetryResolver(LookupRepository(_stream(), previous, following))

    result = resolver.at(
        project_id="project-1", run_id="run-1", observed_at=start + timedelta(seconds=1)
    )["telemetry"]["temperature"]
    assert result["value"] == pytest.approx(12.0)
    assert result["gap_seconds"] == 2.0

    resolver = TelemetryResolver(
        LookupRepository(_stream(max_gap=timedelta(seconds=1)), previous, following)
    )
    result = resolver.at(
        project_id="project-1", run_id="run-1", observed_at=start + timedelta(seconds=1)
    )["telemetry"]["temperature"]
    assert result["value"] is None
    assert result["missing_reason"] == "gap_exceeded"


def test_longitude_linear_lookup_wraps_across_dateline():
    start = datetime(2026, 8, 21, tzinfo=timezone.utc)
    stream = _stream(
        parameter_key="gps.longitude", canonical_unit="degree_east",
        metadata={"circular_period": 360.0, "circular_minimum": -180.0},
    )
    repository = LookupRepository(
        stream,
        {"observed_at": start, "value": 179.0, "qc_flag": None},
        {"observed_at": start + timedelta(seconds=2), "value": -179.0, "qc_flag": None},
    )

    value = TelemetryResolver(repository).at(
        project_id="project-1", run_id="run-1", observed_at=start + timedelta(seconds=1)
    )["telemetry"]["gps.longitude"]["value"]
    assert abs(value) == pytest.approx(180.0)


def test_bulk_alignment_matches_scalar_linear_value():
    start = datetime(2026, 8, 21, tzinfo=timezone.utc)
    repository = LookupRepository(
        _stream(),
        {"observed_at": start, "value": 10.0, "qc_flag": 0},
        {"observed_at": start + timedelta(seconds=2), "value": 14.0, "qc_flag": 0},
    )
    resolver = TelemetryResolver(repository)
    target = start + timedelta(seconds=1)

    scalar = resolver.at(
        project_id="project-1", run_id="run-1", observed_at=target,
    )["telemetry"]["temperature"]["value"]
    bulk = resolver.align_timestamps(
        project_id="project-1", run_id="run-1", observed_at=[target], chunk_size=1,
    )[0]["telemetry"]["temperature"]["value"]

    assert bulk == pytest.approx(scalar)


def test_bulk_alignment_preserves_qc_exclusion_barriers():
    start = datetime(2026, 8, 21, tzinfo=timezone.utc)
    repository = LookupRepository(
        _stream(metadata={"excluded_qc_flags": [4]}),
        {"observed_at": start, "value": 10.0, "qc_flag": 4},
        {"observed_at": start + timedelta(seconds=2), "value": 14.0, "qc_flag": 0},
    )
    resolver = TelemetryResolver(repository)

    aligned = resolver.align_timestamps(
        project_id="project-1", run_id="run-1",
        observed_at=[start, start + timedelta(seconds=1)],
    )

    assert aligned[0]["telemetry"]["temperature"]["missing_reason"] == "qc_excluded"
    assert aligned[1]["telemetry"]["temperature"]["missing_reason"] == "qc_excluded"


@pytest.mark.parametrize("method", ["previous", "nearest"])
def test_lookup_skips_excluded_neighbor_for_selection(method):
    start = datetime(2026, 8, 21, tzinfo=timezone.utc)
    stream = _stream(interpolation=method, metadata={"excluded_qc_flags": [4]})
    repository = ObservationRepository(stream, [
        {"observed_at": start, "value": 10.0, "qc_flag": 0},
        {"observed_at": start + timedelta(seconds=1), "value": 99.0, "qc_flag": 4},
        {"observed_at": start + timedelta(seconds=3), "value": 30.0, "qc_flag": 0},
    ])
    result = TelemetryResolver(repository).at(
        project_id="project-1", run_id="run-1",
        observed_at=start + timedelta(seconds=2),
    )["telemetry"]["temperature"]
    assert result["value"] == (10.0 if method == "previous" else 30.0)


def test_bulk_selection_skips_excluded_neighbors_like_scalar():
    start = datetime(2026, 8, 21, tzinfo=timezone.utc)
    stream = _stream(interpolation="previous", metadata={"excluded_qc_flags": [4]})
    repository = ObservationRepository(stream, [
        {"observed_at": start, "value": 10.0, "qc_flag": 0},
        {"observed_at": start + timedelta(seconds=1), "value": 99.0, "qc_flag": 4},
        {"observed_at": start + timedelta(seconds=3), "value": 30.0, "qc_flag": 0},
    ])
    resolver = TelemetryResolver(repository)
    target = start + timedelta(seconds=2)
    scalar = resolver.at(project_id="project-1", run_id="run-1", observed_at=target)
    bulk = resolver.align_timestamps(project_id="project-1", run_id="run-1", observed_at=[target])
    assert bulk[0]["telemetry"]["temperature"]["value"] == scalar["telemetry"]["temperature"]["value"] == 10.0
