from __future__ import annotations

import hashlib
import json
import sqlite3
import zipfile
from datetime import datetime, timezone

import pytest

from Pelagia.services.exports.telemetry import write_telemetry_bundle


PROJECT_ID = "00000000-0000-0000-0000-000000000001"
RUN_ID = "00000000-0000-0000-0000-000000000002"
SOURCE_ID = "00000000-0000-0000-0000-000000000003"
PARAMETER_ID = "00000000-0000-0000-0000-000000000004"
SENSOR_ID = "00000000-0000-0000-0000-000000000005"
STREAM_ID = "00000000-0000-0000-0000-000000000006"


class _Store:
    def __init__(self, payload: bytes):
        self.payload = payload

    def get_store(self, key: str) -> bytes:
        assert key == "stored-source"
        return self.payload


class _TelemetryRepository:
    def __init__(self, payload: bytes):
        checksum = hashlib.sha256(payload).hexdigest()
        self.source = {
            "id": SOURCE_ID, "project_id": PROJECT_ID, "run_id": RUN_ID,
            "format": "delimited", "parser_name": "pelagia.delimited", "parser_version": "1",
            "source_payload_key": "stored-source", "filename": "instrument.csv", "checksum": checksum,
            "size_bytes": len(payload), "metadata": {
                "timestamp_column": "time", "timestamp_format": "iso8601", "source_timezone": "UTC",
            },
        }
        self.stream = {
            "id": 12, "public_id": STREAM_ID, "project_id": PROJECT_ID, "run_id": RUN_ID,
            "source_id": SOURCE_ID, "sensor_id": SENSOR_ID, "parameter_id": PARAMETER_ID,
            "stream_key": "temperature", "native_unit": "degC", "canonical_unit": "K",
            "sampling_rate_hz": 1.0, "interpolation": "none", "max_gap": None, "priority": 100,
            "is_default": True, "qc_scheme": "qartod", "metadata": {"source_column": "temp"},
        }

    def list_telemetry_sources(self, *, project_id, run_id=None):
        assert project_id == PROJECT_ID
        return [self.source] if run_id in (None, RUN_ID) else []

    def list_telemetry_parameters(self, *, project_id):
        return [{"id": PARAMETER_ID, "parameter_key": "temperature", "canonical_unit": "K", "metadata": {}}]

    def list_telemetry_sensors(self, *, project_id):
        return [{"id": SENSOR_ID, "sensor_key": "ctd", "display_name": "CTD", "metadata": {}}]

    def list_telemetry_streams(self, *, project_id, run_id=None):
        return [self.stream] if run_id == RUN_ID else []

    def list_telemetry_observations(self, *, project_id, run_id, stream_id):
        assert stream_id == 12
        return [{"observed_at": datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc), "value": 274.15, "qc_flag": 1}]

    def list_timeline_event_types(self, *, project_id):
        return [{"id": "00000000-0000-0000-0000-000000000007", "event_type_key": "cast", "metadata": {}}]

    def list_timeline_events(self, *, project_id, run_id):
        return [{"id": "00000000-0000-0000-0000-000000000008", "run_id": run_id, "source_id": SOURCE_ID,
                 "start_at": datetime(2026, 1, 2, tzinfo=timezone.utc), "metadata": {}}]


def test_json_telemetry_bundle_preserves_source_and_import_context(tmp_path):
    payload = b"time,temp\n2026-01-02T03:04:05Z,1\n"
    result = write_telemetry_bundle(
        _TelemetryRepository(payload), project_id=PROJECT_ID, selection={"source_ids": [SOURCE_ID]},
        output_root=tmp_path, kvstore=_Store(payload),
    )

    source_root = tmp_path / "products" / "telemetry" / "runs" / RUN_ID / "sources" / SOURCE_ID
    assert (source_root / "source" / "original.bin").read_bytes() == payload
    profile = json.loads((source_root / "import-profile.json").read_text())
    assert profile["source_payload_path"] == "source/original.bin"
    assert profile["streams"][0]["stream_id"] == STREAM_ID
    observation = json.loads((source_root / "observations.ndjson").read_text())
    assert observation == {"observed_at": "2026-01-02T03:04:05+00:00", "qc_flag": 1, "stream_id": STREAM_ID, "value": 274.15}
    assert result["observation_count"] == 1
    assert any(item["path"].endswith("original.bin") for item in result["files"])


def test_sqlite_telemetry_bundle_uses_public_stream_ids(tmp_path):
    payload = b"time,temp\n"
    write_telemetry_bundle(
        _TelemetryRepository(payload), project_id=PROJECT_ID, selection={"run_ids": [RUN_ID]},
        output_root=tmp_path, file_format="sqlite", kvstore=_Store(payload),
    )
    database = tmp_path / "products" / "telemetry" / "runs" / RUN_ID / "sources" / SOURCE_ID / "telemetry.sqlite"
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT id FROM streams").fetchone() == (STREAM_ID,)
        assert connection.execute("SELECT stream_id, value FROM observations").fetchone() == (STREAM_ID, 274.15)


def test_xlsx_telemetry_bundle_is_a_real_workbook(tmp_path):
    payload = b"time,temp\n"
    write_telemetry_bundle(
        _TelemetryRepository(payload), project_id=PROJECT_ID, selection={}, output_root=tmp_path,
        file_format="xlsx", kvstore=_Store(payload),
    )
    workbook = tmp_path / "products" / "telemetry" / "runs" / RUN_ID / "sources" / SOURCE_ID / "telemetry.xlsx"
    with zipfile.ZipFile(workbook) as archive:
        assert "xl/workbook.xml" in archive.namelist()


def test_telemetry_bundle_rejects_cross_project_or_unsupported_format(tmp_path):
    repository = _TelemetryRepository(b"time,temp\n")
    with pytest.raises(ValueError, match="not found"):
        write_telemetry_bundle(repository, project_id=PROJECT_ID, selection={"source_ids": ["other"]}, output_root=tmp_path)
    with pytest.raises(ValueError, match="Unsupported telemetry export format"):
        write_telemetry_bundle(repository, project_id=PROJECT_ID, selection={}, output_root=tmp_path, file_format="csv")
