from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

from Pelagia.config import CoreConfig
from Pelagia.services.context import AppContext
import Pelagia.cli.app as cli_module


class _TelemetryRepository:
    telemetry: "_TelemetryRepository"

    def __init__(self) -> None:
        self.telemetry = self
        self.import_payload: dict | None = None

    def close(self) -> None:
        pass

    def get_project_by_key(self, project_key: str):
        return {"id": "project-1", "project_key": project_key}

    def get_project(self, project_id: str):
        return {"id": project_id, "kvstore_root_path": None}

    def ingest_telemetry(self, **payload):
        self.import_payload = payload
        return {"source": {"id": "source-1"}, "stream_count": len(payload["streams"])}

    def list_telemetry_streams(self, **_kwargs):
        return [
            {
                "id": 1,
                "public_id": "stream-1",
                "project_id": "project-1",
                "parameter_key": "temperature",
                "canonical_unit": "degC",
                "stream_key": "ctd.temperature",
                "sensor_key": "ctd-01",
                "interpolation": "none",
                "is_default": True,
                "metadata": {},
            }
        ]

    def telemetry_observations_around(self, **_kwargs):
        return {
            "previous": {
                "observed_at": datetime(2026, 8, 21, 18, 4, 5, tzinfo=timezone.utc),
                "value": 7.25,
                "qc_flag": 1,
            },
            "next": None,
        }


def _install_context(monkeypatch) -> _TelemetryRepository:
    repository = _TelemetryRepository()
    class BlobStore:
        def __init__(self):
            self.payloads = {}

        def put_store(self, payload):
            payload = bytes(payload)
            key = hashlib.sha256(payload).hexdigest()
            self.payloads[key] = payload
            return key

        def get_store(self, key):
            return self.payloads[key]

    context = AppContext(config=CoreConfig(), repository=repository, kvstore=BlobStore())
    monkeypatch.setattr(cli_module, "_context_from_options", lambda *args, **kwargs: context)
    return repository


def test_cli_import_telemetry_reads_mapping_and_records_source(tmp_path, monkeypatch):
    repository = _install_context(monkeypatch)
    source = tmp_path / "ship.csv"
    source.write_text("time_utc,temperature_c\n2026-08-21T18:04:05Z,7.25\n", encoding="utf-8")
    mapping = tmp_path / "mapping.json"
    mapping.write_text(
        json.dumps(
            {
                "timestamp_column": "time_utc",
                "streams": [
                    {
                        "column": "temperature_c",
                        "stream_key": "ctd.temperature",
                        "sensor_key": "ctd-01",
                        "parameter_key": "temperature",
                        "native_unit": "degC",
                        "canonical_unit": "degC",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        cli_module.app,
        [
            "import-telemetry", str(source), "run-1", "--mapping", str(mapping),
            "--project-key", "survey", "--collections", "cruise-17,calibrated",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == {"source": {"id": "source-1"}, "stream_count": 1}
    assert repository.import_payload is not None
    assert repository.import_payload["project_id"] == "project-1"
    assert repository.import_payload["asset"]["collections"] == ["cruise-17", "calibrated"]


def test_cli_lookup_telemetry_normalizes_utc_timestamp(monkeypatch):
    _install_context(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "lookup-telemetry", "run-1", "--observed-at", "2026-08-21T10:04:05-08:00",
            "--parameters", "temperature", "--project-key", "survey",
        ],
    )

    assert result.exit_code == 0, result.output
    body = json.loads(result.output)
    assert body["observed_at"] == "2026-08-21T18:04:05+00:00"
    assert body["telemetry"]["temperature"]["method"] == "exact"
    assert body["telemetry"]["temperature"]["value"] == 7.25


def test_cli_lookup_telemetry_rejects_naive_timestamp(monkeypatch):
    _install_context(monkeypatch)

    result = CliRunner().invoke(
        cli_module.app,
        [
            "lookup-telemetry", "run-1", "--observed-at", "2026-08-21T18:04:05",
            "--project-key", "survey",
        ],
    )

    assert result.exit_code != 0
