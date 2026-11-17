from __future__ import annotations

import json

import pytest

pytest.importorskip("typer")
from typer.testing import CliRunner

import Pelagia.cli.app as cli_module
from Pelagia.config import CoreConfig
from Pelagia.storage.output import OutputStore
from Pelagia.storage.postgres import PostgresRepository


def test_cleanup_output_compresses_without_connecting_to_postgres(tmp_path, monkeypatch):
    config = CoreConfig()
    config.logging.log_path = tmp_path
    config.logging.max_bytes = 200
    store = OutputStore(config.logging)
    for index in range(3):
        store.append({"event_type": "message", "message": str(index), "project_id": "p1"})
    monkeypatch.setattr(cli_module, "_config_from_options", lambda *args: config)

    def no_database(*args, **kwargs):
        raise AssertionError("output maintenance must not open PostgreSQL")

    monkeypatch.setattr(PostgresRepository, "connect", no_database)
    runner = CliRunner()
    preview = runner.invoke(cli_module.app, ["cleanup-logs", "--dry-run"])
    assert preview.exit_code == 0, preview.output
    assert json.loads(preview.stdout)["dry_run"] is True
    assert not list(tmp_path.glob("*.zst"))
    result = runner.invoke(cli_module.app, ["cleanup-logs", "--no-dry-run"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["compressed"] > 0
    assert len(store.list_logs(project_id="p1")) == 3
