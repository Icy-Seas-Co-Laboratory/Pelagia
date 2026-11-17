from __future__ import annotations

import logging
import json

import pytest

from Pelagia.config import CoreConfig
from Pelagia.observability import JsonlLogHandler, OutputLogger, configure_core_logging


class _OutputStore:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def append(self, record: dict, *, stream: str = "logs") -> dict:
        stored = {**record, "stream": stream}
        self.records.append(stored)
        return {"id": len(self.records), **stored}


def test_legacy_logger_retains_explicit_project_scope():
    from Pelagia.observability import DatabaseLogger

    class Repository:
        def append_log(self, **record):
            return record

    unscoped = DatabaseLogger(Repository())
    scoped = unscoped.for_project("project-1")
    assert scoped.info("job.output")["project_id"] == "project-1"
    assert unscoped.project_id is None


class _FailingStore:
    def append(self, record: dict, *, stream: str = "logs") -> dict:
        raise OSError("disk full")


def test_output_logger_keeps_structured_context_and_scoped_project():
    store = _OutputStore()
    logger = OutputLogger(store).for_project("project-1")

    row = logger.info(
        "processing.complete", "Finished café", job_id="job-1",
        payload={"frame_count": 2}, logger="pelagia.processing",
    )

    assert row is not None
    record = store.records[0]
    assert record["project_id"] == "project-1"
    assert record["message"] == "Finished café"
    assert record["payload"]["frame_count"] == 2
    assert record["payload"]["hostname"]
    assert record["stream"] == "logs"


def test_output_logger_records_failure_without_hiding_original_error():
    store = _OutputStore()
    logger = OutputLogger(store)

    with pytest.raises(ValueError, match="bad input"):
        with logger.timed("processing.work", payload={"source": "test"}):
            raise ValueError("bad input")

    record = store.records[0]
    assert record["event_type"] == "processing.work.failed"
    assert record["level"] == "error"
    assert record["payload"]["error_type"] == "ValueError"
    assert record["payload"]["error_message"] == "bad input"


def test_stdlib_handler_serializes_unicode_and_exception_text():
    store = _OutputStore()
    handler = JsonlLogHandler(store)
    logger = logging.getLogger("pelagia.test.output")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        try:
            raise RuntimeError("échec")
        except RuntimeError:
            logger.exception("Could not write %s", "μ-data")
    finally:
        logger.handlers.clear()

    record = store.records[0]
    assert record["message"] == "Could not write μ-data"
    assert "RuntimeError: échec" in record["payload"]["exception"]


def test_stdlib_handler_writes_full_record_to_stderr_when_output_fails(capsys):
    handler = JsonlLogHandler(_FailingStore())
    logger = logging.getLogger("pelagia.test.output-failure")
    logger.handlers[:] = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False
    previous = logging.raiseExceptions
    logging.raiseExceptions = False
    try:
        logger.error("could not save %s", "record")
    finally:
        logging.raiseExceptions = previous
        logger.handlers.clear()

    record = json.loads(capsys.readouterr().err)
    assert record["message"] == "could not save record"
    assert record["logger"] == "pelagia.test.output-failure"
    assert record["payload"]["output_write_error"]["message"] == "disk full"


def test_core_logging_falls_back_on_store_initialization_and_is_idempotent(monkeypatch, capsys):
    from Pelagia.storage import output

    class UnavailableStore:
        def __init__(self, _config):
            raise OSError("read-only filesystem")

    config = CoreConfig()
    config.logging.console = False
    monkeypatch.setattr(output, "OutputStore", UnavailableStore)
    logger = configure_core_logging(config, force=True)
    logger.warning("still visible")
    fallback = json.loads(capsys.readouterr().err.splitlines()[0])
    assert fallback["event_type"] == "logging.initialization_failed"
    assert fallback["payload"]["output_write_error"]["message"] == "read-only filesystem"

    store = _OutputStore()
    configured = configure_core_logging(config, force=True, output_store=store)
    handler_count = len(configured.handlers)
    assert configure_core_logging(config, output_store=store) is configured
    assert len(configured.handlers) == handler_count
