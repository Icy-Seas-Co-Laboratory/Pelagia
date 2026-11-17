"""Structured operational output that does not depend on PostgreSQL."""

from __future__ import annotations

import logging
import json
import os
import socket
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Iterator, Protocol

from .config import CoreConfig
from .utils.serialization import json_ready


CORE_LOGGER_NAME = "pelagia"
_CORE_LOGGING_CONFIGURED = False
_CORE_LOGGING_SIGNATURE: tuple[object, ...] | None = None


class OutputAppender(Protocol):
    def append(self, record: dict[str, Any], *, stream: str = "logs") -> dict[str, Any]: ...


class JsonlLogHandler(logging.Handler):
    """Bridge stdlib log records into the shared JSONL output store."""

    def __init__(self, output_store: OutputAppender) -> None:
        super().__init__()
        self.output_store = output_store

    def emit(self, record: logging.LogRecord) -> None:
        payload = _stdlib_record_payload(record, self)
        try:
            self.output_store.append(payload)
        except Exception as exc:
            # Never use logging here: this handler may be the failed sink and
            # logging.raiseExceptions can otherwise discard the record.
            write_output_fallback(payload, exc)


def configure_core_logging(config: CoreConfig, *, force: bool = False, output_store: OutputAppender | None = None) -> logging.Logger:
    """Configure JSONL operational logging and an optional human stderr sink."""
    global _CORE_LOGGING_CONFIGURED, _CORE_LOGGING_SIGNATURE
    logger = logging.getLogger(CORE_LOGGER_NAME)
    signature = (config.logging.log_path, config.logging.file_name, config.logging.level,
                 config.logging.console, config.logging.max_bytes, config.logging.backup_count,
                 config.logging.compression_level, config.logging.rotation_interval_seconds,
                 config.logging.max_total_bytes, None if output_store is None else id(output_store))
    if _CORE_LOGGING_CONFIGURED and _CORE_LOGGING_SIGNATURE == signature and not force:
        return logger
    level = _coerce_log_level(config.logging.level)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    formatter = logging.Formatter("%(asctime)s %(levelname)s [%(process)d] %(name)s - %(message)s")
    try:
        if output_store is None:
            from .storage.output import OutputStore
            output_store = OutputStore(config.logging)
        output_handler = JsonlLogHandler(output_store)
        output_handler.setLevel(level)
        output_handler.setFormatter(formatter)
        logger.addHandler(output_handler)
    except Exception as exc:
        fallback = logging.StreamHandler()
        fallback.setFormatter(formatter)
        fallback.setLevel(level)
        logger.addHandler(fallback)
        write_output_fallback({
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "event_type": "logging.initialization_failed",
            "message": "Unable to initialize JSONL logging",
            "level": "error",
            "logger": CORE_LOGGER_NAME,
            "payload": _base_payload({"configured_log_path": str(config.logging.log_path)}),
        }, exc)
    if config.logging.console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        logger.addHandler(stream_handler)
    _CORE_LOGGING_CONFIGURED = True
    _CORE_LOGGING_SIGNATURE = signature
    return logger


def get_core_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(CORE_LOGGER_NAME if not name else f"{CORE_LOGGER_NAME}.{name}")


def _coerce_log_level(value: str) -> int:
    return int(getattr(logging, str(value).upper(), logging.INFO))


class OutputLogger:
    """Best-effort structured output writer with an optional project scope."""

    def __init__(self, output_store: OutputAppender, *, logger_name: str = CORE_LOGGER_NAME, project_id: str | None = None) -> None:
        self.output_store = output_store
        self.logger_name = logger_name
        self.project_id = project_id
        self.core_logger = logging.getLogger(logger_name)

    def for_project(self, project_id: str | None) -> "OutputLogger":
        if project_id is None or project_id == self.project_id:
            return self
        return type(self)(self.output_store, logger_name=self.logger_name, project_id=str(project_id))

    def log(self, *, event_type: str, message: str | None = None, level: str = "info", project_id: str | None = None,
            run_id: str | None = None, asset_id: str | None = None, job_id: str | None = None,
            worker_id: str | None = None, request_id: str | None = None, duration_ms: float | None = None,
            payload: dict[str, Any] | None = None, logger: str = CORE_LOGGER_NAME) -> dict[str, Any] | None:
        record = {
            "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type, "message": message, "level": str(level).lower(), "logger": logger,
            "project_id": project_id or self.project_id, "run_id": run_id, "asset_id": asset_id,
            "job_id": job_id, "worker_id": worker_id, "request_id": request_id,
            "duration_ms": duration_ms, "payload": _base_payload(payload),
        }
        try:
            return self.output_store.append(record)
        except Exception as exc:
            write_output_fallback(record, exc)
            return None

    def debug(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="debug", **kwargs)

    def info(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="info", **kwargs)

    def warning(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="warning", **kwargs)

    def error(self, event_type: str, message: str | None = None, **kwargs) -> dict[str, Any] | None:
        return self.log(event_type=event_type, message=message, level="error", **kwargs)

    @contextmanager
    def timed(self, event_type: str, *, message: str | None = None, level: str = "info", payload: dict[str, Any] | None = None, **kwargs) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:
            error_payload = dict(payload or {})
            error_payload.update(error_type=type(exc).__name__, error_message=str(exc))
            self.log(event_type=f"{event_type}.failed", message=message or str(exc), level="error", duration_ms=(time.perf_counter() - started) * 1000, payload=error_payload, **kwargs)
            raise
        else:
            self.log(event_type=event_type, message=message, level=level, duration_ms=(time.perf_counter() - started) * 1000, payload=payload, **kwargs)


class DatabaseLogger(OutputLogger):
    """Compatibility adapter for test and plugin callers that inject a repository."""

    def __init__(self, repository, *, logger_name: str = CORE_LOGGER_NAME) -> None:
        self.repository = repository
        self.logger_name = logger_name
        self.project_id = None
        self.core_logger = logging.getLogger(logger_name)

    def for_project(self, project_id: str | None) -> "DatabaseLogger":
        if project_id is None or project_id == self.project_id:
            return self
        scoped = type(self)(self.repository, logger_name=self.logger_name)
        scoped.project_id = str(project_id)
        return scoped

    def log(self, **kwargs) -> dict[str, Any] | None:  # type: ignore[override]
        kwargs["project_id"] = kwargs.get("project_id") or self.project_id
        kwargs["payload"] = _base_payload(kwargs.get("payload"))
        try:
            return self.repository.append_log(**kwargs)
        except Exception as exc:
            write_output_fallback({"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), **kwargs}, exc)
            return None


def _base_payload(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    base = {"hostname": socket.gethostname(), "pid": os.getpid()}
    base.update(payload or {})
    return json_ready(base)


def _stdlib_record_payload(record: logging.LogRecord, handler: logging.Handler) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "created_at": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
        "level": record.levelname.lower(),
        "logger": record.name,
        "event_type": "log.record",
        "message": record.getMessage(),
        "payload": {"hostname": socket.gethostname(), "pid": record.process, "thread": record.threadName},
    }
    if record.exc_info:
        payload["payload"]["exception"] = handler.format(record)
    return payload


def write_output_fallback(record: dict[str, Any], error: BaseException) -> None:
    """Emit an intact record directly to stderr when its normal sink fails."""
    try:
        fallback = dict(json_ready(record))
        payload = dict(fallback.get("payload") or {})
        payload["output_write_error"] = {"type": type(error).__name__, "message": str(error)}
        fallback["payload"] = payload
        sys.stderr.write(json.dumps(fallback, ensure_ascii=False, separators=(",", ":")) + "\n")
        sys.stderr.flush()
    except Exception:
        pass
