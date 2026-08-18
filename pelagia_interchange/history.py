from __future__ import annotations

import json
import os
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
from uuid import uuid4

from .constants import LIBRARY_VERSION
from .exceptions import FormatError
from .util import utc_now


@dataclass(slots=True)
class History:
    path: Path

    def append(
        self,
        *,
        operation: str,
        parameters: Mapping[str, Any] | None = None,
        inputs: Iterable[Mapping[str, Any]] = (),
        outputs: Iterable[Mapping[str, Any]] = (),
        status: str = "success",
        message: str | None = None,
        operator: str | None = None,
        git_commit: str | None = None,
        software: str = "pelagia_interchange",
        software_version: str = LIBRARY_VERSION,
        environment: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if re.fullmatch(r"[a-z][a-z0-9_]*", operation) is None:
            raise ValueError("operation must be a non-empty machine-readable token")
        event = {
            "event_schema_version": "1",
            "event_uuid": str(uuid4()),
            "timestamp": utc_now(),
            "operation": operation,
            "software": software,
            "software_version": software_version,
            "git_commit": git_commit,
            "operator": operator,
            "parameters": dict(parameters or {}),
            "inputs": list(inputs),
            "outputs": list(outputs),
            "environment": dict(environment or {"python": platform.python_version(), "platform": platform.platform()}),
            "status": status,
            "message": message,
        }
        encoded = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        return event

    def __iter__(self) -> Iterator[dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if not line.strip():
                        continue
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise FormatError(f"invalid history event at line {line_number}: {exc}") from exc
                    if not isinstance(value, dict):
                        raise FormatError(f"history event at line {line_number} is not an object")
                    yield value
        except OSError as exc:
            raise FormatError(f"cannot read history: {exc}") from exc
