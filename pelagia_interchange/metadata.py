from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path
from typing import Any, Mapping

from .exceptions import FormatError

_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(value: str) -> str:
    return value if _BARE_KEY.match(value) else _string(value)


def _string(value: str) -> str:
    return '"' + value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _string(value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return repr(value)
    if isinstance(value, (datetime, date, time)):
        return value.isoformat()
    if isinstance(value, list) and all(not isinstance(item, (dict, list)) for item in value):
        return "[" + ", ".join(_scalar(item) for item in value) + "]"
    raise TypeError(f"unsupported TOML scalar: {type(value).__name__}")


def dumps_toml(data: Mapping[str, Any]) -> str:
    """Serialize the conservative TOML subset used by format metadata."""
    lines: list[str] = []

    def array_tables(values: list[Any], path: tuple[str, ...]) -> None:
        for value in values:
            if not isinstance(value, dict):
                raise TypeError("mixed arrays of tables are unsupported")
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[[" + ".".join(_key(part) for part in path) + "]]" )
            children: dict[str, Any] = {}
            for item_key, item_value in value.items():
                if isinstance(item_value, dict) or (isinstance(item_value, list) and any(isinstance(x, dict) for x in item_value)):
                    children[item_key] = item_value
                else:
                    lines.append(f"{_key(item_key)} = {_scalar(item_value)}")
            for item_key, item_value in children.items():
                child_path = path + (item_key,)
                if isinstance(item_value, dict):
                    table(item_value, child_path)
                else:
                    array_tables(item_value, child_path)

    def table(mapping: Mapping[str, Any], path: tuple[str, ...]) -> None:
        scalars = {key: value for key, value in mapping.items() if not isinstance(value, (dict, list)) or (isinstance(value, list) and all(not isinstance(item, dict) for item in value))}
        nested = {key: value for key, value in mapping.items() if isinstance(value, dict)}
        arrays = {key: value for key, value in mapping.items() if isinstance(value, list) and any(isinstance(item, dict) for item in value)}
        if path:
            if lines and lines[-1] != "":
                lines.append("")
            lines.append("[" + ".".join(_key(part) for part in path) + "]")
        for key, value in scalars.items():
            lines.append(f"{_key(key)} = {_scalar(value)}")
        for key, value in nested.items():
            table(value, path + (key,))
        for key, values in arrays.items():
            array_tables(values, path + (key,))

    table(data, ())
    return "\n".join(lines).rstrip() + "\n"


@dataclass(slots=True)
class Metadata:
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> str | None:
        value = self.data.get("dataset", {}).get("title")
        return str(value) if value is not None else None

    @property
    def extensions(self) -> dict[str, Any]:
        return self.data.setdefault("extensions", {})

    def validate(self, *, require_title: bool = False) -> list[str]:
        errors: list[str] = []
        dataset = self.data.get("dataset")
        if not isinstance(dataset, dict):
            errors.append("metadata requires a [dataset] table")
        elif require_title and not isinstance(dataset.get("title"), str):
            errors.append("metadata requires dataset.title")
        for plural in ("investigators", "instruments", "deployments", "streams", "funding"):
            value = self.data.get(plural, [])
            if not isinstance(value, list) or any(not isinstance(item, dict) for item in value):
                errors.append(f"metadata {plural} must be an array of tables")
        return errors

    def write(self, path: Path) -> None:
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(dumps_toml(self.data), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def read(cls, path: Path) -> "Metadata":
        try:
            return cls(tomllib.loads(path.read_text(encoding="utf-8")))
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise FormatError(f"cannot read metadata {path}: {exc}") from exc


def default_metadata(title: str, description: str = "") -> Metadata:
    return Metadata({
        "schema": {"name": "scientific-image-interchange-metadata", "version": "1"},
        "dataset": {"title": title, "description": description},
        "investigators": [], "instruments": [], "deployments": [], "streams": [], "funding": [],
        "extensions": {},
    })
