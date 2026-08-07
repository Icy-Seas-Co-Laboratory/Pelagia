from __future__ import annotations

import tomllib
from copy import deepcopy
from datetime import date, datetime
from functools import lru_cache
from importlib.resources import files
from typing import Any


DEFAULT_TAXONOMY_FILENAME = "taxonomy_0.1.1.toml"


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


@lru_cache(maxsize=1)
def _default_taxonomy() -> dict[str, Any]:
    resource = files("Pelagia").joinpath("assets", DEFAULT_TAXONOMY_FILENAME)
    data = _json_value(tomllib.loads(resource.read_text(encoding="utf-8")))
    metadata = data.get("vocabulary") or {}
    nodes = (data.get("taxonomy") or {}).get("nodes") or []
    ids = [node.get("id") for node in nodes]
    if not metadata.get("id") or not metadata.get("version"):
        raise RuntimeError("Default taxonomy vocabulary identity is incomplete.")
    if any(not value for value in ids) or len(set(ids)) != len(ids):
        raise RuntimeError("Default taxonomy contains missing or duplicate concept IDs.")
    known_ids = set(ids)
    for node in nodes:
        if node.get("parent_id") and node["parent_id"] not in known_ids:
            raise RuntimeError(f"Default taxonomy has unknown parent {node['parent_id']!r}.")
    return data


def default_taxonomy_dictionary() -> dict[str, Any]:
    """Return the packaged taxonomy and its project-label import candidates."""

    data = deepcopy(_default_taxonomy())
    metadata = data["vocabulary"]
    nodes = data["taxonomy"]["nodes"]
    return {
        "key": f"{metadata['id']}@{metadata['version']}",
        "filename": DEFAULT_TAXONOMY_FILENAME,
        "vocabulary": metadata,
        "sources": data.get("sources") or [],
        "labels": nodes,
        "selectable_count": sum(bool(node.get("selectable", True)) for node in nodes),
    }

