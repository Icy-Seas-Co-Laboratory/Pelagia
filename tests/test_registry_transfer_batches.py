from __future__ import annotations

import pytest

from Pelagia.services import registry_transfer
from Pelagia.services.registry_transfer import _load_assets


def _asset(asset_id: str, payload: bytes | None, *, external_uri: str | None = None) -> dict[str, object]:
    return {"asset_id": asset_id, "content_sha256": f"hash-{asset_id}", "payload": payload,
            "external_uri": external_uri, "encoding": "raw", "media_type": "application/octet-stream",
            "shape_json": "[]", "dtype": "uint8", "original_filename": f"{asset_id}.bin",
            "metadata_json": "{}", "created_at": "2026-01-01T00:00:00Z"}


class _Cursor:
    def __init__(self) -> None:
        self.executions: list[tuple[str, tuple[object, ...]]] = []

    def execute(self, statement: str, values: tuple[object, ...]) -> None:
        self.executions.append((statement, values))


class _Store:
    def __init__(self) -> None:
        self.put_batches: list[list[bytes]] = []

    def put_many(self, payloads: list[bytes]) -> list[str]:
        batch = list(payloads)
        self.put_batches.append(batch)
        return [f"ref-{len(self.put_batches)}-{index}" for index in range(len(batch))]

class _Repository:
    def __init__(self, hot_store: object) -> None:
        self.hot_store = hot_store


def _load(monkeypatch, assets: list[dict[str, object]], store: object) -> _Cursor:
    monkeypatch.setattr(registry_transfer, "_rows", lambda source, table: assets)
    cursor = _Cursor()
    _load_assets(_Repository(store), cursor, "pelagia", "workspace", object())
    return cursor


def test_registry_assets_batch_writes_before_inserting(monkeypatch):
    monkeypatch.setattr(registry_transfer, "_REGISTRY_ASSET_BATCH_ITEMS", 2)
    store = _Store()
    cursor = _load(monkeypatch, [
        _asset("first", b"a"), _asset("external", None, external_uri="s3://bucket/object"),
        _asset("second", b"bb"), _asset("third", b"ccc"),
    ], store)

    assert store.put_batches == [[b"a", b"bb"], [b"ccc"]]
    assert [values[1] for _, values in cursor.executions] == ["first", "external", "second", "third"]
    assert [values[3] for _, values in cursor.executions] == ["ref-1-0", None, "ref-1-1", "ref-2-0"]
    assert cursor.executions[1][1][4:6] == (None, "s3://bucket/object")


def test_registry_assets_flushes_before_exceeding_byte_cap(monkeypatch):
    monkeypatch.setattr(registry_transfer, "_REGISTRY_ASSET_BATCH_ITEMS", 250)
    monkeypatch.setattr(registry_transfer, "_REGISTRY_ASSET_BATCH_BYTES", 3)
    store = _Store()
    _load(monkeypatch, [_asset("one", b"aa"), _asset("two", b"bb"), _asset("large", b"four")], store)
    assert store.put_batches == [[b"aa"], [b"bb"], [b"four"]]


def test_registry_assets_write_failure_inserts_no_current_batch_rows(monkeypatch):
    class _FailingStore(_Store):
        def put_many(self, payloads: list[bytes]) -> list[str]:
            raise RuntimeError("write failed")

    store = _FailingStore()
    cursor = _Cursor()
    monkeypatch.setattr(registry_transfer, "_rows", lambda source, table: [_asset("first", b"payload")])
    with pytest.raises(RuntimeError, match="write failed"):
        _load_assets(_Repository(store), cursor, "pelagia", "workspace", object())
    assert store.put_batches == []
    assert cursor.executions == []

