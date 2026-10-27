from __future__ import annotations

import json
from io import BytesIO
from urllib.error import HTTPError

import pytest

from Pelagia.storage.skerrykv import SkerryKVError, SkerryKVStore


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_put_and_get_use_skerry_object_protocol(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, request.get_method(), request.data, request.headers, timeout))
        if request.get_method() == "PUT":
            return _Response(json.dumps({"hash": "a" * 64, "duplicate": False}).encode())
        return _Response(b"payload")

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", fake_urlopen)
    store = SkerryKVStore("http://skerry.test/", token="secret", timeout_seconds=12)

    key = store.put_store(b"payload")
    assert key == "a" * 64
    assert store.get_store(key) == b"payload"
    assert requests[0][0] == "http://skerry.test/v1/objects"
    assert requests[0][1:3] == ("PUT", b"payload")
    assert requests[0][3]["Authorization"] == "Bearer secret"


@pytest.mark.parametrize("key", ["../x", "A" * 64, "a" * 63])
def test_rejects_non_skerry_hashes(key):
    store = SkerryKVStore("http://skerry.test")
    with pytest.raises(ValueError, match="64 lowercase"):
        store.get_store(key)


def test_missing_object_is_key_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 404, "not found", None, None)

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", fake_urlopen)
    with pytest.raises(KeyError):
        SkerryKVStore("http://skerry.test").get_store("a" * 64)


def test_health_reports_status_and_connection_failures(monkeypatch):
    store = SkerryKVStore("http://skerry.test")

    def healthy_urlopen(request, timeout):
        assert request.full_url == "http://skerry.test/v1/status"
        return _Response(b'{"objects": 3}')

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", healthy_urlopen)
    assert store.check_health() == {
        "healthy": True,
        "errors": [],
        "warnings": [],
        "endpoint": "http://skerry.test",
        "status": {"initialized": True, "root_path": "http://skerry.test", "objects": 3},
    }

    def unavailable_urlopen(request, timeout):
        raise HTTPError(request.full_url, 503, "unavailable", None, BytesIO(b"offline"))

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", unavailable_urlopen)
    health = store.check_health()
    assert health["healthy"] is False
    assert "503" in health["errors"][0]



def test_delete_and_gc_use_admin_protocol(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, request.get_method(), request.headers))
        return _Response(b'{"state":"complete"}' if request.get_method() == "POST" else b"")

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", fake_urlopen)
    store = SkerryKVStore("http://skerry.test", token="secret")
    store.delete_object("a" * 64)
    assert store.collect_garbage(force=True) == {"state": "complete"}
    assert requests[0] == (f"http://skerry.test/v1/objects/{'a' * 64}", "DELETE", {"Accept": "application/json", "Authorization": "Bearer secret"})
    assert requests[1][0:2] == ("http://skerry.test/v1/admin/gc?force=true", "POST")
    assert requests[1][2]["Authorization"] == "Bearer secret"
    with pytest.raises(ValueError):
        store.delete_object("../bad")


def test_gc_defaults_to_non_forced_and_rejects_invalid_responses(monkeypatch):
    requests = []

    def fake_urlopen(request, timeout):
        requests.append((request.full_url, request.get_method(), request.headers))
        return _Response(b"not json")

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", fake_urlopen)
    store = SkerryKVStore("http://skerry.test", token="secret")

    with pytest.raises(SkerryKVError, match="invalid garbage collection response"):
        store.collect_garbage()
    assert requests == [
        (
            "http://skerry.test/v1/admin/gc?force=false",
            "POST",
            {"Accept": "application/json", "Authorization": "Bearer secret"},
        )
    ]

    def non_object_urlopen(request, timeout):
        return _Response(b"[]")

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", non_object_urlopen)
    with pytest.raises(SkerryKVError, match="non-object garbage collection response"):
        store.collect_garbage()


def test_delete_surfaces_v1_conflict(monkeypatch):
    def fake_urlopen(request, timeout):
        raise HTTPError(request.full_url, 409, "conflict", None, BytesIO(b"upgrade required"))

    monkeypatch.setattr("Pelagia.storage.skerrykv.urlopen", fake_urlopen)
    with pytest.raises(SkerryKVError, match="409.*upgrade required"):
        SkerryKVStore("http://skerry.test", token="secret").delete_object("a" * 64)
