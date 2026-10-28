from __future__ import annotations

import json
import struct
from io import BytesIO
from urllib.error import HTTPError

import pytest

from Pelagia.storage.skerrykv import SkerryKVBatchError, SkerryKVError, SkerryKVStore


class _Response:
    def __init__(self, body: bytes):
        self.body = body

    def read(self) -> bytes:
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


class _BatchResponse:
    status = 200

    def __init__(self, body: bytes):
        self._body = BytesIO(body)

    def read(self, size: int = -1) -> bytes:
        return self._body.read(size)

    def getheader(self, name: str, default=None):
        return "application/vnd.skerry.batch-v1" if name.lower() == "content-type" else default


class _BatchConnection:
    def __init__(self, response: _BatchResponse):
        self.response = response
        self.sent: list[bytes] = []
        self.headers: dict[str, str] = {}

    def putrequest(self, *_args, **_kwargs):
        pass

    def putheader(self, name, value):
        self.headers[name] = value

    def endheaders(self):
        pass

    def send(self, data):
        self.sent.append(data)

    def getresponse(self):
        return self.response

    def close(self):
        pass


def _batch_result(ordinal: int, digest: bytes, status: int = 0) -> bytes:
    return struct.pack("<I4sBBHQ32sQ", 56, b"SKB1", 129, status, 0, ordinal, digest, 0)


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


def test_put_many_uses_streamed_batch_protocol(monkeypatch):
    connection = _BatchConnection(_BatchResponse(
        _batch_result(0, bytes.fromhex("a" * 64)) + _batch_result(1, bytes.fromhex("b" * 64))
    ))
    store = SkerryKVStore("http://skerry.test", token="secret")
    monkeypatch.setattr(store, "_open_batch_connection", lambda: (connection, "/v1/objects/batch-put"))

    assert store.put_many([b"first", b"second"]) == ["a" * 64, "b" * 64]
    assert connection.headers["Content-Type"] == "application/vnd.skerry.batch-v1"
    assert connection.headers["Transfer-Encoding"] == "chunked"
    assert connection.headers["Authorization"] == "Bearer secret"
    assert b"SKB1" in connection.sent[0]
    assert connection.sent[-1] == b"0\r\n\r\n"


def test_put_many_caps_requests_and_exposes_partial_result_indexes(monkeypatch):
    connections = iter([
        _BatchConnection(_BatchResponse(
            _batch_result(0, bytes.fromhex("a" * 64)) + _batch_result(1, bytes.fromhex("b" * 64))
        )),
        _BatchConnection(_BatchResponse(_batch_result(0, bytes(32), status=6))),
    ])
    store = SkerryKVStore("http://skerry.test", batch_max_items=2)
    monkeypatch.setattr(store, "_open_batch_connection", lambda: (next(connections), "/v1/objects/batch-put"))

    with pytest.raises(SkerryKVBatchError) as raised:
        store.put_many([b"one", b"two", b"three"])

    assert raised.value.completed == {0: "a" * 64, 1: "b" * 64}
    assert raised.value.failed == {2: 6}


def test_put_many_rejects_duplicate_or_payload_bearing_results(monkeypatch):
    duplicate = _BatchConnection(_BatchResponse(
        _batch_result(0, bytes.fromhex("a" * 64)) + _batch_result(0, bytes.fromhex("a" * 64))
    ))
    store = SkerryKVStore("http://skerry.test")
    monkeypatch.setattr(store, "_open_batch_connection", lambda: (duplicate, "/v1/objects/batch-put"))
    with pytest.raises(SkerryKVError, match="duplicate"):
        store.put_many([b"one"])

    payload_bearing = struct.pack("<I4sBBHQ32sQ", 57, b"SKB1", 129, 0, 0, 0, bytes(32), 1) + b"x"
    monkeypatch.setattr(store, "_open_batch_connection", lambda: (_BatchConnection(_BatchResponse(payload_bearing)), "/v1/objects/batch-put"))
    with pytest.raises(SkerryKVError, match="unexpected"):
        store.put_many([b"one"])


def test_batch_defaults_use_250_items_and_64_mib():
    store = SkerryKVStore("http://skerry.test")
    assert store.batch_max_items == 250
    assert store.batch_max_bytes == 64 * 1024 * 1024
    assert [len(batch) for batch in store._put_batches([b"x"] * 501)] == [250, 250, 1]


def test_put_many_matches_out_of_order_responses_by_ordinal(monkeypatch):
    connection = _BatchConnection(_BatchResponse(
        _batch_result(2, bytes.fromhex("c" * 64))
        + _batch_result(0, bytes.fromhex("a" * 64))
        + _batch_result(1, bytes.fromhex("b" * 64))
    ))
    store = SkerryKVStore("http://skerry.test")
    monkeypatch.setattr(store, "_open_batch_connection", lambda: (connection, "/v1/objects/batch-put"))
    assert store.put_many([b"first", b"second", b"third"]) == ["a" * 64, "b" * 64, "c" * 64]
