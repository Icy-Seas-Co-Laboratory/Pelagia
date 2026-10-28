"""Synchronous adapter for SkerryKV's immutable HTTP object API."""

from __future__ import annotations

import json
import http.client
import re
import struct
import threading
from collections.abc import Iterable, Iterator
from typing import Any
from urllib.parse import urlsplit
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SkerryKVError(RuntimeError):
    """Raised when SkerryKV rejects or cannot complete an operation."""


class SkerryKVBatchError(SkerryKVError):
    """A batch PUT was only partly acknowledged by SkerryKV.

    ``completed`` maps input indexes to their durable object hashes.  Entries
    absent from it have an unknown outcome unless listed in ``failed``: the
    service may have committed them before a transport failure.  Callers may
    safely retry missing PUTs because SkerryKV is content addressed.
    """

    def __init__(self, message: str, *, completed: dict[int, str], failed: dict[int, int]) -> None:
        super().__init__(message)
        self.completed = completed
        self.failed = failed


_BATCH_MEDIA_TYPE = "application/vnd.skerry.batch-v1"
_BATCH_HEADER = struct.Struct("<I4sBBHQ32sQ")
_BATCH_FIXED_LENGTH = 56
_BATCH_PUT_REQUEST = 1
_BATCH_PUT_RESULT = 129
_BATCH_TERMINAL_ERROR = 255


class SkerryKVStore:
    """Content-addressed blob storage served by one SkerryKV endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        token: str | None = None,
        timeout_seconds: float = 30.0,
        batch_max_items: int = 250,
        batch_max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        self.endpoint = str(endpoint).strip().rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("SkerryKV endpoint must be an http(s) URL.")
        self.token = token or None
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("SkerryKV timeout_seconds must be positive.")
        if batch_max_items < 1:
            raise ValueError("SkerryKV batch_max_items must be positive.")
        if batch_max_bytes < 1:
            raise ValueError("SkerryKV batch_max_bytes must be positive.")
        self.batch_max_items = int(batch_max_items)
        self.batch_max_bytes = int(batch_max_bytes)
        self.root_path = self.endpoint
        self.initialized = True

    def _request(self, path: str, *, data: bytes | None = None, method: str = "GET") -> bytes:
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/octet-stream"
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        request = Request(f"{self.endpoint}{path}", data=data, method=method, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout_seconds) as response:
                return response.read()
        except HTTPError as exc:
            if exc.code == 404:
                raise KeyError(path.rsplit("/", 1)[-1]) from exc
            detail = exc.read().decode("utf-8", "replace")
            raise SkerryKVError(f"SkerryKV {method} {path} failed ({exc.code}): {detail}") from exc
        except URLError as exc:
            raise SkerryKVError(f"SkerryKV {method} {path} failed: {exc.reason}") from exc

    @staticmethod
    def _validate_key(key: str) -> str:
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise ValueError("SkerryKV object key must be 64 lowercase hexadecimal characters.")
        return key

    def put_store(self, payload: bytes | bytearray | str) -> str:
        payload_bytes = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
        response = self._request("/v1/objects", data=payload_bytes, method="PUT")
        try:
            key = json.loads(response)["hash"]
        except (KeyError, TypeError, json.JSONDecodeError) as exc:
            raise SkerryKVError("SkerryKV returned an invalid object PUT response.") from exc
        if not isinstance(key, str) or re.fullmatch(r"[0-9a-f]{64}", key) is None:
            raise SkerryKVError("SkerryKV returned an invalid object hash.")
        return key

    @staticmethod
    def _payload_bytes(payload: bytes | bytearray | str) -> bytes:
        return payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)

    def _put_batches(self, payloads: Iterable[bytes | bytearray | str]) -> Iterator[list[bytes]]:
        """Yield finite requests without retaining the entire caller iterable."""
        batch: list[bytes] = []
        batch_bytes = 0
        for payload in payloads:
            payload_bytes = self._payload_bytes(payload)
            # A large object is sent by itself; the byte cap bounds aggregation,
            # rather than silently rejecting a valid SkerryKV object.
            if batch and (
                len(batch) >= self.batch_max_items
                or batch_bytes + len(payload_bytes) > self.batch_max_bytes
            ):
                yield batch
                batch = []
                batch_bytes = 0
            batch.append(payload_bytes)
            batch_bytes += len(payload_bytes)
            if len(batch) >= self.batch_max_items or batch_bytes >= self.batch_max_bytes:
                yield batch
                batch = []
                batch_bytes = 0
        if batch:
            yield batch

    def _open_batch_connection(self) -> tuple[http.client.HTTPConnection, str]:
        parsed = urlsplit(self.endpoint)
        host = parsed.hostname
        if host is None:
            raise ValueError("SkerryKV endpoint must include a host.")
        connection_type = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(host, port=parsed.port, timeout=self.timeout_seconds)
        root = parsed.path.rstrip("/")
        return connection, f"{root}/v1/objects/batch-put"

    @staticmethod
    def _chunk(data: bytes) -> bytes:
        return f"{len(data):X}\r\n".encode("ascii") + data + b"\r\n"

    @staticmethod
    def _read_exact(response: http.client.HTTPResponse, size: int, *, allow_eof: bool = False) -> bytes | None:
        data = bytearray()
        while len(data) < size:
            block = response.read(size - len(data))
            if not block:
                if allow_eof and not data:
                    return None
                raise SkerryKVError("SkerryKV batch response ended in the middle of a frame.")
            data.extend(block)
        return bytes(data)

    def _put_batch(self, payloads: list[bytes], *, offset: int) -> list[str]:
        """Send one finite batch while draining its result stream concurrently.

        Chunked request framing lets the writer keep feeding input while this
        thread reads result frames.  That matters for SkerryKV's bidirectional
        flow control and avoids assembling a whole request body in memory.
        """
        connection, path = self._open_batch_connection()
        writer_error: list[BaseException] = []
        writer_done = threading.Event()

        headers = {"Content-Type": _BATCH_MEDIA_TYPE, "Accept": _BATCH_MEDIA_TYPE, "Transfer-Encoding": "chunked"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        try:
            connection.putrequest("POST", path, skip_accept_encoding=True)
            for name, value in headers.items():
                connection.putheader(name, value)
            connection.endheaders()

            def write_request() -> None:
                try:
                    for ordinal, payload in enumerate(payloads):
                        header = _BATCH_HEADER.pack(
                            _BATCH_FIXED_LENGTH + len(payload), b"SKB1", _BATCH_PUT_REQUEST,
                            0, 0, ordinal, bytes(32), len(payload),
                        )
                        # HTTP chunks have no protocol meaning; splitting here
                        # prevents an extra full-payload concatenation.
                        connection.send(self._chunk(header))
                        if payload:
                            connection.send(self._chunk(payload))
                    connection.send(b"0\r\n\r\n")
                except BaseException as exc:  # reader reports the operation context below
                    writer_error.append(exc)
                    connection.close()
                finally:
                    writer_done.set()

            writer = threading.Thread(target=write_request, name="skerrykv-batch-put", daemon=True)
            writer.start()
            response = connection.getresponse()
            if response.status < 200 or response.status >= 300:
                detail = response.read().decode("utf-8", "replace")
                raise SkerryKVError(f"SkerryKV batch PUT failed ({response.status}): {detail}")
            if response.getheader("Content-Type", "").split(";", 1)[0].strip().lower() != _BATCH_MEDIA_TYPE:
                raise SkerryKVError("SkerryKV returned an invalid batch PUT media type.")

            completed: dict[int, str] = {}
            failed: dict[int, int] = {}
            while True:
                prefix = self._read_exact(response, 4, allow_eof=True)
                if prefix is None:
                    break
                frame_length = struct.unpack("<I", prefix)[0]
                if frame_length < _BATCH_FIXED_LENGTH:
                    raise SkerryKVError("SkerryKV returned an invalid batch response frame length.")
                header = self._read_exact(response, _BATCH_FIXED_LENGTH)
                assert header is not None
                magic, kind, status, flags, ordinal, digest, payload_length = struct.unpack("<4sBBHQ32sQ", header)
                if magic != b"SKB1" or flags != 0 or payload_length != frame_length - _BATCH_FIXED_LENGTH:
                    raise SkerryKVError("SkerryKV returned an invalid batch response frame.")
                if kind == _BATCH_TERMINAL_ERROR:
                    if ordinal != 2**64 - 1 or digest != bytes(32) or status == 0 or payload_length != 0:
                        raise SkerryKVError("SkerryKV returned an invalid terminal batch error.")
                    failed.update({index: status for index in range(len(payloads)) if index not in completed and index not in failed})
                    break
                if kind != _BATCH_PUT_RESULT or ordinal >= len(payloads) or payload_length != 0:
                    raise SkerryKVError("SkerryKV returned an unexpected batch PUT result.")
                if ordinal in completed or ordinal in failed:
                    raise SkerryKVError("SkerryKV returned a duplicate batch PUT result.")
                if status == 0:
                    key = digest.hex()
                    if re.fullmatch(r"[0-9a-f]{64}", key) is None:
                        raise SkerryKVError("SkerryKV returned an invalid batch object hash.")
                    completed[ordinal] = key
                else:
                    failed[ordinal] = status
            writer.join(timeout=self.timeout_seconds)
            if writer.is_alive():
                raise SkerryKVError("SkerryKV batch PUT writer did not finish before the timeout.")
            if writer_error:
                raise SkerryKVBatchError(
                    "SkerryKV batch PUT transport failed; unacknowledged items have unknown outcomes.",
                    completed={offset + index: key for index, key in completed.items()},
                    failed={offset + index: status for index, status in failed.items()},
                ) from writer_error[0]
            missing = set(range(len(payloads))) - set(completed) - set(failed)
            if failed or missing:
                raise SkerryKVBatchError(
                    "SkerryKV batch PUT was only partially acknowledged; retry failed or unacknowledged items.",
                    completed={offset + index: key for index, key in completed.items()},
                    failed={offset + index: status for index, status in failed.items()},
                )
            return [completed[index] for index in range(len(payloads))]
        except (OSError, http.client.HTTPException) as exc:
            raise SkerryKVBatchError(
                "SkerryKV batch PUT transport failed; item outcomes may be unknown.", completed={}, failed={}
            ) from exc
        finally:
            connection.close()

    def put_many(self, payloads: Iterable[bytes | bytearray | str]) -> list[str]:
        """Durably store many payloads with bounded, streaming SkerryKV batches.

        This intentionally has no per-item compatibility fallback: falling back
        to N single-object requests would reintroduce the candidate-detection
        performance regression and hide a misconfigured/incompatible service.
        """
        results: list[str] = []
        for batch in self._put_batches(payloads):
            try:
                results.extend(self._put_batch(batch, offset=len(results)))
            except SkerryKVBatchError as exc:
                # Preserve acknowledgements from preceding finite requests too;
                # callers can then retry exactly the failed/unknown input slots.
                exc.completed = {**{index: key for index, key in enumerate(results)}, **exc.completed}
                raise
        return results

    def get_store(self, key: str) -> bytes:
        return self._request(f"/v1/objects/{self._validate_key(key)}")

    def key_exists(self, key: str) -> bool:
        try:
            self._request(f"/v1/objects/{self._validate_key(key)}", method="HEAD")
        except KeyError:
            return False
        return True

    def delete_object(self, key: str) -> None:
        """Delete an object after the caller has established that it is unreferenced.

        SkerryKV requires an administrator token and treats a missing hash as a
        successful deletion. This low-level method does not check Pelagia references.
        """
        self._request(f"/v1/objects/{self._validate_key(key)}", method="DELETE")

    def collect_garbage(self, *, force: bool = False) -> dict[str, Any]:
        """Ask SkerryKV to reclaim space occupied by already-deleted objects."""
        response = self._request(f"/v1/admin/gc?force={'true' if force else 'false'}", method="POST")
        try:
            result = json.loads(response)
        except json.JSONDecodeError as exc:
            raise SkerryKVError("SkerryKV returned an invalid garbage collection response.") from exc
        if not isinstance(result, dict):
            raise SkerryKVError("SkerryKV returned a non-object garbage collection response.")
        return result

    def status(self, *, deep: bool = True) -> dict[str, Any]:
        del deep
        response = self._request("/v1/status")
        try:
            result = json.loads(response)
        except json.JSONDecodeError as exc:
            raise SkerryKVError("SkerryKV returned an invalid status response.") from exc
        if not isinstance(result, dict):
            raise SkerryKVError("SkerryKV returned a non-object status response.")
        return {"initialized": True, "root_path": self.endpoint, **result}

    def check_health(self) -> dict[str, Any]:
        """Check that the configured service is reachable and returns status."""
        try:
            status = self.status(deep=False)
        except Exception as exc:
            return {
                "healthy": False,
                "errors": [str(exc)],
                "warnings": [],
                "endpoint": self.endpoint,
            }
        return {
            "healthy": True,
            "errors": [],
            "warnings": [],
            "endpoint": self.endpoint,
            "status": status,
        }

    def close(self) -> None:
        """No persistent resources are held by the standard-library HTTP client."""

