"""Synchronous adapter for SkerryKV's immutable HTTP object API."""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class SkerryKVError(RuntimeError):
    """Raised when SkerryKV rejects or cannot complete an operation."""


class SkerryKVStore:
    """Content-addressed blob storage served by one SkerryKV endpoint."""

    def __init__(self, endpoint: str, *, token: str | None = None, timeout_seconds: float = 30.0) -> None:
        self.endpoint = str(endpoint).strip().rstrip("/")
        if not self.endpoint.startswith(("http://", "https://")):
            raise ValueError("SkerryKV endpoint must be an http(s) URL.")
        self.token = token or None
        self.timeout_seconds = float(timeout_seconds)
        if self.timeout_seconds <= 0:
            raise ValueError("SkerryKV timeout_seconds must be positive.")
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

    def get_store(self, key: str) -> bytes:
        return self._request(f"/v1/objects/{self._validate_key(key)}")

    def key_exists(self, key: str) -> bool:
        try:
            self._request(f"/v1/objects/{self._validate_key(key)}", method="HEAD")
        except KeyError:
            return False
        return True

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

