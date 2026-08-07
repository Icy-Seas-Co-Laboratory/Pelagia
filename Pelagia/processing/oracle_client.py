from __future__ import annotations

import hashlib
import io
import json
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote

import httpx
import numpy as np

from ..config import OracleConfig
from .detection_refinement import RoiRefinementInput, RoiRefinementPrediction


NPZ_MEDIA_TYPE = "application/vnd.oracle-builder.inference+npz"
REQUEST_SCHEMA = "oracle_builder.inference_request"
SCHEMA_VERSION = "1.0.0"


class OracleInferenceError(RuntimeError):
    """Base error for Oracle Builder inference."""


class OracleUnavailableError(OracleInferenceError):
    """The configured Oracle Builder service could not be reached."""


class OracleRejectedError(OracleInferenceError):
    """Oracle Builder rejected a selector or payload."""


def _array_sha256(value: np.ndarray) -> str:
    buffer = io.BytesIO()
    np.save(buffer, value, allow_pickle=False)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _manifest_array(value: dict[str, Any]) -> np.ndarray:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return np.frombuffer(encoded, dtype=np.uint8)


@dataclass(frozen=True, slots=True)
class OracleInferenceItem:
    """Task-neutral input passed through the Oracle Builder contract."""

    resource_type: str
    resource_id: str
    inputs: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OracleInferenceResult:
    transport_request_id: str
    result: dict[str, Any]


def _encode_request(request_id: str, inputs: list[OracleInferenceItem]) -> bytes:
    arrays: dict[str, np.ndarray] = {}
    items: list[dict[str, Any]] = []
    for index, item in enumerate(inputs):
        item_inputs: dict[str, dict[str, Any]] = {}
        for input_name, value in item.inputs.items():
            array = np.asarray(value)
            transport_key = f"item_{index}_{input_name}"
            arrays[transport_key] = array
            item_inputs[input_name] = {
                "transport_key": transport_key,
                "asset_id": str(uuid.uuid4()),
                "sha256": _array_sha256(array),
                "shape": list(array.shape),
                "dtype": str(array.dtype),
            }
        items.append(
            {
                "item_id": str(uuid.uuid4()),
                "request_id": request_id,
                "source": {
                    "system": "pelagia",
                    "resource_type": item.resource_type,
                    "resource_id": item.resource_id,
                },
                "metadata": dict(item.metadata),
                "inputs": item_inputs,
            }
        )
    manifest = {
        "schema_name": REQUEST_SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "request_id": request_id,
        "items": items,
    }
    buffer = io.BytesIO()
    np.savez_compressed(buffer, manifest=_manifest_array(manifest), **arrays)
    return buffer.getvalue()


def _decode_result(payload: bytes, *, max_payload_bytes: int) -> dict[str, Any]:
    if len(payload) > max_payload_bytes:
        raise OracleInferenceError("Oracle Builder response exceeds configured payload limit")
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            manifest = json.loads(
                np.asarray(archive["manifest"], dtype=np.uint8).tobytes().decode("utf-8")
            )

            def restore(value: Any) -> Any:
                if isinstance(value, dict):
                    key = value.get("transport_key")
                    if key is not None:
                        if str(key) not in archive.files:
                            raise OracleInferenceError(f"Oracle result is missing array {key!r}")
                        return np.asarray(archive[str(key)])
                    return {str(k): restore(v) for k, v in value.items()}
                if isinstance(value, list):
                    return [restore(item) for item in value]
                return value

            return restore(manifest)
    except OracleInferenceError:
        raise
    except Exception as exc:
        raise OracleInferenceError("Oracle Builder returned an invalid NPZ result") from exc


@dataclass
class OracleInferenceClient:
    """Task-neutral, connection-pooled Oracle Builder service gateway."""

    config: OracleConfig
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.max_items_per_request < 1:
            raise ValueError("oracle.max_items_per_request must be at least 1")
        if self.config.max_payload_bytes < 1:
            raise ValueError("oracle.max_payload_bytes must be at least 1")

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers = {"Accept": NPZ_MEDIA_TYPE}
            if self.config.api_token:
                headers["Authorization"] = f"Bearer {self.config.api_token}"
            self._client = httpx.Client(
                base_url=self.config.base_url,
                headers=headers,
                timeout=httpx.Timeout(
                    connect=self.config.connect_timeout_seconds,
                    read=self.config.read_timeout_seconds,
                    write=self.config.read_timeout_seconds,
                    pool=self.config.connect_timeout_seconds,
                ),
            )
        return self._client

    def list_models(self, *, task: str = "segmentation") -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        try:
            response = self._http().get("/v1/models", params={"task": task})
            response.raise_for_status()
            return list(response.json().get("models") or [])
        except httpx.HTTPError as exc:
            raise OracleUnavailableError(f"Oracle Builder model catalog is unavailable: {exc}") from exc

    def health(self) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False, "status": "disabled"}
        try:
            response = self._http().get("/health/ready")
            return {
                "enabled": True,
                "status": "ready" if response.is_success else "unavailable",
            }
        except httpx.HTTPError as exc:
            return {"enabled": True, "status": "unavailable", "error": str(exc)}

    def predict_batch(
        self,
        model_ref: str,
        inputs: list[OracleInferenceItem],
    ) -> list[OracleInferenceResult]:
        if not self.config.enabled:
            raise OracleUnavailableError("Oracle Builder inference is disabled")
        if not model_ref.strip():
            raise ValueError("An Oracle Builder model selector is required")
        predictions: list[OracleInferenceResult] = []
        for start in range(0, len(inputs), self.config.max_items_per_request):
            chunk = inputs[start : start + self.config.max_items_per_request]
            request_id = str(uuid.uuid4())
            body = _encode_request(request_id, chunk)
            if len(body) > self.config.max_payload_bytes:
                raise OracleRejectedError("Oracle Builder request exceeds configured payload limit")
            selector = quote(model_ref, safe="")
            try:
                response = self._http().post(
                    f"/v1/models/{selector}:predict",
                    content=body,
                    headers={"Content-Type": NPZ_MEDIA_TYPE},
                )
            except httpx.HTTPError as exc:
                raise OracleUnavailableError(f"Oracle Builder inference request failed: {exc}") from exc
            if response.status_code in {404, 415, 422}:
                raise OracleRejectedError(
                    f"Oracle Builder rejected inference ({response.status_code}): {response.text}"
                )
            if response.status_code >= 500:
                raise OracleUnavailableError(
                    f"Oracle Builder inference failed ({response.status_code}): {response.text}"
                )
            response.raise_for_status()
            result_set = _decode_result(
                response.content, max_payload_bytes=self.config.max_payload_bytes
            )
            rows = result_set.get("results") or []
            if len(rows) != len(chunk):
                raise OracleInferenceError("Oracle Builder returned an unexpected result count")
            for row in rows:
                if row.get("status") != "ok":
                    raise OracleInferenceError(
                        f"Oracle Builder item inference failed: {row.get('error') or row.get('status')}"
                    )
                predictions.append(OracleInferenceResult(request_id, row))
        return predictions

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


@dataclass(slots=True)
class OracleRoiRefinementBackend:
    """Adapts Pelagia refinement candidates to the task-neutral Oracle gateway."""

    client: OracleInferenceClient
    model_ref: str

    @property
    def method_name(self) -> str:
        return f"oracle_builder:{self.model_ref}"

    def refine_batch(self, inputs: list[RoiRefinementInput]) -> list[RoiRefinementPrediction]:
        oracle_items = [
            OracleInferenceItem(
                resource_type="candidate_detection",
                resource_id=item.detection_id,
                inputs={"image": item.image, "candidate_mask": item.candidate_mask},
                metadata=dict(item.metadata),
            )
            for item in inputs
        ]
        results = self.client.predict_batch(self.model_ref, oracle_items)
        predictions: list[RoiRefinementPrediction] = []
        for response in results:
            row = response.result
            output = row.get("output") or {}
            mask = output.get("reconstructed_mask", output.get("mask"))
            if not isinstance(mask, np.ndarray):
                raise OracleInferenceError("Oracle Builder result is missing a mask array")
            predictions.append(
                RoiRefinementPrediction(
                    mask=np.asarray(mask),
                    probability_map=output.get(
                        "reconstructed_probability_map", output.get("probability_map")
                    ),
                    metadata={
                        "oracle_request_id": response.transport_request_id,
                        "oracle_result_id": row.get("result_id"),
                        "oracle_result_set_id": row.get("result_set_id"),
                        "oracle_model": row.get("model"),
                        "oracle_input_sha256": row.get("input_sha256"),
                        "oracle_execution": row.get("execution"),
                        "oracle_threshold": output.get("threshold"),
                        "oracle_transform": output.get("transform"),
                        "oracle_logits_source": output.get("logits_source"),
                    },
                )
            )
        return predictions
