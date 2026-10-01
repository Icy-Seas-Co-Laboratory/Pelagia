from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import quote

import httpx
import numpy as np
from oracle_data_contracts.inference_v2 import (
    InferenceItem as V2InferenceItem,
    SourceReference,
    V2InferenceRequest,
    V2ResponseOptions,
    decode_v2_inference_result_set,
    encode_v2_inference_request,
)

from ..config import OracleConfig
from .detection_refinement import RoiRefinementInput, RoiRefinementPrediction
from .timing import measure_phase


NPZ_MEDIA_TYPE = "application/vnd.oracle-builder.inference.v2+npz"
_MAX_RESPONSE_BYTES = 256 * 1024 * 1024


class OracleInferenceError(RuntimeError):
    """Base error for Oracle Builder inference."""


class OracleUnavailableError(OracleInferenceError):
    """Oracle Builder could not be reached."""


class OracleRejectedError(OracleInferenceError):
    """Oracle Builder rejected a request or its contract."""


class OracleFingerprintMismatchError(OracleRejectedError):
    """The selected sealed artifact has changed."""


@dataclass(frozen=True, slots=True)
class OracleInferenceItem:
    resource_type: str
    resource_id: str
    inputs: dict[str, np.ndarray]
    metadata: dict[str, Any] = field(default_factory=dict)
    item_id: str | None = None


@dataclass(frozen=True, slots=True)
class OracleInferenceResult:
    transport_request_id: str
    result: dict[str, Any]


def _detail(response: httpx.Response) -> str:
    return response.text.strip()[:1000] or "no detail supplied"


@dataclass
class OracleInferenceClient:
    """Dependency-light client for catalog-addressed Oracle Builder V2 inference."""

    config: OracleConfig
    _client: httpx.Client | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.config.max_items_per_request < 1:
            raise ValueError("oracle.max_items_per_request must be at least 1")
        if self.config.max_payload_bytes < 1:
            raise ValueError("oracle.max_payload_bytes must be at least 1")

    def _http(self) -> httpx.Client:
        if self._client is None:
            headers = {"Accept": f"application/json, {NPZ_MEDIA_TYPE}"}
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

    def _retry_attempts(self) -> int:
        return max(0, int(getattr(self.config, "retry_attempts", 2)))

    def _max_output_bytes(self) -> int:
        return int(getattr(self.config, "max_output_bytes_per_item", 32 * 1024 * 1024))

    def _request(self, method: str, url: str, **kwargs: Any) -> httpx.Response:
        """Retries transport timeouts and 503 only; inputs are never replayed on 4xx."""
        attempts = self._retry_attempts() + 1
        failure: Exception | None = None
        for attempt in range(attempts):
            try:
                response = self._http().request(method, url, **kwargs)
            except httpx.TimeoutException as exc:
                failure = exc
            except httpx.HTTPError as exc:
                raise OracleUnavailableError(f"Oracle Builder request failed: {exc}") from exc
            else:
                if response.status_code == 409:
                    raise OracleFingerprintMismatchError(
                        f"Oracle Builder rejected the pinned artifact fingerprint: {_detail(response)}"
                    )
                if response.status_code in {400, 404, 413, 415, 422}:
                    raise OracleRejectedError(
                        f"Oracle Builder rejected request ({response.status_code}): {_detail(response)}"
                    )
                if response.status_code != 503:
                    try:
                        response.raise_for_status()
                    except httpx.HTTPError as exc:
                        raise OracleUnavailableError(
                            f"Oracle Builder request failed ({response.status_code}): {_detail(response)}"
                        ) from exc
                    return response
                failure = OracleUnavailableError("Oracle Builder resident runtime is unavailable (503)")
            if attempt + 1 == attempts:
                break
        raise OracleUnavailableError(f"Oracle Builder request failed after {attempts} attempts: {failure}")

    def list_models(self, *, task: str | None = None) -> list[dict[str, Any]]:
        if not self.config.enabled:
            return []
        try:
            response = self._request("GET", "/v2/inference/models")
            models = list(response.json().get("models") or [])
        except (ValueError, OracleInferenceError) as exc:
            raise OracleUnavailableError(f"Oracle Builder model catalog is unavailable: {exc}") from exc
        return models if task is None else [model for model in models if model.get("task") == task]

    def get_model(self, model_ref: str) -> dict[str, Any]:
        """Fetch an artifact's fingerprint and capabilities before queueing work."""
        if not model_ref.strip():
            raise ValueError("An Oracle Builder artifact ID is required")
        response = self._request("GET", f"/v2/inference/models/{quote(model_ref, safe='')}")
        try:
            payload = response.json()
        except ValueError as exc:
            raise OracleInferenceError("Oracle Builder returned invalid model metadata") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("fingerprint_sha256"), str):
            raise OracleInferenceError("Oracle Builder model metadata lacks fingerprint_sha256")
        return payload

    model_detail = get_model

    def warm_model(self, model_ref: str, fingerprint: str) -> dict[str, Any]:
        response = self._request(
            "POST", f"/v2/inference/models/{quote(model_ref, safe='')}:warm",
            headers={"X-Artifact-Fingerprint": fingerprint},
        )
        try:
            payload = response.json()
        except ValueError as exc:
            raise OracleInferenceError("Oracle Builder returned invalid warm response") from exc
        if not isinstance(payload, dict):
            raise OracleInferenceError("Oracle Builder returned invalid warm response")
        return payload

    def health(self) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False, "status": "disabled"}
        try:
            response = self._http().get("/health/ready")
            return {"enabled": True, "status": "ready" if response.is_success else "unavailable"}
        except httpx.HTTPError as exc:
            return {"enabled": True, "status": "unavailable", "error": str(exc)}

    @staticmethod
    def _v2_item(item: OracleInferenceItem, request_id: str) -> V2InferenceItem:
        try:
            return V2InferenceItem.from_array(
                item.inputs["image"], candidate_mask=item.inputs.get("candidate_mask"),
                item_id=item.item_id, request_id=request_id,
                source=SourceReference("pelagia", item.resource_type, item.resource_id),
                metadata=dict(item.metadata),
            )
        except KeyError as exc:
            raise OracleRejectedError("Oracle Builder items require an image input") from exc
        except (TypeError, ValueError) as exc:
            raise OracleRejectedError(f"Oracle Builder inference item is invalid: {exc}") from exc

    def prepare_items(
        self, inputs: Iterable[OracleInferenceItem], *, request_id: str,
    ) -> list[V2InferenceItem]:
        """Create stable V2 items so a caller can persist IDs and input hashes.

        Pelagia's durable batch ledger should call this before submission and
        retain each item's ``item_id`` and ``input_sha256``. Reusing the same
        request ID and item IDs makes a retry correlate with the original
        evidence reservation.
        """
        if not request_id.strip():
            raise ValueError("An Oracle Builder request ID is required")
        return [self._v2_item(item, request_id) for item in inputs]

    def predict_batch(
        self,
        model_ref: str,
        inputs: list[OracleInferenceItem],
        *,
        fingerprint: str | None = None,
        task: str = "classification",
        outputs: Iterable[str] | None = None,
        request_id: str | None = None,
    ) -> list[OracleInferenceResult]:
        """Submit bounded V2 frames and return rows in the original item order.

        ``model_ref`` is the sealed public artifact id.  A caller which did not
        pin it earlier can omit ``fingerprint``; the catalog is then consulted
        once. Per-item failures remain result rows so callers can record their
        stable item IDs rather than losing the entire response.
        """
        if not self.config.enabled:
            raise OracleUnavailableError("Oracle Builder inference is disabled")
        if not model_ref.strip():
            raise ValueError("An Oracle Builder artifact ID is required")
        if not inputs:
            return []
        fingerprint = fingerprint or str(self.get_model(model_ref)["fingerprint_sha256"])
        options = V2ResponseOptions(
            outputs=frozenset(outputs) if outputs is not None else None,
            max_output_bytes=self._max_output_bytes(),
        )
        pending = list(inputs)
        complete: list[OracleInferenceResult] = []
        while pending:
            frame_request_id = request_id or str(uuid.uuid4())
            candidates = [self._v2_item(item, frame_request_id)
                          for item in pending[:min(self.config.max_items_per_request, 64)]]

            def encoded_prefix(count: int) -> bytes:
                with measure_phase("oracle_transport.request_encode"):
                    return encode_v2_inference_request(
                        V2InferenceRequest(frame_request_id, task, candidates[:count], options)
                    )

            body = encoded_prefix(len(candidates))
            if len(body) <= self.config.max_payload_bytes:
                selected_count = len(candidates)
            else:
                lower, upper = 0, len(candidates)
                while lower + 1 < upper:
                    middle = (lower + upper) // 2
                    encoded = encoded_prefix(middle)
                    if len(encoded) <= self.config.max_payload_bytes:
                        lower, body = middle, encoded
                    else:
                        upper = middle
                selected_count = lower
                if selected_count == 0:
                    raise OracleRejectedError("One inference item exceeds the configured V2 frame limit")
            selected = candidates[:selected_count]
            pending = pending[selected_count:]
            if request_id is not None and pending:
                raise OracleRejectedError(
                    "A caller-supplied request_id must fit in one configured V2 frame"
                )
            with measure_phase("oracle_transport.http_round_trip"):
                response = self._request(
                    "POST", f"/v2/inference/models/{quote(model_ref, safe='')}:predict",
                    content=body,
                    headers={"Content-Type": NPZ_MEDIA_TYPE, "X-Artifact-Fingerprint": fingerprint},
                )
            with measure_phase("oracle_transport.response_decode"):
                try:
                    result_set = decode_v2_inference_result_set(
                        response.content, max_payload_bytes=_MAX_RESPONSE_BYTES,
                    )
                except (TypeError, ValueError) as exc:
                    raise OracleInferenceError("Oracle Builder returned an invalid V2 result") from exc
            rows = result_set.get("results")
            if not isinstance(rows, list):
                raise OracleInferenceError("Oracle Builder result is missing result rows")
            expected = {item.item_id for item in selected}
            indexed: dict[str, dict[str, Any]] = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise OracleInferenceError("Oracle Builder result row is invalid")
                item_id = row.get("item_id")
                if row.get("request_id") != frame_request_id or not isinstance(item_id, str) or item_id not in expected:
                    raise OracleInferenceError("Oracle Builder result does not match the submitted request")
                if item_id in indexed:
                    raise OracleInferenceError("Oracle Builder returned duplicate item results")
                model = row.get("model") or {}
                if model.get("artifact_id") != model_ref or model.get("artifact_fingerprint") != fingerprint:
                    raise OracleInferenceError("Oracle Builder result does not match the pinned artifact")
                indexed[item_id] = row
            if set(indexed) != expected:
                raise OracleInferenceError("Oracle Builder returned incomplete item results")
            complete.extend(OracleInferenceResult(frame_request_id, indexed[item.item_id]) for item in selected)
        return complete

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


@dataclass(slots=True)
class OracleRoiRefinementBackend:
    client: OracleInferenceClient
    model_ref: str
    fingerprint: str | None = None

    @property
    def method_name(self) -> str:
        return f"oracle_builder:{self.model_ref}"

    def refine_batch(self, inputs: list[RoiRefinementInput]) -> list[RoiRefinementPrediction]:
        results = self.client.predict_batch(
            self.model_ref,
            [OracleInferenceItem("candidate_detection", item.detection_id,
                {"image": item.image, "candidate_mask": item.candidate_mask}, dict(item.metadata)) for item in inputs],
            task="mask_refinement", fingerprint=self.fingerprint,
        )
        predictions: list[RoiRefinementPrediction] = []
        for response in results:
            row, output = response.result, response.result.get("output") or {}
            mask = output.get("mask")
            if row.get("status") != "ok" or not isinstance(mask, np.ndarray):
                raise OracleInferenceError(f"Oracle Builder refinement failed: {row.get('error') or 'missing mask'}")
            predictions.append(RoiRefinementPrediction(
                mask=np.asarray(mask), probability_map=output.get("probability_map"), metadata={
                    "oracle_request_id": response.transport_request_id,
                    "oracle_result_id": row.get("result_id"), "oracle_result_set_id": row.get("result_set_id"),
                    "oracle_model": row.get("model"), "oracle_input_sha256": row.get("input_sha256"),
                    "oracle_execution": row.get("execution"), "oracle_threshold": output.get("threshold"),
                    "oracle_transform": output.get("transform"), "oracle_logits_source": output.get("logits_source"),
                },
            ))
        return predictions
