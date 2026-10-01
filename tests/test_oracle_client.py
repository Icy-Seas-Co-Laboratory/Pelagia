from __future__ import annotations

import hashlib
import io
import json
import uuid

import httpx
import numpy as np
from oracle_data_contracts.inference_v2 import (
    V2InferenceRequest, V2ResponseOptions, encode_v2_inference_request,
)

from Pelagia.config import OracleConfig
from Pelagia.processing.detection_refinement import RoiRefinementInput
from Pelagia.processing.oracle_client import (
    NPZ_MEDIA_TYPE,
    OracleFingerprintMismatchError,
    OracleInferenceClient,
    OracleInferenceItem,
    OracleRoiRefinementBackend,
)


def _array_description(values: np.ndarray, key: str) -> dict[str, object]:
    encoded = io.BytesIO()
    np.save(encoded, values, allow_pickle=False)
    return {"transport_key": key, "asset_id": str(uuid.uuid4()),
            "sha256": hashlib.sha256(encoded.getvalue()).hexdigest(),
            "shape": list(values.shape), "dtype": str(values.dtype),
            "media_type": "application/x-npy"}


def _response_payload(rows: list[dict[str, object]], arrays: dict[str, np.ndarray]) -> bytes:
    manifest = {"schema_name": "oracle_builder.inference.v2.result_set", "schema_version": "2.0.0", "results": rows}
    buffer = io.BytesIO()
    np.savez_compressed(buffer, manifest=np.frombuffer(json.dumps(manifest).encode(), dtype=np.uint8), **arrays)
    return buffer.getvalue()


def _row(request_id: str, item_id: str, artifact_id: str, fingerprint: str, mask: np.ndarray) -> dict[str, object]:
    return {"request_id": request_id, "item_id": item_id, "status": "ok", "input_sha256": "input-hash",
            "result_id": str(uuid.uuid4()), "result_set_id": str(uuid.uuid4()),
            "model": {"artifact_id": artifact_id, "artifact_fingerprint": fingerprint, "task": "segmentation"},
            "output": {"type": "mask_refinement", "mask": _array_description(mask, f"mask_{item_id}")}}


def _client(handler) -> OracleInferenceClient:
    config = OracleConfig(base_url="http://oracle.test", default_mask_model="artifact-1")
    client = OracleInferenceClient(config)
    client._client = httpx.Client(base_url=config.base_url, transport=httpx.MockTransport(handler))
    return client


def _request_manifest(request: httpx.Request) -> dict[str, object]:
    with np.load(io.BytesIO(request.content), allow_pickle=False) as archive:
        return json.loads(np.asarray(archive["manifest"], dtype=np.uint8).tobytes().decode())


def test_v2_predict_uses_pinned_artifact_and_correlates_out_of_order_rows():
    seen: dict[str, object] = {}
    fingerprint = "a" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v2/inference/models/artifact-1":
            return httpx.Response(200, json={"artifact_id": "artifact-1", "fingerprint_sha256": fingerprint})
        seen.update(path=request.url.path, content_type=request.headers["content-type"],
                    fingerprint=request.headers["x-artifact-fingerprint"])
        manifest = _request_manifest(request)
        seen["task"] = manifest["task"]
        items = manifest["items"]
        assert isinstance(items, list)
        mask = np.ones((2, 2), dtype=np.uint8)
        rows = [_row(manifest["request_id"], item["item_id"], "artifact-1", fingerprint, mask) for item in reversed(items)]
        arrays = {row["output"]["mask"]["transport_key"]: mask for row in rows}
        return httpx.Response(200, content=_response_payload(rows, arrays), headers={"Content-Type": NPZ_MEDIA_TYPE})

    results = _client(handler).predict_batch("artifact-1", [
        OracleInferenceItem("roi", "one", {"image": np.ones((2, 2), dtype=np.uint8)}),
        OracleInferenceItem("roi", "two", {"image": np.zeros((2, 2), dtype=np.uint8)}),
    ], task="mask_refinement")

    assert seen == {"path": "/v2/inference/models/artifact-1:predict", "content_type": NPZ_MEDIA_TYPE,
                    "fingerprint": fingerprint, "task": "mask_refinement"}
    assert len(results) == 2
    assert results[0].result["output"]["mask"].shape == (2, 2)


def test_v2_refinement_backend_preserves_provenance():
    fingerprint = "b" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        manifest = _request_manifest(request)
        item = manifest["items"][0]
        mask = np.array([[0, 1], [1, 0]], dtype=np.uint8)
        row = _row(manifest["request_id"], item["item_id"], "artifact-1", fingerprint, mask)
        return httpx.Response(200, content=_response_payload([row], {row["output"]["mask"]["transport_key"]: mask}))

    backend = OracleRoiRefinementBackend(_client(handler), "artifact-1", fingerprint=fingerprint)
    prediction = backend.refine_batch([RoiRefinementInput("detection-1", np.zeros((2, 2), dtype=np.uint8), np.ones((2, 2), dtype=np.uint8))])[0]

    assert prediction.metadata["oracle_model"]["artifact_id"] == "artifact-1"
    assert int(np.count_nonzero(prediction.mask)) == 2


def test_v2_fingerprint_conflict_has_specific_error():
    client = _client(lambda request: httpx.Response(409, text="changed"))
    try:
        client.warm_model("artifact-1", "c" * 64)
    except OracleFingerprintMismatchError:
        pass
    else:
        raise AssertionError("expected fingerprint mismatch")


def test_v2_retries_a_temporary_runtime_unavailable_response():
    fingerprint = "d" * 64
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, text="warming")
        manifest = _request_manifest(request)
        item = manifest["items"][0]
        mask = np.ones((1, 1), dtype=np.uint8)
        row = _row(manifest["request_id"], item["item_id"], "artifact-1", fingerprint, mask)
        return httpx.Response(200, content=_response_payload([row], {row["output"]["mask"]["transport_key"]: mask}))

    result = _client(handler).predict_batch(
        "artifact-1", [OracleInferenceItem("roi", "one", {"image": np.ones((1, 1), dtype=np.uint8)})],
        task="mask_refinement", fingerprint=fingerprint,
    )

    assert attempts == 2
    assert result[0].result["status"] == "ok"


def test_v2_predict_splits_frames_before_the_configured_byte_limit():
    fingerprint = "e" * 64
    calls = []
    image = np.random.default_rng(7).integers(0, 256, (128, 128), dtype=np.uint8)
    inputs = [OracleInferenceItem("roi", str(index), {"image": image}) for index in range(2)]

    def handler(request: httpx.Request) -> httpx.Response:
        manifest = _request_manifest(request)
        calls.append(manifest)
        mask = np.ones((1, 1), dtype=np.uint8)
        rows = [_row(manifest["request_id"], item["item_id"], "artifact-1", fingerprint, mask)
                for item in manifest["items"]]
        return httpx.Response(200, content=_response_payload(rows, {
            row["output"]["mask"]["transport_key"]: mask for row in rows
        }))

    client = _client(handler)
    prepared = client.prepare_items(inputs, request_id="size-check")
    single_size = len(encode_v2_inference_request(V2InferenceRequest(
        "size-check", "mask_refinement", prepared[:1], V2ResponseOptions(),
    )))
    client.config.max_payload_bytes = single_size + 1024
    results = client.predict_batch("artifact-1", inputs, fingerprint=fingerprint,
                                   task="mask_refinement")

    assert len(results) == 2
    assert [len(call["items"]) for call in calls] == [1, 1]
