from __future__ import annotations

import io
import json

import httpx
import numpy as np

from Pelagia.config import OracleConfig
from Pelagia.processing.detection_refinement import RoiRefinementInput
from Pelagia.processing.oracle_client import (
    NPZ_MEDIA_TYPE,
    OracleInferenceClient,
    OracleRoiRefinementBackend,
)


def _response_payload(mask: np.ndarray) -> bytes:
    manifest = {
        "schema_name": "oracle_builder.inference_result_set",
        "schema_version": "1.0.0",
        "counts": {"requested": 1, "succeeded": 1, "rejected": 0, "failed": 0},
        "results": [
            {
                "result_id": "result-1",
                "result_set_id": "result-set-1",
                "status": "ok",
                "input_sha256": "input-hash",
                "model": {
                    "artifact_id": "artifact-1",
                    "run_id": "run-1",
                    "task": "segmentation",
                    "architecture": "unet",
                    "contract_version": "1.0.0",
                },
                "execution": {"duration_ms": 12.5},
                "output": {
                    "type": "mask_refinement",
                    "mask": {"transport_key": "mask_0"},
                    "threshold": {"value": 0.5, "source": "artifact"},
                },
            }
        ],
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    buffer = io.BytesIO()
    np.savez_compressed(
        buffer,
        manifest=np.frombuffer(encoded, dtype=np.uint8),
        mask_0=mask,
    )
    return buffer.getvalue()


def test_oracle_client_uses_npz_contract_and_preserves_model_provenance():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["content_type"] = request.headers["content-type"]
        with np.load(io.BytesIO(request.content), allow_pickle=False) as archive:
            request_manifest = json.loads(
                np.asarray(archive["manifest"], dtype=np.uint8).tobytes().decode("utf-8")
            )
            seen["item_count"] = len(request_manifest["items"])
        mask = np.zeros((4, 4), dtype=np.uint8)
        mask[1:3, 1:3] = 1
        return httpx.Response(
            200,
            content=_response_payload(mask),
            headers={"Content-Type": NPZ_MEDIA_TYPE},
        )

    config = OracleConfig(base_url="http://oracle.test", default_mask_model="refiner-v1")
    client = OracleInferenceClient(config)
    client._client = httpx.Client(
        base_url=config.base_url,
        transport=httpx.MockTransport(handler),
    )
    image = np.arange(16, dtype=np.uint8).reshape(4, 4)
    candidate = np.zeros((4, 4), dtype=np.uint8)
    candidate[1:3, 1:3] = 255
    backend = OracleRoiRefinementBackend(client, "refiner-v1")

    predictions = backend.refine_batch(
        [RoiRefinementInput("detection-1", image, candidate)]
    )

    assert seen == {
        "path": "/v1/models/refiner-v1:predict",
        "content_type": NPZ_MEDIA_TYPE,
        "item_count": 1,
    }
    assert predictions[0].metadata["oracle_model"]["artifact_id"] == "artifact-1"
    assert predictions[0].metadata["oracle_threshold"]["value"] == 0.5
    assert int(np.count_nonzero(predictions[0].mask)) == 4
