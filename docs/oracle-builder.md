# Oracle Builder inference

Pelagia owns ROI inputs, project scope, processing jobs, human labels, and
evidence storage. Oracle Builder owns sealed model artifacts, model execution,
preprocessing, tiling, and thresholding. Pelagia never imports TensorFlow or
Keras. The default ROI refiner is Pelagia's CPU-only `heuristic_edge_v1`.

## Interactive V2 contract

Pelagia sends bounded batches to Oracle Builder's Orchestrator at
`POST /v2/inference/models/{artifact_id}:predict`. The former `/v1/models`
resident HTTP service and `oracle-serve` are retired. Pelagia selects a sealed
artifact by its catalog UUID, reads its fingerprint and capabilities, and pins
the fingerprint on inference requests. A display name is never a selector.
The Python environment requires `oracle-data-contracts` 0.3.x for the shared
V2 NPZ encoder and decoder; upgrade that package with this Pelagia release.

Image models for classification, embedding, and clustering share one V2 request
task (`classification`) and one versioned image-output shape. Pelagia requests
the fields supported by the selected artifact and records its per-item status,
request/item IDs, input hash, model fingerprint, output schema version, and
project-scoped evidence. A null output field is not a human label. Segmentation
models use the V2 `mask_refinement` task; Oracle returns the final reconstructed
mask for candidate-delta models. Pelagia retains frame expansion, residual
discovery, overlap reconciliation, measurement, and refined-ROI publication.

```toml
[oracle]
enabled = true
base_url = "http://127.0.0.1:8110" # Orchestrator, not the private runtime
default_mask_model = ""            # sealed catalog artifact UUID, if desired
default_classification_model = ""  # sealed catalog artifact UUID, if desired
connect_timeout_seconds = 5
read_timeout_seconds = 120
max_items_per_request = 64
max_payload_bytes = 33554432
max_output_bytes_per_item = 33554432
retry_attempts = 2
```

Set `PELAGIA_ORACLE_API_TOKEN` to an Oracle Builder analyst-or-higher role
token. Keep it server-side. The Oracle Builder stack must run its Orchestrator
and private inference runtime; see its `docs/inference-api.md` for runtime and
accelerator placement. The private runtime is not a Pelagia endpoint.

The HTTP V2 path returns one result per submitted item. Pelagia records progress
after each completed batch. Oracle Builder does not currently publish a
separate progress message queue. On an unknown transport outcome, Pelagia may
repeat the same request with the same correlation IDs and pinned fingerprint;
evidence writes preserve the first stored result per run and ROI. A fingerprint mismatch
requires reselecting the intended artifact. Oversized or invalid requests must
be corrected before retrying.

`GET /roi-refinement/options` and `GET /curation/options` expose compatible
catalog models through Pelagia. PelagiaView does not call Oracle Builder.
Oracle outages fail learned inference without silently substituting an
identity model. Existing historical evidence remains readable.

Oracle Builder's durable V1 inference runs remain available for explicit
frozen-dataset batch operations. Pelagia's operational ROI jobs use V2 because
Pelagia owns and batches those inputs.
