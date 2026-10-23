# Oracle Builder inference

Pelagia owns ROI inputs, project scope, processing jobs, human labels, and
evidence storage. Oracle Builder owns sealed model artifacts, model execution,
preprocessing, tiling, and thresholding. Pelagia never imports TensorFlow or
Keras. The default ROI refiner is Pelagia's CPU-only `heuristic_edge_v1`.
All three built-in heuristics finish with versioned, padded-border hole filling
and validate the final binary mask before publication. Ordinary enclosed holes
are filled. If an enclosed background component contains another candidate's
seed, the component remains background and the audit records it as a protected
hole so candidate identity is not destroyed. This behavior is algorithm
version 2 for `heuristic_edge_v1`, version 3 for `heuristic_v2`, and version 4
for `heuristic_v3`.
Select `method="heuristic_v2"` to expand each original candidate seed through
successively lower intensity thresholds. It scores boundary contrast and growth
stability, retains every seed pixel, and excludes neighboring candidate seeds.
Candidate generation records a scalar foreground threshold when its method has
one; v2 uses that threshold for stored processed ROI payloads and otherwise
infers a starting level from the seed. The original candidate mask is supplied
again on every crop-expansion iteration for all refiners.
Select `method="heuristic_v3"` for a seed-constrained minimum cut over pixel
intensity and eight-neighbor gradient costs. V3 allows different portions of
the outer boundary to have different intensities. It keeps candidate pixels
immutable, excludes neighboring seeds, and removes disconnected foreground
islands after optimization. `v3_max_graph_pixels` bounds graph memory; crops
that exceed it fail explicitly so the search limit can be adjusted.

Every refinement method runs a split/merge assessment before publication.
Configured residual discovery and overlap reconciliation then apply, except
that all built-in heuristics retain separate candidate identities. After
publication, Pelagia automatically reassesses cross-frame connections for
assets declared as line scans; other assets are skipped. The assessment run is
recorded with the refinement result. Ingestion declares this geometry with
`metadata.line_scan_geometry={"scan_axis":"x"}` or `{"scan_axis":"y"}`;
Pelagia validates and stores that canonical field on the asset and extracted
frames. Queued refinement retries reuse each refined output by its parent
candidate and ROI index, preserving distinct split/residual children, and reuse
the continuity run already written by the same job, including when continuity
fails after the refined rows commit. Direct refinement reports a
continuity-specific error in `connection_assessment` while retaining its
successfully stored ROI results. That error names the continuity-only retry
endpoint; it does not ask clients to repeat refinement. A failure to refresh
the processing-status projection is reported separately in
`status_publication` for the same reason.

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
