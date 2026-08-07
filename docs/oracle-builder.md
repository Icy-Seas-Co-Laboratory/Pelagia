# Oracle Builder inference

Pelagia does not load or execute ML frameworks. Oracle Builder is the sole
model host for ROI mask refinement and future classification tasks.

Pelagia sends whole ROI crops and candidate masks in bounded NPZ batches.
Oracle Builder owns model preprocessing, tiling, batching, thresholding, and
immutable model provenance. Pelagia owns candidate selection, frame expansion,
residual discovery, overlap reconciliation, measurements, job state, and
storage.

## Configuration

```toml
[oracle]
enabled = true
base_url = "http://127.0.0.1:8100"
default_mask_model = "pelagia-refiner"
connect_timeout_seconds = 5
read_timeout_seconds = 120
max_items_per_request = 32
max_payload_bytes = 268435456
```

Use `PELAGIA_ORACLE_API_TOKEN` for the optional bearer token. Do not put service
credentials in browser configuration or processing presets.

Start Oracle Builder before Pelagia refinement workers:

```bash
cd ../oracle-builder
oracle-serve --model pelagia-refiner=/absolute/path/to/sealed/run --port 8100
```

`GET /roi-refinement/options` reports Oracle availability and registered mask
models through Pelagia. PelagiaView never connects to Oracle directly.

Oracle outages fail immediate requests with a service error and cause queued
jobs to follow normal Pelagia retry policy. There is no silent identity-model
fallback. Successful refined detections record Oracle request/result IDs,
artifact/run identity, fingerprint, input hash, threshold, transforms, and
execution timing.
