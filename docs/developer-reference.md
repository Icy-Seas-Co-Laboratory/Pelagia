# Pelagia developer reference

[Documentation index](index.md)

## System Architecture

Pelagia keeps interface code thin and moves reusable behavior into shared services and processing modules.

```text
API / CLI / Workers
        |
        v
     Services
        |
        +--> Storage adapters
        |      - Postgres metadata, queue, worker sessions, events
        |      - Cold payload storage for large frame data
        |
        +--> Processing routines
               - frame extraction
               - segmentation
               - ROI storage
               - correction, encoding, and analysis helpers
```

The repository layout follows that split:

```text
Pelagia/
  config.py              typed config loaded from TOML/env/CLI overrides
  domain.py              stable records, enums, and row models
  storage/               persistence adapters and SQL schema resources
  processing/            image and model processing routines
  services/              application workflows shared by interfaces
  workers/               job claiming, dispatch, heartbeat, and execution
  api/                   FastAPI application and route modules
  cli/                   command-line entrypoint and commands
  utils/                 small shared helpers
docs/assets/             documentation images and branding assets
```

## Core Capabilities

The current system is organized around an ROI-centered pipeline:

- **Video, image, and interchange ingestion**: register source assets, extract or directly import frames, store full-frame payloads in cold storage, and record searchable metadata.
- **Telemetry and timeline context**: preserve native-resolution sensor observations and point/interval events, then resolve them against frame and ROI timestamps without duplicating values into image records. See [Telemetry ingestion and lookup](../docs/telemetry.md).
- **Frame correction**: apply image normalization such as flatfield correction before storage or analysis.
- **Segmentation**: detect candidate ROIs from extracted frames, store ROI crops, masks, geometry, and image statistics.
- **Segmentation refinement**: run the CPU heuristics bundled with Pelagia or call an optional learned mask-refinement artifact through Oracle Builder.
- **Labeling and classification**: retain Oracle predictions as model evidence and keep human annotations as explicit project-owned assertions; a prediction is never silently promoted to ground truth.
- **Clustering and exploration**: retain Oracle embeddings and clustering evidence and expose bounded exploratory projections such as UMAP; exploratory groups are not labels or taxonomy claims.
- **Training set curation**: review ROIs, accept or replace predictions with an explicit human action, preserve annotation history, and export curated products.
- **Model training and evaluation**: Oracle Builder owns TensorFlow/Keras training, sealed model artifacts, and evaluation evidence; Pelagia consumes selected artifacts and records project-scoped results and provenance.
- **Export**: produce robust exports for ROIs, masks, labels, measurements, embeddings, model outputs, and curated datasets.
- **Workers and provenance**: run independent worker processes per job type while retaining durable job events, project scope, input context, and output references.

## Data Model

Pelagia separates runtime image data from persistent row models:

- `FrameData` is the runtime container for image arrays, masks, geometry, and source metadata while processing.
- `FrameRecord` is the database row model for a stored frame, including geometry and a payload reference for large frame data.
- `DetectionRecord` is the database row model for an ROI, including object bounds, padded crop bounds, ROI payload, mask payload, and measurements.
- `raw_assets.collections` is a first-class grouping field assigned at ingestion. Values can be provided as a comma-separated string or list, are normalized into a non-empty array, and default to `none` when unspecified.

This keeps the boundary clear: large source frames can remain cold and only be fetched when needed, while compact ROI-centered data stays available for analysis and curation.

### Importing Pelagia interchange datasets

The Ingestion page uses one Analyze and queue workflow for interchange
datasets, video files, and image-sequence folders. Select a path to search
recursively; every complete interchange directory found beneath it appears as
one analyzed asset alongside any videos and image folders. A complete package
can also be queued directly:

```bash
pelagia queue_ingest /data/deployment_042_interchange
```

Pelagia recognizes the package from its manifest and required files, registers
it as one raw asset, and uses the dataset title as the collection name unless a
collection is supplied explicitly. Frame counts and a manifest-based package
size estimate are available during assessment without scanning the image
shards. The queued import creates one worker job per shard, so multiple extract
workers can import shards concurrently. Each job reads only its assigned shard;
the package remains one raw asset. Background and flatfield windows and
multi-frame tiles close at shard boundaries, even when the same source file
continues in the next shard. Tiling stacks consecutive compatible frames
from the same source vertically within a shard; each tile records its member
frame provenance.
Full frame reads begin in the queued import jobs. The assessment does
not establish package integrity; run interchange validation when that
assurance is needed. Dataset metadata, source-file identities, stream
identities, frame timestamps, timestamp provenance, and original frame
numbers are retained. Compatible JPEG, PNG, JPEG XL, JPEG XS, raw array, and
Zstandard array frame payloads are copied
directly into cold SkerryKV storage without re-encoding, even when the project has a different
default frame encoding. Unsupported but decodable image codecs fall back to the
project's configured frame encoding. Each frame is
decoded once for geometry validation and its browser thumbhash.

Interchange imports disable background and flatfield generation by default for
a fast archival-to-operational load. They can be enabled explicitly when those
derived products are wanted. Absolute per-frame timestamps are preferred;
otherwise Pelagia uses a source start timestamp or a timestamp parsed from the
original source filename and advances it using the declared rational frame
rate. All original timestamp fields remain available in frame metadata even
when they cannot safely be converted to `captured_at`.

## Jobs And Workers

Pelagia uses Postgres-backed jobs to coordinate long-running processing. Workers are independent processes that claim jobs for specific stages, heartbeat while active, write job events, and can be stopped through worker-session state.
Workers defer new claims while available host or container memory is below 20%.
They resume claiming automatically after memory recovers. Set
`PELAGIA_QUEUE_MIN_AVAILABLE_MEMORY_FRACTION` to adjust the threshold (for
example, `0.25` for 25%). Running jobs finish their current work, so worker
concurrency still needs to fit the server's peak memory budget.

Each claim carries a rotated lease token. Workers use that token when renewing a
lease, reporting progress, or finishing work, so a worker that loses a lease
cannot overwrite a newer claimant's state. For multi-stage processing series,
run `reconcile_processing_series` from one scheduled supervisor; it
idempotently resumes terminal-ready series and conservatively returns abandoned
empty planning steps to the director queue. The command takes a PostgreSQL
advisory lock, requeues expired leases, materializes durable successor
dispatches, and reports `retry_scheduled`, `dead_lettered`, `expired_leases`,
`pending_dispatches`, and `failed_dispatches` counters. Schedule it frequently
enough for the desired recovery time and alert on non-zero dead letters, failed
dispatches, and persistently increasing lease-expiry counts.

For a systemd deployment, install
`scripts/systemd/pelagia-job-supervisor.service` and
`scripts/systemd/pelagia-job-supervisor.timer`, adjust the user, paths, and
`/etc/pelagia/pelagia.env`, then run `systemctl enable --now
pelagia-job-supervisor.timer`. The timer invokes a short-lived supervisor every
minute; the database advisory lock remains the singleton guard across manual
invocations and multiple hosts. Inspect each run with `systemctl status
pelagia-job-supervisor.service` and use the command's JSON counters for alerts.

Example worker roles:

```text
extract_frames workers     video/image frame extraction
segment workers            ROI detection and mask generation
classify workers           ROI labeling with trained models
curation workers           embedding, clustering, and dataset refinement tasks
export workers             dataset/model/result export tasks
training workers           model training and evaluation jobs
```

This architecture allows the system to grow from one local process to many specialized background daemons without changing the shape of the pipeline.

## HTTP API

The v0 API exposes the same core workflows as the CLI and workers:

```bash
uvicorn Pelagia.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

When a reverse proxy exposes the API below a prefix, set its externally visible
ASGI root path in configuration or the environment, for example
`PELAGIA_API_ROOT_PATH=/pelagia-api`. The proxy should strip that prefix before
forwarding requests to Pelagia.

For local end-to-end testing, the dev stack script initializes storage, starts
the API, and starts workers for currently runnable stages:

```bash
./scripts/pelagia_dev_stack.sh start
./scripts/pelagia_dev_stack.sh status
./scripts/pelagia_dev_stack.sh stop
```

Override defaults with environment variables such as `PELAGIA_DATABASE_DSN`,
`PELAGIA_DATABASE_SCHEMA`, `PELAGIA_COLD_STORAGE_ENDPOINT`, `PELAGIA_HOT_STORAGE_ENDPOINT`, `PELAGIA_API_PORT`, `PELAGIA_API_ROOT_PATH`,
`PELAGIA_RUN_DIR`, and `PELAGIA_WORKER_STAGES`. Use `PELAGIA_WORKER_COUNT=3` to
start three workers for every configured stage, or
`PELAGIA_WORKER_COUNTS=extract_frames=2,segment=4` for per-stage counts.

Useful endpoint groups:

- `GET /health`, `/health/postgres`, `/health/kvstore`
- `POST /auth/login`, `GET /auth/me`, `POST /auth/logout`, `POST /auth/switch-project`
- `GET /projects`, `POST /projects`, `DELETE /projects/{project_id_or_key}`
- `GET /users`, `POST /users`, `POST /users/{user_id_or_username}/reset-password`,
  `POST /users/{user_id_or_username}/deactivate`, `DELETE /users/{user_id_or_username}`
- `GET /system`, `/system/status`, `/system/status/{project_id_or_key}`, `/system/usage`, `/system/use`, `/system/config`
- `POST /system/initialize`
- `POST /ingestion/analyze`, `POST /ingestion/assets`, `POST /ingestion/videos`
- `POST /frame/preprocess`, `POST /frame/preprocess/jobs`
- `GET /frame/original`, `GET /frame/preprocessed`
- `POST /segmentation/frames/{frame_id}`
- `GET /segmentation/options`, `POST /segmentation/jobs`
- `GET /live/threshold`, `GET /live/detection-candidate`
- `GET /live/sandbox`, `DELETE /live/sandbox/{sandbox_frame_id}`
- `GET /roi-refinement/options`, `POST /roi-refinement`, `POST /roi-refinement/jobs`
- `GET /jobs`, `GET /jobs/summary`, `POST /jobs`, `GET /jobs/events`
- `POST /jobs/{job_id}/pause`, `/resume`, `/retry`, `/priority`
- `GET /workers`, `POST /workers/{worker_id}/shutdown`
- `GET /runs`, `/assets`, `/models`, `/kvstore` (`/models` records model provenance; inference discovery lives in Oracle Builder)
- `GET /collections`, `GET /assets?collection=test`, `GET /runs?collection=test`
- `GET /frames/{frame_id}/context`
- `GET /detections`, `/detections/{detection_id}/framedata`, `/mask`, `/refined-roi`, `/refined-mask`
- `GET /logs`
- `GET /exports/options`, `POST /exports`, `GET /exports/{export_id}`, `/download`

Detection list and detail responses expose the parent frame's nullable
`captured_at` timestamp. ROI metadata exports and curation ROI responses expose
the same value; ROI processing and classification timestamps remain separate
operational `created_at` fields.

Ingestion requests can generate aligned fields while decoded frames are already
in memory. Area-scan ingestion normally uses `generate_backgrounds=true`, while
line-scan ingestion normally uses `generate_flatfield_profiles=true`; callers
may explicitly enable both. Area-scan means use `background_window_stride` and
`background_window_width`, which default to `25/25`. Line-scan profiles use
`flatfield_window_stride` and `flatfield_window_width`, which default to `1/1`.
The schedules are independent when both products are enabled. Mean background
images are written to the project KV
store, while column-mean flatfield profiles are stored directly as PostgreSQL
`real[]` values and can be read from
`GET /frames/{frame_id}/flatfield-profile`. Non-nominal partial frames receive
the applicable field references. They are excluded from full-frame background
means, but can still contribute to flatfield profiles. `flatfield_axis=0` is the
default: frames are effectively stacked vertically, their widths must match, and
column means are stored. With `flatfield_axis=1`, frames are effectively stacked
horizontally, their heights must match, and row means are stored. The stacking is
accumulated without allocating a concatenated image. Ingestion field window
widths and strides may be any positive integers with stride no larger than
width. For example, width 4 and stride 4 combine four consecutive frames
per flatfield profile; axis 0 averages their stacked columns and axis 1
averages their stacked rows. A final incomplete window uses the available
frames and records its contributing frame IDs. Windows count stored frames;
when multi-frame tiling is enabled, a window of 4 spans four stored tiles.

Preprocessing no longer estimates correction fields. It automatically applies
an ingestion-generated mean background when one is attached to the frame;
otherwise it applies the stored flatfield profile. If neither field exists,
preprocessing continues with crop, mask, and configured intensity inversion
without correction. Preprocessing and segmentation requests now expose only
`min_field_value`, `max_field_value`, `apply_mask`, and the `crop_*` controls for
this stage. Quantile, axis, correction-toggle, ad-hoc background-selection, and
inversion request parameters have been removed.
At ingestion, video and interchange assets may set `source_foreground_polarity`
to `dark` or `bright`. Dark foreground is inverted during preprocessing;
bright foreground is kept. Assets without this value use the configured
`invert_intensity` default. Source frames remain unchanged.
When neither switch is supplied, effective correction settings choose one mode,
with background generation taking precedence over flatfield profiles.

General list endpoints are intentionally shaped as limited searches. For example,
`GET /assets` does not require a run id and can be narrowed with filters such as
`collection`, `kind`, `filename`, `path`, `checksum`, size bounds, and `limit`.
Frame metadata can be searched by range with
`GET /assets/{asset_id}/frames?start_frame=1&end_frame=100`, and frame image data
can be loaded with `GET /assets/{asset_id}/framedata/{frame_num}?format=png`.
Supported frame data response formats are `png`, `jpg`/`jpeg`, `jxl`, `jxs`, `matrix`,
and `preview`. Stored full-frame payloads default to `jpg` at quality `75`; set
`[processing.frame_storage].image_encoding = "jxl"` or `"jxs"` to store JPEG XL
or JPEG XS payloads. Both codecs use in-memory `imagecodecs` bindings. JPEG XS
requires an `imagecodecs` source build with `libjxs` enabled because binary wheels
do not include that optional codec.
Preview requests return a small PNG placeholder and accept `preview_max_dim`, for example
`GET /assets/{asset_id}/framedata/{frame_num}?format=preview&preview_max_dim=128`.

Frame metadata carries standard ThumbHash placeholders as
`preview_thumbhash_base64` (and `preprocessed_preview_thumbhash_base64` when
available). Their encoder input is controlled separately by
`[processing.thumbhash].max_dim` / `PELAGIA_THUMBHASH_MAX_DIM`: an integer from
1 to 100, default 100. Generation preserves aspect ratio without upscaling,
resizes grayscale/BGR/BGRA data with bilinear interpolation before RGBA expansion,
and passes a compact byte buffer to the encoder. Non-uint8 inputs retain
full-source min/max normalization and uint8 quantization before resizing.
Bilinear resizing prioritizes throughput for these graphical placeholders;
aliasing and missed small features are acceptable. Lower limits reduce encoder
work but can change placeholder appearance.
Encoding uses the pinned `fast-thumbhash-rs` native implementation, which borrows
the RGBA byte buffer and releases the GIL during encoding. Transparent BGRA is
supported. New hashes remain standard ThumbHash payloads, but floating-point
differences can change bytes compared with the former Python encoder; they are
display placeholders, not content identifiers. Very small or single-row/column
inputs can also show larger decoded differences from numerical rounding in the
undersampled transform. Existing stored hashes are left intact. The encoder
version is pinned in `pyproject.toml` and `uv.lock` for
reproducibility.

Update environments with `./scripts/pelagia_env sync dev` (or `sync cpu`). The
former `thumbhash` distribution and `fast-thumbhash-rs` share the `thumbhash`
import namespace and must not be installed together. The managed sync removes
the old distribution. Version 0.1.0 provides Python 3.12 wheels for macOS arm64
and Linux x86_64 (glibc 2.28+); Linux arm64 requires a Rust source build or an
internally built wheel.

Benchmark the actual production image dimensions with
`.venv/bin/python scripts/benchmark_thumbhash.py --input frame.png`. The JSON
result separates the encoder call (including the Python/Rust boundary) from
complete ThumbHash generation; image decoding is excluded. Without `--input`,
use `--width`, `--height`, `--channels`, and `--dtype` for seeded synthetic data.
Use `--threads` to request production OpenCV parallelism; the result records both
requested and reported counts because some OpenCV backends ignore this setting.
Non-uint8 normalization still scans the full source image and can dominate
the resize and native encoder costs.

## Configuration

Pelagia loads configuration in this order:

```text
Pelagia/default.config.toml < ./config.toml < environment variables < explicit CLI options
```

The packaged [default.config.toml](../Pelagia/default.config.toml) contains development-friendly defaults. Create a local `config.toml` in the repository root for machine-specific overrides; it is ignored by git.

Video ingest defaults to one OpenCV thread and one decoder thread per worker to
avoid CPU oversubscription when worker count tracks CPU core count. Benchmark
larger values with:

```toml
[processing.video_ingest]
opencv_threads = 1
decoder_threads = 1
```

The equivalent environment variables are
`PELAGIA_VIDEO_INGEST_OPENCV_THREADS` and
`PELAGIA_VIDEO_INGEST_DECODER_THREADS`. The decoder setting applies to both
OpenCV's FFmpeg backend and the FFmpeg CLI fallback.

ROI storage uses an explicit small/large image policy. Size is the padded crop
width multiplied by height; masks have an independent lossless codec. Candidate
masks are stored only when the ROI image meets the payload storage thresholds
and `always_store_mask` is enabled. Candidates without a stored ROI crop are
excluded from refinement and from neighboring seeds. For eligible candidates
without a stored mask, refinement can derive a seed from the ROI crop.
An installation can restrict the codecs advertised and accepted by the API:

```toml
[image_data_storage]
allowed_encodings = ["zstd", "png", "jpg"]

[processing.roi_recording]
small_roi_encoding = "zstd"
large_roi_encoding = "jpg"
large_roi_min_pixels = 1000
roi_quality = 75
mask_encoding = "zstd"
```

Every configured frame, ROI, and mask codec must be present in the deployment
allowlist. The equivalent comma-separated allowlist environment variable is
`PELAGIA_IMAGE_DATA_STORAGE_ALLOWED_ENCODINGS`.
To re-encode existing project images after changing storage settings, use
**Update datastores** in PelagiaView Administration; see
[`docs/storage-reencode.md`](../docs/storage-reencode.md) for scope and recovery behavior.

Model and plugin artifacts are split between packaged assets under
`Pelagia/assets/` and a local runtime library under `./.pelagia/`. See
[docs/artifacts.md](../docs/artifacts.md) for the manifest layout and discovery
rules.

For storage maintenance, migration, and recovery procedures, use:

- [End-User Operations Guide](../docs/end-user-support.md)
- [Resetting Pelagia](../docs/reset-system.md)
- [Migrating Pelagia](../docs/migration.md)
- [Backup And Restore](../docs/backup.md)
- [ROI Curation And ML Evidence](../docs/curation.md)
- [Telemetry Ingestion And Lookup](../docs/telemetry.md)

## Python Environment

Pelagia uses a cross-platform `uv.lock` and separate runtime profiles:

```bash
./scripts/pelagia_env sync cpu       # API and all workers
./scripts/pelagia_env sync dev       # CPU profile plus tests
```

The legacy requirements files remain available as compatibility inputs, but
new installations should use uv profiles. See
[docs/python-environment.md](../docs/python-environment.md) for details.

## Storage Strategy

Pelagia treats storage as a replaceable adapter layer.

- Postgres stores durable metadata, jobs, worker sessions, events, frame records, ROI records, model records, and classification results.
- Large full-frame payloads are stored by a cold payload store adapter because they are expensive and usually read infrequently.
- ROI crops and masks are stored in hot SkerryKV, with their references and metadata in Postgres.

Cold and hot storage can use the same SkerryKV service or separate instances. See [SkerryKV operations](../docs/skerrykv.md) and [migration instructions](../docs/skerry-migration.md). Legacy KVStore implementations remain available for copying existing stores.

`KVStore2` reuses up to 16 idle SQLite index connections per thread, evicting the
least recently used connection when the cache is full. A later connection
acquisition closes caches belonging to finished threads. Call `KVStore2.close()`
when a standalone store handle is no longer needed; it waits for active
operations, closes cached connections, and makes the handle unusable. Application
context shutdown closes its stores automatically. The internal `_connect` method
is a context manager that commits or rolls back transactions; connections retain
the existing WAL journal mode and `synchronous=NORMAL` setting.

## Development Direction

The near-term goal is to keep Pelagia small at the interfaces and strong in the pipeline core:

- keep API, CLI, and worker entrypoints thin;
- keep processing routines focused and testable;
- keep row models aligned one-to-one with database records;
- keep storage adapters replaceable;
- keep job history explicit through `job_events`;
- build curation, training, and export features on top of the same ROI-centered data model.
