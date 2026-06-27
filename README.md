<p align="center">
  <img src="docs/assets/pelagia_logo.png" alt="Pelagia" width="520">
</p>

# Pelagia

Pelagia is a scalable image-analysis system for extracting, segmenting, organizing, labeling, and training models on biological image data. It is built around video-frame ingestion, ROI-first processing, durable metadata, background workers, and reproducible data products.

<p align="center">
  <img src="docs/assets/pelagia_icon.png" alt="Pelagia icon" width="96">
</p>

Pelagia is designed for workflows where full source frames are large and mostly cold, while segmented ROIs are the primary unit of analysis. Full-frame payloads can live in external cold storage, while ROI crops, masks, measurements, classifications, and curation state remain close to the database-backed analysis workflow.

## Quick Start

Pelagia is split into a Python backend (`Pelagia`) and an optional SvelteKit
frontend (`PelagiaView`). The backend owns storage, processing, the job queue,
workers, and the HTTP API. PelagiaView connects to that API from a browser.

For operational procedures after installation, see the
[End-User Operations Guide](docs/end-user-support.md). It links the reset,
migration, KVStore migration, backup, and restore procedures that keep the
database and blob storage in sync.

### Prerequisites

- Python 3.10 or newer. Python 3.11+ is preferred.
- PostgreSQL reachable from the Pelagia machine.
- A writable runtime directory for logs, pid files, and cold frame storage.
- Node.js 18+ and npm if you will run PelagiaView.

On Debian/Ubuntu systems, these packages are commonly useful before creating the
Python environment:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip postgresql postgresql-client libgl1 libglib2.0-0
```

### Install The Backend

From the Pelagia repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For development and tests, install the dev target instead:

```bash
python -m pip install -r requirements-dev.txt
```

For machines that will run learned ROI refinement models, install the optional
ML dependencies after the normal backend install:

```bash
python -m pip install -r requirements-ml.txt
```

Apple Metal users should use `requirements-ml-apple-metal.txt` instead of
`requirements-ml.txt`.

### Configure Storage

Pelagia defaults to a local PostgreSQL database named `pelagia` with this DSN:

```text
postgresql://postgres:postgres@127.0.0.1:5432/pelagia
```

Either create/configure a database matching that DSN, or override the DSN in
`config.toml`, environment variables, or the worker-stack TOML.

Create a local `config.toml` for machine-specific backend settings:

```toml
[database]
dsn = "postgresql://postgres:postgres@127.0.0.1:5432/pelagia"
schema_name = "pelagia"

[kvstore]
backend = "kvstore" # Options: kvstore, kvstore2
root_path = "./data/kvstore"
hash_algorithm = "blake3"
prefix_length = 2
max_db_bytes = 42949672960   # Used by kvstore.
max_rows = 1000000           # Used by kvstore.
max_blob_bytes = 67108864    # Used by kvstore2.

[api]
host = "0.0.0.0"
port = 8000
cors_allow_origin_regex = "https?://(localhost|127\\.0\\.0\\.1|10(?:\\.\\d{1,3}){3}|100\\.(?:6[4-9]|[7-9]\\d|1[01]\\d|12[0-7])(?:\\.\\d{1,3}){2}|192\\.168(?:\\.\\d{1,3}){2}|172\\.(?:1[6-9]|2\\d|3[01])(?:\\.\\d{1,3}){2})(?::\\d+)?"

[auth]
enabled = true
session_ttl_seconds = 604800
dev_project_key = "default"
```

`config.toml` is ignored by git. You can also use environment variables such as
`PELAGIA_DATABASE_DSN`, `PELAGIA_DATABASE_SCHEMA`, `PELAGIA_KVSTORE_BACKEND`,
`PELAGIA_KVSTORE_ROOT`,
`PELAGIA_API_HOST`, `PELAGIA_API_PORT`, `PELAGIA_AUTH_ENABLED`,
`PELAGIA_AUTH_SESSION_TTL_SECONDS`, and `PELAGIA_AUTH_DEV_PROJECT_KEY`.

### Initialize And Run The Backend

Initialize database tables and storage:

```bash
python -m Pelagia.cli.app init-system
python -m Pelagia.cli.app create-dev-login --username dev-admin --password pelagia-dev
```

`create-dev-login` creates the default project if needed, creates or reuses the
admin user, adds project membership, and prints a JSON response containing a
session token. For manual setup use:

```bash
python -m Pelagia.cli.app create-user ada --password secret --admin
python -m Pelagia.cli.app create-project field-survey --project-name "Field Survey"
python -m Pelagia.cli.app add-project-user ada field-survey --role editor
python -m Pelagia.cli.app list-projects --username ada
```

For a local one-command backend stack, start the API and workers:

```bash
./scripts/pelagia_dev_stack.sh start
./scripts/pelagia_dev_stack.sh status
```

Stop the stack when finished:

```bash
./scripts/pelagia_dev_stack.sh stop
```

The development stack can be adjusted with environment variables:

```bash
PELAGIA_API_HOST=0.0.0.0 \
PELAGIA_API_PORT=8000 \
PELAGIA_WORKER_COUNT=2 \
./scripts/pelagia_dev_stack.sh start
```

For a more explicit worker layout, use the TOML-driven stack:

```bash
cp scripts/pelagia_workers.example.toml scripts/pelagia_workers.toml
./scripts/pelagia_stack_from_toml.sh start scripts/pelagia_workers.toml
./scripts/pelagia_stack_from_toml.sh status scripts/pelagia_workers.toml
./scripts/pelagia_stack_from_toml.sh stop scripts/pelagia_workers.toml
```

Runtime logs and pid files are written under the stack run directory. By
default, the TOML stack uses:

```text
<repo>/.pelagia/run/<stack-name>/
```

On shared Linux machines, set an explicit writable run directory. For example,
if the repo is cloned at `/scratch/Pelagia`:

```bash
export PELAGIA_RUN_DIR=/scratch/Pelagia/.pelagia/run/dev-workers
./scripts/pelagia_stack_from_toml.sh start scripts/pelagia_workers.toml
```

You can also put it in `scripts/pelagia_workers.toml`:

```toml
[stack]
run_dir = "/scratch/Pelagia/.pelagia/run/dev-workers"
```

### Verify The API

After the stack starts:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/system/status
curl -H "Authorization: Bearer $PELAGIA_TOKEN" http://127.0.0.1:8000/system/status/default
curl -H "Authorization: Bearer $PELAGIA_TOKEN" http://127.0.0.1:8000/jobs
```

To expose the API to another computer, bind to `0.0.0.0`, make sure the OS
firewall allows the API port, and set `cors_allow_origin_regex` so the browser
origin running PelagiaView is allowed.

### Run PelagiaView

From a sibling checkout named `PelagiaView`:

```bash
cd ../PelagiaView
npm install
npm run dev -- --host 0.0.0.0 --port 5173
```

Open the UI at:

```text
http://127.0.0.1:5173
```

Then connect to the backend API endpoint, for example:

```text
http://127.0.0.1:8000
```

PelagiaView should log in before loading project resources:

1. `POST /auth/login` with `username`, `password`, and either `project_key` or `project_id`.
2. Store the returned `token` for the browser session.
3. Send `Authorization: Bearer <token>` on API calls.
4. Use `GET /auth/me` to restore the active user/project after refresh.
5. Use `GET /projects` and `POST /auth/switch-project` to move between projects.
   `GET /projects?include_all_names=true` also includes `all_project_names`
   for global project-name pickers while leaving `projects` scoped to the user.
6. User admins can create projects with `POST /projects`. User admins and
   project managers/admins can soft-delete non-default projects with
   `DELETE /projects/{project_id_or_key}`.
7. Logged-in users can list users in their active project with `GET /users`.
   User admins can pass `include_all_projects=true` for a global user list.
8. User admins and project managers/admins can manage user accounts with
   `POST /users`, `POST /users/{user_id_or_username}/reset-password`,
   `POST /users/{user_id_or_username}/deactivate`, and
   `DELETE /users/{user_id_or_username}`. Project managers/admins are scoped
   to users in their active project; user admins can manage accounts globally.

With `auth.enabled=false`, the backend accepts requests without a token for
single-user local development, but still scopes all resource access to
`auth.dev_project_key`. New runs, assets, jobs, models, logs, and generated
artifacts are project-owned.

On another computer, use the backend machine's IP address for both PelagiaView
and the API endpoint.

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

Current and planned capabilities are organized around an ROI-centered pipeline:

- **Video and image ingestion**: register source assets, extract frames, store full-frame payloads in cold storage, and record searchable metadata.
- **Frame correction**: apply image normalization such as flatfield correction before storage or analysis.
- **Segmentation**: detect candidate ROIs from extracted frames, store ROI crops, masks, geometry, and image statistics.
- **Segmentation refinement**: support learned mask refinement models such as U-Net-style models for better ROI boundaries.
- **Labeling and classification**: attach model predictions and human labels to ROIs using CNN-style classifiers or related image models.
- **Clustering and exploration**: support unsupervised grouping with embeddings, PCA, k-means, and related tools for dataset discovery.
- **Training set curation**: refine labels, identify uncertain examples, manage balanced training sets, and track curation decisions.
- **Model training and evaluation**: train, register, compare, and archive segmentation and classification models.
- **Export**: produce robust exports for ROIs, masks, labels, measurements, embeddings, model outputs, and curated datasets.
- **Scalability**: run independent worker processes per job type, scale across CPU cores or machines, and keep a durable job/event history.

## Data Model

Pelagia separates runtime image data from persistent row models:

- `FrameData` is the runtime container for image arrays, masks, geometry, and source metadata while processing.
- `FrameRecord` is the database row model for a stored frame, including geometry and a payload reference for large frame data.
- `DetectionRecord` is the database row model for an ROI, including object bounds, padded crop bounds, ROI payload, mask payload, and measurements.
- `raw_assets.collections` is a first-class grouping field assigned at ingestion. Values can be provided as a comma-separated string or list, are normalized into a non-empty array, and default to `none` when unspecified.

This keeps the boundary clear: large source frames can remain cold and only be fetched when needed, while compact ROI-centered data stays available for analysis and curation.

## Jobs And Workers

Pelagia uses Postgres-backed jobs to coordinate long-running processing. Workers are independent processes that claim jobs for specific stages, heartbeat while active, write job events, and can be stopped through worker-session state.

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

For local end-to-end testing, the dev stack script initializes storage, starts
the API, and starts workers for currently runnable stages:

```bash
./scripts/pelagia_dev_stack.sh start
./scripts/pelagia_dev_stack.sh status
./scripts/pelagia_dev_stack.sh stop
```

Override defaults with environment variables such as `PELAGIA_DATABASE_DSN`,
`PELAGIA_DATABASE_SCHEMA`, `PELAGIA_KVSTORE_ROOT`, `PELAGIA_API_PORT`,
`PELAGIA_RUN_DIR`, and `PELAGIA_WORKER_STAGES`. Use `PELAGIA_WORKER_COUNT=3` to
start three workers for every configured stage, or
`PELAGIA_WORKER_COUNTS=extract_frames=2,segment=4` for per-stage counts.

Useful endpoint groups:

- `GET /health`, `/health/postgres`, `/health/kvstore`
- `POST /auth/login`, `GET /auth/me`, `POST /auth/logout`, `POST /auth/switch-project`
- `GET /projects`, `POST /projects`, `DELETE /projects/{project_id_or_key}`
- `GET /users`, `POST /users`, `POST /users/{user_id_or_username}/reset-password`,
  `POST /users/{user_id_or_username}/deactivate`, `DELETE /users/{user_id_or_username}`
- `GET /system`, `/system/status`, `/system/status/{project_id_or_key}`, `/system/use`, `/system/config`
- `POST /system/initialize`
- `POST /ingestion/videos`
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
- `GET /runs`, `/assets`, `/models`, `/kvstore`
- `GET /collections`, `GET /assets?collection=test`, `GET /runs?collection=test`
- `GET /frames/{frame_id}/context`
- `GET /detections`, `/detections/{detection_id}/framedata`, `/mask`, `/refined-roi`, `/refined-mask`
- `GET /logs`
- `GET /io/export/options`
- `GET /io/export/table/{table_name}`, `GET /io/export/tables`
- `GET /io/export/datasets/frame-metadata`, `/roi-metadata`

General list endpoints are intentionally shaped as limited searches. For example,
`GET /assets` does not require a run id and can be narrowed with filters such as
`collection`, `kind`, `filename`, `path`, `checksum`, size bounds, and `limit`.
Frame metadata can be searched by range with
`GET /assets/{asset_id}/frames?start_frame=1&end_frame=100`, and frame image data
can be loaded with `GET /assets/{asset_id}/framedata/{frame_num}?format=png`.
Supported frame data formats are `png`, `jpg`/`jpeg`, `matrix`, and `preview`.
Preview requests return a small PNG placeholder and accept `preview_max_dim`, for example
`GET /assets/{asset_id}/framedata/{frame_num}?format=preview&preview_max_dim=128`.

## Configuration

Pelagia loads configuration in this order:

```text
Pelagia/default.config.toml < ./config.toml < environment variables < explicit CLI options
```

The packaged [default.config.toml](Pelagia/default.config.toml) contains development-friendly defaults. Create a local `config.toml` in the repository root for machine-specific overrides; it is ignored by git.

Model and plugin artifacts are split between packaged assets under
`Pelagia/assets/` and a local runtime library under `./.pelagia/`. See
[docs/artifacts.md](docs/artifacts.md) for the manifest layout and discovery
rules.

For storage maintenance, migration, and recovery procedures, use:

- [End-User Operations Guide](docs/end-user-support.md)
- [Resetting Pelagia](docs/reset-system.md)
- [Migrating Pelagia](docs/migration.md)
- [Backup And Restore](docs/backup.md)

## Python Environment

Pelagia provides requirements files for common backend environments:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use `requirements-dev.txt` for test/development tools and `requirements-ml.txt`
for optional TensorFlow/Keras U-Net refinement support. On Linux x86_64,
`requirements-ml.txt` prefers TensorFlow's CUDA-enabled pip extra. For Apple
Metal acceleration on macOS, use `requirements-ml-apple-metal.txt`. See
[docs/python-environment.md](docs/python-environment.md) for the full setup
walkthrough.

## Storage Strategy

Pelagia treats storage as a replaceable adapter layer.

- Postgres stores durable metadata, jobs, worker sessions, events, frame records, ROI records, model records, and classification results.
- Large full-frame payloads are stored by a cold payload store adapter because they are expensive and usually read infrequently.
- ROI crops and masks are small enough to store directly with ROI records, making downstream analysis and export simpler.

`KVStore` is currently the cold payload store used by Pelagia during development. It is intentionally separable and is expected to become an external dependency rather than a core Pelagia subsystem.

## Development Direction

The near-term goal is to keep Pelagia small at the interfaces and strong in the pipeline core:

- keep API, CLI, and worker entrypoints thin;
- keep processing routines focused and testable;
- keep row models aligned one-to-one with database records;
- keep storage adapters replaceable;
- keep job history explicit through `job_events`;
- build curation, training, and export features on top of the same ROI-centered data model.
