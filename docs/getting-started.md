# Getting started

[Documentation index](index.md)

## Quick Start

Pelagia is split into a Python backend (`Pelagia`) and an optional SvelteKit
frontend (`PelagiaView`). The backend owns storage, processing, the job queue,
workers, and the HTTP API. PelagiaView connects to that API from a browser. Power users may also be interested in leveraging the API directly via scripts (e.g., python, R, Julia) or through the command line interface (CLI) either locally or through SSH. 

For operational procedures after installation, see the
[End-User Operations Guide](end-user-support.md). It links the reset,
migration, KVStore migration, backup, and restore procedures that keep the
database and blob storage in sync.

For large-frame filtering and project processing summaries, see the
[Processing Status Projection](processing-status.md) guide.

### Registry workspaces

PelagiaView includes Registry at `/registry/`. Opening an Oracle Dataset SQLite
file queues a progress-reporting worker job that loads a validated, user- and project-scoped working copy into PostgreSQL;
the source file is not modified. Registry edits are recorded in PostgreSQL and
can be exported as a new child SQLite revision, explicitly written back to the
current source (with a backup and hash guard), or purged from PostgreSQL. The
portable schema remains owned by the existing `oracle_data_contracts` package.
In the Registry gallery, open an ROI with its magnifying-glass button to inspect
the full-resolution image, pixel scale, and measurements. Edit mode lets you
adjust the object bounding box inside the stored crop. For mask-refinement
items, the candidate and current masks can be viewed as separate layers; saved
eraser strokes create a new mask annotation while retaining the candidate mask
and earlier annotations. Identical mask bytes reuse an existing asset. Image
inversion changes only the display.
Run at least one worker with `registry_load,registry_export,registry_generate` capabilities.

### Prerequisites

- [uv](https://docs.astral.sh/uv/) for Python and dependency management.
- PostgreSQL reachable from the Pelagia machine.
- A SkerryKV service for object storage and a writable runtime directory for logs and pid files.
- Node.js 18+ and npm if you will run PelagiaView.

On Debian/Ubuntu systems, these packages are commonly useful before creating the
Python environment:

```bash
sudo apt update
sudo apt install -y curl ca-certificates postgresql postgresql-client libgl1 libglib2.0-0
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Debian 13 ships Python 3.13 rather than Python 3.12. This is supported: uv
downloads Pelagia's pinned Python 3.12 runtime independently and does not modify
the system Python.

### Install The Backend

From the Pelagia repository root, create the API/CPU-worker environment:

```bash
./scripts/pelagia_env sync cpu
```

This uses `uv.lock`, installs a uv-managed Python 3.12 when necessary, and
synchronizes `.venv` exactly. Activation is optional; Pelagia's stack scripts
select the managed environment automatically.

Run the CLI examples in this guide from the repository root. They use
`.venv/bin/pelagia` explicitly so an unrelated system, Conda, or user Python
cannot accidentally run Pelagia without its runtime extras.

For development and tests, install the dev target instead:

```bash
./scripts/pelagia_env sync dev
```

Oracle Builder owns TensorFlow/Keras and GPU isolation. Pelagia only verifies
its CPU environment:

```bash
./scripts/pelagia_env doctor all
./scripts/pelagia_env doctor cpu --require-jpegxs
```

`pelagia env sync ...` and `pelagia env doctor ...` provide the same commands
after the CPU environment is installed. Pass `--imagecodecs-wheel PATH` to
`sync` when installing the internally built JPEG XS-enabled wheel.

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

[cold_storage]
endpoint = "http://127.0.0.1:7788"
timeout_seconds = 30

[hot_storage]
endpoint = "http://127.0.0.1:7788"
timeout_seconds = 30

[api]
host = "0.0.0.0"
port = 8000
root_path = ""
cors_allow_origin_regex = "https?://(localhost|127\\.0\\.0\\.1|10(?:\\.\\d{1,3}){3}|100\\.(?:6[4-9]|[7-9]\\d|1[01]\\d|12[0-7])(?:\\.\\d{1,3}){2}|192\\.168(?:\\.\\d{1,3}){2}|172\\.(?:1[6-9]|2\\d|3[01])(?:\\.\\d{1,3}){2})(?::\\d+)?"

[auth]
enabled = true
session_ttl_seconds = 604_800
dev_project_key = "default"
```

`config.toml` is ignored by git. You can also use environment variables such as
`PELAGIA_DATABASE_DSN`, `PELAGIA_DATABASE_SCHEMA`,
`PELAGIA_COLD_STORAGE_ENDPOINT`, `PELAGIA_HOT_STORAGE_ENDPOINT`,
`PELAGIA_COLD_STORAGE_TOKEN`, `PELAGIA_HOT_STORAGE_TOKEN`,
`PELAGIA_API_HOST`, `PELAGIA_API_PORT`, `PELAGIA_API_ROOT_PATH`, `PELAGIA_AUTH_ENABLED`,
`PELAGIA_AUTH_SESSION_TTL_SECONDS`, and `PELAGIA_AUTH_DEV_PROJECT_KEY`.

### Docker deployment

For a single-host deployment with PelagiaView, PostgreSQL, CPU workers, and
Oracle Builder's existing Docker stack, see the
[Docker deployment guide](../deploy/docker/README.md).

### Initialize And Run The Backend

Initialize the database schema and bootstrap the admin user:

```bash
.venv/bin/pelagia init-system
.venv/bin/pelagia create-dev-login --username dev-admin --password pelagia-dev
```

For code upgrades where storage already exists, apply database migrations with:

```bash
.venv/bin/pelagia migrate-db
.venv/bin/pelagia check-system
```

With the default projectless configuration, `create-dev-login` creates or
reuses the admin user without creating a project or KV store. The first login
returns an authenticated projectless session with `project: null` and
`project_creation_required: true`. Use that token with `POST /projects`, then
`POST /auth/switch-project` to obtain a project-scoped session. Supplying
`--project-key` preserves the one-command development flow by creating that
project, adding membership, and returning a session token. For manual setup use:

```bash
.venv/bin/pelagia create-user ada --password secret --admin
.venv/bin/pelagia create-project field-survey --project-name "Field Survey"
.venv/bin/pelagia add-project-user ada field-survey --role editor
.venv/bin/pelagia list-projects --username ada
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

PostgreSQL job events and structured logs have separate age-based retention;
see [operational log retention](operational-logs.md).

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

The TOML stack chooses its Python interpreter from `[worker_profiles]`. All
Pelagia stages use the CPU environment; ROI refinement calls Oracle Builder.

```toml
[worker_profiles]
default = "cpu"
roi_refinement = "cpu"
```

Validate the resolved worker profiles before starting services:

```bash
./scripts/pelagia_stack_from_toml.sh validate scripts/pelagia_workers.toml
```

See [Oracle Builder inference](oracle-builder.md) for service setup and
failure behavior.

See [ROI curation and ML evidence](curation.md) for the
human-review workflow, storage boundaries, and classification worker setup.
That workflow also accepts Oracle Builder self-supervised clustering models and
retains their run-local feature-space evidence separately from project labels.

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

1. `POST /auth/login` with `username` and `password`; include `project_key` or
   `project_id` when selecting an existing project.
2. Store the returned `token` for the browser session.
3. Send `Authorization: Bearer <token>` on API calls.
4. Use `GET /auth/me` to restore the active user/project after refresh.
5. Use `GET /projects` and `POST /auth/switch-project` to move between projects.
   `GET /projects?include_all_names=true` also includes `all_project_names`
   for global project-name pickers while leaving `projects` scoped to the user.
6. When no project exists, a user admin receives a projectless token with
   `project: null` and `project_creation_required: true`. Use it with
   `POST /projects`, then call `POST /auth/switch-project`.
7. User admins can create projects with `POST /projects`. User admins and
   project managers/admins can soft-delete non-default projects with
   `DELETE /projects/{project_id_or_key}`.
8. Logged-in users can list users in their active project with `GET /users`.
   User admins can pass `include_all_projects=true` for a global user list.
9. User admins and project managers/admins can manage user accounts with
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

