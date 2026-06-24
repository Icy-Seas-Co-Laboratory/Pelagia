# End-User Operations Guide

This guide is the operational entry point for running, maintaining, resetting,
migrating, and backing up a Pelagia installation.

Pelagia stores state in several places:

- PostgreSQL stores users, projects, sessions, metadata, assets, frames,
  detections, jobs, logs, and processing history.
- KVStore stores large binary frame payloads and generated frame payloads.
- `config.toml` and environment variables define where PostgreSQL and KVStore
  live.
- `.pelagia/` stores local model and plugin artifacts by default.
- Raw source files may live outside Pelagia and are usually referenced by path
  during ingestion.

Because the database and KVStore reference each other, treat them as one system
when resetting, migrating, or backing up.

## Related Guides

- [Resetting Pelagia](reset-system.md): clear a development system, rebuild
  storage, or perform a complete destructive reset.
- [Migrating Pelagia](migration.md): move Pelagia to a new machine, upgrade a
  checkout, or move the PostgreSQL database and KVStore.
- [Backup And Restore](backup.md): recommended backup schedule, commands, and
  restore checks.
- [Python Environment Setup](python-environment.md): Python, dependency, and
  local configuration setup.
- [Artifact Organization](artifacts.md): model and plugin artifact layout.

## Daily Health Checks

Before large imports or processing runs, verify that the API, database, and
KVStore are reachable.

```bash
python -m Pelagia.cli.app check-system
curl http://127.0.0.1:8000/health
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  http://127.0.0.1:8000/system/status/default
```

For deeper KVStore validation, use:

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/system/status/default?deep_kvstore=true"
```

Deep KVStore checks can touch many SQLite shard files and should be used
sparingly on large installations.

## Configuration Locations

Pelagia loads configuration in this order:

```text
Pelagia/default.config.toml < ./config.toml < environment variables < CLI flags
```

The local `config.toml` file is ignored by git and should be backed up with the
installation. The most important storage fields are:

```toml
[database]
dsn = "postgresql://postgres:postgres@127.0.0.1:5432/pelagia"
schema_name = "pelagia"

[kvstore]
root_path = "./data/kvstore"

[file_browser]
root_path_kvstore = "./data/kvstore"
root_path_import_dir = "./data/import"
allowed_root_paths = []

[auth]
enabled = true
session_ttl_seconds = 604800
dev_project_key = "default"
```

Common environment overrides include:

```text
PELAGIA_DATABASE_DSN
PELAGIA_DATABASE_SCHEMA
PELAGIA_KVSTORE_ROOT
PELAGIA_FILE_BROWSER_ROOT_PATH_KVSTORE
PELAGIA_FILE_BROWSER_ROOT_PATH_IMPORT_DIR
PELAGIA_FILE_BROWSER_ALLOWED_ROOT_PATHS
PELAGIA_AUTH_ENABLED
PELAGIA_AUTH_SESSION_TTL_SECONDS
PELAGIA_AUTH_DEV_PROJECT_KEY
```

If you start Pelagia with `scripts/pelagia_stack_from_toml.sh`, the stack TOML
can also include a `[file_browser]` section. The launcher translates those
values into the `PELAGIA_FILE_BROWSER_*` environment variables used by the API:

```toml
[file_browser]
root_path_kvstore = "/storage/kvstore"
root_path_import_dir = "/storage"
allowed_root_paths = ["/scratch", "/storage"]
```

## Storage Layout

The default local development layout is:

```text
data/
  kvstore/
    config.json
    manifest.json
    <hash-prefix>/
      store_000001.sqlite
    projects/
      <project-id>/
        config.json
        manifest.json
        <hash-prefix>/
          store_000001.sqlite
  import/
.pelagia/
  models/
  plugins/
logs/
  pelagia.log
```

The default project can continue to use the top-level `data/kvstore` root. New
projects normally use derived KVStore roots under:

```text
<kvstore.root_path>/projects/<project_id>
```

Projects may also have an explicit `projects.kvstore_root_path` in PostgreSQL.
When that field is set, Pelagia uses that physical path for the project instead
of the derived path.

## Safe Operating Pattern

For maintenance that touches storage, use this pattern:

1. Stop PelagiaView if users are active.
2. Stop API and worker processes.
3. Confirm the active configuration with `python -m Pelagia.cli.app check-system`.
4. Back up PostgreSQL, KVStore, `config.toml`, and `.pelagia/`.
5. Perform the reset, migration, or restore.
6. Run `python -m Pelagia.cli.app init-system`.
7. Run `python -m Pelagia.cli.app check-system`.
8. Start the API and workers.
9. Confirm project-specific status with `/system/status/{project}`.

The development stack can be stopped and started with:

```bash
./scripts/pelagia_dev_stack.sh stop
./scripts/pelagia_dev_stack.sh start
./scripts/pelagia_dev_stack.sh status
```

For the TOML-driven worker stack:

```bash
./scripts/pelagia_stack_from_toml.sh stop scripts/pelagia_workers.toml
./scripts/pelagia_stack_from_toml.sh start scripts/pelagia_workers.toml
./scripts/pelagia_stack_from_toml.sh status scripts/pelagia_workers.toml
```

## Accounts And Projects

Initialize a local admin session with:

```bash
python -m Pelagia.cli.app init-system
python -m Pelagia.cli.app create-dev-login \
  --username dev-admin \
  --password pelagia-dev
```

Manual account and project setup:

```bash
python -m Pelagia.cli.app create-user ada --password secret --admin
python -m Pelagia.cli.app create-project field-survey --project-name "Field Survey"
python -m Pelagia.cli.app add-project-user ada field-survey --role editor
python -m Pelagia.cli.app list-projects --username ada
```

PelagiaView should log in with `POST /auth/login`, store the returned bearer
token, and send `Authorization: Bearer <token>` with API requests. A session is
tied to both a user and an active project, so API resource access is naturally
scoped to that project.

## When To Ask For Help

Get help before proceeding if:

- `check-system` reports missing database tables after `init-system`.
- `/system/status/{project}` reports an uninitialized KVStore for a project that
  already has frame data.
- A backup contains PostgreSQL data but no matching KVStore copy.
- You need to downgrade to an older Pelagia commit.
- You changed `projects.kvstore_root_path` manually and frame retrieval starts
  returning missing-key errors.
