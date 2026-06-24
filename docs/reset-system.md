# Resetting Pelagia

This page explains how to reset Pelagia storage. Most reset operations are
destructive, so make a backup first unless the installation is disposable.

See [Backup And Restore](backup.md) before resetting a system that contains
data you may need later.

## What A Reset Touches

Pelagia has two linked storage layers:

- PostgreSQL rows: users, projects, sessions, runs, assets, frames, detections,
  jobs, logs, models, and memberships.
- KVStore payloads: full-frame image bytes, preprocessed frame bytes, generated
  background payloads, and related large binary data.

The built-in CLI reset clears PostgreSQL rows and reinitializes the configured
top-level KVStore. It preserves the database schema and recreates the default
project.

Important current limitation: project-specific KVStores under
`<kvstore.root_path>/projects/<project_id>` or explicit
`projects.kvstore_root_path` locations are not automatically deleted by the CLI
reset. Remove those directories manually when you want a full physical reset.

## Development Reset

Use this when you want a clean local development system and do not need to keep
existing data.

```bash
./scripts/pelagia_dev_stack.sh stop

python -m Pelagia.cli.app reset --delete
python -m Pelagia.cli.app create-dev-login \
  --username dev-admin \
  --password pelagia-dev

./scripts/pelagia_dev_stack.sh start
python -m Pelagia.cli.app check-system
```

If you use the TOML stack, stop and start that stack instead:

```bash
./scripts/pelagia_stack_from_toml.sh stop scripts/pelagia_workers.toml
python -m Pelagia.cli.app reset --delete
./scripts/pelagia_stack_from_toml.sh start scripts/pelagia_workers.toml
```

## Complete Physical Reset

Use this when you need to delete all local data, including orphaned project
KVStore directories.

1. Stop the API and workers.

```bash
./scripts/pelagia_dev_stack.sh stop
```

2. Confirm the configured database and KVStore root.

```bash
python -m Pelagia.cli.app check-system
```

3. Reset the database rows and top-level KVStore.

```bash
python -m Pelagia.cli.app reset --delete
```

4. Remove project KVStore directories that are no longer referenced.

For the default local layout:

```bash
rm -rf ./data/kvstore/projects
```

If projects used explicit KVStore roots, remove those directories too. Only
delete directories that you have confirmed are Pelagia KVStores.

5. Recreate the default login or your real users and projects.

```bash
python -m Pelagia.cli.app create-dev-login \
  --username dev-admin \
  --password pelagia-dev
```

6. Restart and verify.

```bash
./scripts/pelagia_dev_stack.sh start
python -m Pelagia.cli.app check-system
```

## Reset Only PostgreSQL Rows

The CLI does not currently expose a database-only reset command. If you clear
PostgreSQL without clearing KVStore, the KVStore will contain orphaned payloads.
That is usually safe but wastes disk space.

For development systems, prefer:

```bash
python -m Pelagia.cli.app reset --delete
```

For production-like systems, restore from a known backup or use PostgreSQL
administration tools only after taking a fresh backup.

## Reset Only KVStore

Do not reset KVStore while keeping PostgreSQL rows unless you intentionally want
frame and image retrieval to fail. PostgreSQL stores the content keys that point
to KVStore payloads. If the payloads are deleted, those database references
become broken.

The only generally safe KVStore-only reset is for an empty database or a fresh
installation.

```bash
python -m Pelagia.cli.app init-kvstore
```

If the KVStore path already contains data, use the full reset procedure instead.

## Recreate The Schema Without Deleting Data

`init-system` is safe to run repeatedly. It applies the current schema template,
creates missing tables, adds missing columns where supported, and initializes
the configured KVStore if needed.

```bash
python -m Pelagia.cli.app init-system
python -m Pelagia.cli.app check-system
```

This is the first repair step when a new checkout expects newer tables or
columns.

## Common Reset Failures

`Refusing to reset Pelagia storage without --delete`

The CLI requires an explicit destructive flag:

```bash
python -m Pelagia.cli.app reset --delete
```

`KVStore is not initialized`

Run:

```bash
python -m Pelagia.cli.app init-system
```

`Project status shows missing KVStore data after reset`

The database may reference a project KVStore path that was deleted, or the
project KVStore may still exist but was not copied back after a restore. Check
`/system/status/{project}` and the project's `kvstore_root_path`.

## Safety Checklist

Before any non-disposable reset:

- Stop API and worker processes.
- Run `pg_dump`.
- Copy the configured KVStore root.
- Copy any explicit project KVStore roots.
- Copy `config.toml`.
- Copy `.pelagia/` if it contains local models or plugin manifests.
- Confirm where raw source files live if you need to reproduce ingestion.
