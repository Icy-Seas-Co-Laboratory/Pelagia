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

The built-in CLI reset discovers every project KVStore, clears and reinitializes
each available store, then clears PostgreSQL rows. It preserves the database
schema but does not create a default project or KVStore. A system with no
projects or stores is valid and resets PostgreSQL normally.

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

3. Reset the database rows and all configured project KVStores.

```bash
python -m Pelagia.cli.app reset --delete
```

4. Review the `kvstores.results` entries in the command output. Stores marked
`reset` were cleared; `missing` means the configured path did not contain an
initialized store. Remove the now-empty store directories manually only when
you also want their configuration files removed.

5. Recreate the admin login. The first UI login can create the first project.

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

## Reset A Projectless System

The same command works when no projects or KVStores exist. In that case it
resets only PostgreSQL and reports a KVStore count of zero:

```bash
python -m Pelagia.cli.app reset --delete
```

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
creates missing tables, and adds missing columns where supported. Project
KVStores are created when projects are created.

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

Create or select a project with a configured KVStore. Projectless system reset
does not require a KVStore.

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
