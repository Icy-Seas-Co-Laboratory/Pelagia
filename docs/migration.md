# Migrating Pelagia

This page covers three common migration tasks:

- upgrading an existing Pelagia checkout;
- moving Pelagia to a new machine;
- moving the KVStore root or project-specific KVStore roots.

Pelagia uses a lightweight SQL migration ledger in PostgreSQL. The packaged base
schema remains idempotent, and migration files under
`Pelagia/storage/sql/migrations/` are applied once and recorded in
`schema_migrations` with checksums. Always back up before upgrading.

`init-system` initializes storage and applies database migrations. If you only
need to migrate PostgreSQL, use `migrate-db`.

## Upgrade An Existing Checkout

Use this for routine code updates on the same machine.

1. Stop API and workers.

```bash
./scripts/pelagia_dev_stack.sh stop
```

2. Back up PostgreSQL, KVStore, `config.toml`, and `.pelagia/`.

See [Backup And Restore](backup.md).

3. Update the checkout and synchronize the Python environments.

```bash
git pull
./scripts/pelagia_env sync cpu
```

For development installs:

```bash
./scripts/pelagia_env sync dev
```

For learned ROI refinement:

```bash
./scripts/pelagia_env sync ml-cuda
```

4. Apply database migrations and verify storage.

```bash
.venv/bin/pelagia migrate-db
.venv/bin/pelagia check-system
```

5. Start the stack.

```bash
./scripts/pelagia_dev_stack.sh start
./scripts/pelagia_dev_stack.sh status
```

6. Verify the API.

```bash
curl http://127.0.0.1:8000/health
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  http://127.0.0.1:8000/system/status/default
```

## Move To A New Machine

Use this when moving a working installation to a different host.

1. Install system prerequisites, PostgreSQL, Python, and Pelagia dependencies on
   the new machine.

2. Stop Pelagia on the old machine.

```bash
./scripts/pelagia_dev_stack.sh stop
```

3. Create a fresh backup on the old machine.

```bash
mkdir -p backups/$(date +%Y%m%d-%H%M%S)
```

Follow the full backup steps in [Backup And Restore](backup.md).

4. Copy these items to the new machine:

- PostgreSQL dump.
- Configured top-level KVStore root.
- Any explicit project KVStore roots outside the top-level root.
- `config.toml`.
- `.pelagia/` model and plugin artifacts.
- Raw import/source directories if you want source paths to remain usable.
- Worker TOML files or service files if you use them.

5. Restore PostgreSQL on the new machine.

Example with a custom-format dump:

```bash
createdb pelagia
pg_restore --dbname=pelagia /path/to/pelagia.dump
```

6. Put the KVStore at the path configured by `[kvstore].root_path`, or update
   `config.toml` to point to the restored path.

7. If projects have explicit `kvstore_root_path` values, either restore those
   directories to the same paths or update the project rows to the new paths.

Example:

```sql
UPDATE pelagia.projects
SET kvstore_root_path = '/new/path/to/project-kvstore'
WHERE project_key = 'field-survey';
```

8. Apply database migrations and verify.

```bash
python -m Pelagia.cli.app migrate-db
python -m Pelagia.cli.app check-system
```

9. Start the API and workers, then verify project status.

```bash
./scripts/pelagia_dev_stack.sh start
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  http://127.0.0.1:8000/system/status/default
```

## Move The Default KVStore Root

Use this when the database stays in place but large payload storage needs a new
disk or mount point.

1. Stop API and workers.

```bash
./scripts/pelagia_dev_stack.sh stop
```

2. Copy the KVStore root while preserving files.

```bash
rsync -a --info=progress2 ./data/kvstore/ /new/storage/pelagia-kvstore/
```

3. Update `config.toml`.

```toml
[kvstore]
root_path = "/new/storage/pelagia-kvstore"

[file_browser]
root_path_kvstore = "/new/storage/pelagia-kvstore"
```

4. Run checks.

```bash
python -m Pelagia.cli.app check-system
```

5. Start the stack and verify frame retrieval in PelagiaView.

```bash
./scripts/pelagia_dev_stack.sh start
```

The content keys stored in PostgreSQL do not include the default KVStore root
path, so moving the root only requires copying the files and updating config.

## Move A Project-Specific KVStore

New non-default projects usually use:

```text
<kvstore.root_path>/projects/<project_id>
```

If the project uses this derived path, moving the top-level KVStore root moves
the project with it.

If `projects.kvstore_root_path` is set for a project, that explicit path wins.
To move it:

1. Stop API and workers.
2. Copy the old project KVStore directory to the new location.
3. Update the project row.
4. Run a project status check.

Example:

```bash
rsync -a --info=progress2 /old/project-kvstore/ /new/project-kvstore/
```

```sql
UPDATE pelagia.projects
SET kvstore_root_path = '/new/project-kvstore'
WHERE project_key = 'field-survey';
```

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/system/status/field-survey?deep_kvstore=true"
```

## Change Database DSN Or Schema

Update `config.toml` or environment variables:

```toml
[database]
dsn = "postgresql://pelagia_user:secret@db.example.org:5432/pelagia"
schema_name = "pelagia"
```

Then initialize and check:

```bash
python -m Pelagia.cli.app init-system
python -m Pelagia.cli.app check-system
```

Changing `schema_name` points Pelagia at a different PostgreSQL schema. It does
not move data from the old schema to the new schema. Use PostgreSQL tools when
you need to rename or copy schemas.

## Downgrades

Downgrades are not automated. If you need to return to an older commit, restore
the database and KVStore backup taken before the upgrade.

## Migration Checklist

- Stop writers before copying storage.
- Back up PostgreSQL and all KVStore roots together.
- Preserve `config.toml` and environment overrides.
- Preserve `.pelagia/` when local models or plugins are used.
- Copy raw source data if source paths need to stay valid.
- Run `migrate-db` or `init-system` after code upgrades.
- Run `check-system` before starting workers.
- Verify at least one project with `/system/status/{project}`.
- Verify image/frame retrieval in PelagiaView.

## Database Migration Ledger

Migration state is stored in:

```text
<schema>.schema_migrations
```

Each row records:

- `migration_id`
- `checksum`
- `description`
- `metadata`
- `applied_at`

Check migration readiness with:

```bash
python -m Pelagia.cli.app check-system
```

The JSON output includes:

```json
{
  "database": {
    "migrations": {
      "available_count": 1,
      "applied_count": 1,
      "pending_count": 0,
      "checksum_mismatches": [],
      "ready": true
    }
  }
}
```

The API reports the same database migration status from:

```text
GET /system/status
GET /system/status/{project}
```

The first packaged migration is `0001_processing_status`, which creates and
indexes the frame processing status projection used by
`/processing/status/*`.
