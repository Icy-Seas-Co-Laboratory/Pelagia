# Backup And Restore

Pelagia backups should capture PostgreSQL and KVStore from the same maintenance
window. PostgreSQL contains the metadata and content keys. KVStore contains the
large payload bytes those keys refer to.

## What To Back Up

Back up all of the following:

- PostgreSQL database or the Pelagia schema inside that database.
- Configured top-level KVStore root, usually `./data/kvstore`.
- Explicit project KVStore roots from `projects.kvstore_root_path`, if any.
- `config.toml`.
- `.pelagia/` if it contains local model or plugin artifacts.
- Worker stack TOML files or service files.
- Raw import/source directories when source paths need to remain reproducible.

Logs are optional for recovery but useful for support:

```text
logs/pelagia.log
.pelagia/run/
```

## Recommended Schedule

For active systems:

- PostgreSQL: nightly `pg_dump` plus retention.
- KVStore: nightly file-level sync or filesystem snapshot.
- Config and `.pelagia/`: back up whenever changed, plus nightly with the rest
  of the system.
- Before upgrades or resets: take an immediate manual backup.

For large installations, prefer filesystem or volume snapshots for KVStore.
KVStore contains many SQLite shard files, so incremental snapshot tools are
usually faster and safer than repeatedly making compressed archives.

## Consistent Backup Procedure

The safest backup is taken with API and workers stopped.

1. Stop API and workers.

```bash
./scripts/pelagia_dev_stack.sh stop
```

2. Create a timestamped backup directory.

```bash
BACKUP_DIR="$PWD/backups/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"
```

3. Dump PostgreSQL.

If your DSN points at the `pelagia` database:

```bash
pg_dump --format=custom --file "$BACKUP_DIR/pelagia.dump" pelagia
```

If you use a URL DSN:

```bash
pg_dump --format=custom --file "$BACKUP_DIR/pelagia.dump" "$PELAGIA_DATABASE_DSN"
```

4. Copy KVStore.

```bash
rsync -a --delete ./data/kvstore/ "$BACKUP_DIR/kvstore/"
```

If you have explicit project KVStore roots outside `./data/kvstore`, copy each
one too:

```bash
mkdir -p "$BACKUP_DIR/project-kvstores"
rsync -a /path/to/project-kvstore/ "$BACKUP_DIR/project-kvstores/field-survey/"
```

5. Copy configuration and local artifacts.

```bash
cp config.toml "$BACKUP_DIR/config.toml"
rsync -a ./.pelagia/ "$BACKUP_DIR/.pelagia/"
```

6. Record basic backup metadata.

```bash
git rev-parse HEAD > "$BACKUP_DIR/git-commit.txt"
python -m Pelagia.cli.app check-system > "$BACKUP_DIR/check-system.json"
```

7. Start Pelagia again.

```bash
./scripts/pelagia_dev_stack.sh start
```

## Online Backups

If you cannot stop the system, take the PostgreSQL dump first, then copy
KVStore as soon as possible. This may still miss payloads created after the
database dump or include unreferenced payloads created before the KVStore copy
finished.

For high-value systems, use database and filesystem snapshots from the same
storage snapshot point. If that is not available, schedule a short maintenance
window.

## Restore Procedure

1. Stop API and workers.

```bash
./scripts/pelagia_dev_stack.sh stop
```

2. Restore PostgreSQL.

For a fresh local restore:

```bash
dropdb --if-exists pelagia
createdb pelagia
pg_restore --dbname=pelagia /path/to/backup/pelagia.dump
```

3. Restore KVStore to the configured root.

```bash
rm -rf ./data/kvstore
mkdir -p ./data
rsync -a /path/to/backup/kvstore/ ./data/kvstore/
```

4. Restore `config.toml` and `.pelagia/` if needed.

```bash
cp /path/to/backup/config.toml ./config.toml
rsync -a /path/to/backup/.pelagia/ ./.pelagia/
```

5. Restore explicit project KVStores if your projects use them.

```bash
rsync -a /path/to/backup/project-kvstores/field-survey/ /path/to/project-kvstore/
```

If restored paths differ from the old machine, update
`projects.kvstore_root_path` in PostgreSQL before starting workers.

6. Apply current schema compatibility updates.

```bash
python -m Pelagia.cli.app init-system
python -m Pelagia.cli.app check-system
```

7. Start Pelagia.

```bash
./scripts/pelagia_dev_stack.sh start
```

8. Verify API and project status.

```bash
curl http://127.0.0.1:8000/health
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/system/status/default?deep_kvstore=true"
```

9. Open PelagiaView and verify that at least one known frame or ROI image loads.

## Backup Validation

A backup has not really succeeded until it has been restored somewhere.
Recommended validation:

- Restore to a temporary database name or separate machine.
- Point a temporary `config.toml` at a copied KVStore.
- Run `python -m Pelagia.cli.app init-system`.
- Run `python -m Pelagia.cli.app check-system`.
- Run `/system/status/{project}?deep_kvstore=true` for at least one project.
- Load a known original frame, preprocessed frame, candidate ROI, and refined
  ROI through the API or PelagiaView.

## Retention

A practical default retention policy:

- Keep daily backups for 14 days.
- Keep weekly backups for 8 weeks.
- Keep monthly backups for 12 months.
- Keep a pre-upgrade backup for every significant Pelagia upgrade until the
  next upgrade has been validated.

Adjust retention to match available storage and the cost of rerunning imports
or analysis.

## Notes On Raw Source Data

Pelagia stores frame payloads in KVStore after ingestion, but raw source files
can still matter for reproducibility and future reprocessing. If the original
video or image paths are on removable drives, network shares, or temporary
import folders, back those up separately.
