# Migrating existing cold objects to SkerryKV

Stop Pelagia API processes and workers before migrating, and back up PostgreSQL
and the legacy KVStore directories. Configure and start the cold SkerryKV
service first. Run the copy once per project, using that project's existing
KVStore root and PostgreSQL project UUID:

```sh
export SKERRY_TOKEN='the-service-token'
.venv/bin/python -m Pelagia.services.migrate_cold_storage \
  --dsn "$PELAGIA_DATABASE_DSN" --schema pelagia \
  --project-id PROJECT_UUID --legacy-root /path/to/project/store \
  --legacy-backend kvstore2 --skerry-url http://127.0.0.1:7788
```

Use `--legacy-backend kvstore` for the older store format. `--token-env` selects
another environment variable if needed. The command copies frame, preprocessed
frame, background, stored embedding, and telemetry source objects; it checks the bytes read back
from SkerryKV before updating their database references. Its database checkpoint
allows retrying a failed run against the same endpoint. Keep writers stopped
until every project's copy and the separate ROI and Registry migrations are complete. Do not
switch a checkpointed endpoint to a different underlying store.

After copying cold objects, run `pelagia migrate-roi-storage` to move candidate
and refined ROI images and masks, then `pelagia migrate-registry-assets` to move
Registry workspace assets. Both commands accept `--project-id` to limit their
scope. Verify their reported counts before restarting writers. Once every project
has been copied, run `pelagia migrate-db` to apply subsequent schema migrations.
Migration 0037 refuses to run while either detection table still
contains inline ROI or mask bytes. Migration 0037 then removes those columns
from both detection tables. Upgrade an older database through migration 0035 before
running the ROI copy; do not start the new application until the copy is complete.

Keep the legacy stores and database backup until the migrated application has
been checked. SkerryKV supports object deletion and garbage collection, but Pelagia does not
yet automatically collect unreferenced objects, so failed uploads may leave them. The
cold and hot endpoints can be the same service or separate services. See the
SkerryKV documentation for its backup and recovery procedures.
