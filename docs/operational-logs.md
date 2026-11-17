# Operational output and retention

Pelagia stores diagnostic output and job-event history as UTF-8 JSON Lines on
the shared local disk. PostgreSQL retains the existing job row: each status or
progress update replaces its current values. Leases, worker heartbeats, pause
and cancellation controls, results, and scientific provenance remain database
state. Routine heartbeats and worker touches do not produce history events.

New output does not insert rows into PostgreSQL `logs` or `job_events`. The
tables and their existing contents remain intact for compatibility and manual
archival; the output endpoints now read files, not those historical tables.
Scientific interchange `history.jsonl` is independent of operational output and
is not rotated or expired by this system.

## Files and deployment

All API and worker processes in an installation must use the same absolute
`logging.log_path` on a local filesystem. `logging.file_name` supplies the
journal name prefix; the default is `pelagia.jsonl`. Existing configurations
using `pelagia.log` continue to work as a prefix, but new files contain JSONL.
Use separate directories for separate installations/database schemas. This is
a single-machine design; do not place its lock files on NFS or use independent
directories on multiple machines and expect a shared event stream.

Every record includes a schema version, globally increasing integer `id`, event
UUID, UTC timestamp, stream, event type, and available project/job/run/asset and
producer context. The shared journal uses interprocess file locks so API and
worker appends and rotation cannot interleave. Lifecycle history is stored
once and exposed through both event and log queries; progress history is exposed
through event queries only. Core process logs may have no project and are only
available through administrator file access.

Active files end in `.jsonl`. Files are sealed on size/time rotation and later
compressed to `.jsonl.zst` with zstd checksums. API and worker maintenance passes
run at startup and every `logging.cleanup_interval_seconds`; the manual command
below also performs maintenance without requiring a PostgreSQL connection.
Compression uses a temporary file and preserves its source until the compressed
file has been finalized. Interrupted appends and compression can be recovered
on subsequent writes/maintenance passes.

`max_bytes` bounds a segment between records; an individual record can exceed
that setting. Output is diagnostic history, not a source of job truth. Database
transaction events are published only after successful commit; rollbacks do not
publish them. A crash after commit can still lose a message. Output failures
fall back to structured stderr and must not undo a committed job operation.

## Retention and maintenance

| Category | Default age |
| --- | ---: |
| Job progress and legacy heartbeat/touch output | 7 days |
| Other job events | 90 days |
| Routine structured logs | 90 days |
| Warning/error/critical logs and `jobs.deleted` | 365 days |

Retention operates on sealed files. `backup_count` and `max_total_bytes` provide
additional disk safety caps, so age settings are retention limits rather than
guarantees that all history will remain available for that duration. Active
files are protected from deletion. Maintenance is coordinated between processes;
scientific source and derived data are outside this cleanup.

```bash
pelagia cleanup-logs --dry-run
pelagia cleanup-logs --no-dry-run
```

The dry run reports eligible file removals without modifying output. A real pass
compresses sealed files and expires eligible segments. The command reports file
counts, not database-row counts. Existing `database_dsn`/`schema` CLI options are
accepted for command compatibility but do not select the output location.

Settings live under `[logging]`:

```toml
log_path = "./logs"
file_name = "pelagia.jsonl"
level = "INFO"
console = true
max_bytes = 10485760
backup_count = 10
compression_level = 3
rotation_interval_seconds = 86400
max_total_bytes = 1073741824
high_frequency_retention_days = 7
job_event_retention_days = 90
routine_log_retention_days = 90
important_log_retention_days = 365
cleanup_interval_seconds = 300
cleanup_batch_size = 50000
```

The new settings have `PELAGIA_LOG_COMPRESSION_LEVEL`,
`PELAGIA_LOG_ROTATION_INTERVAL_SECONDS`, and `PELAGIA_LOG_MAX_TOTAL_BYTES`
environment overrides. Existing `PELAGIA_LOG_*` path, size, count, and retention
overrides continue to apply. `cleanup_batch_size` remains accepted for
configuration compatibility; maintenance works on whole files rather than SQL
row batches.

## Reading output

`GET /logs` retains project-scoped filters and numeric cursors. History pages
are limited to 1,000 records and an offset of 10,000; use cursors for longer
traversals. New HTTP log records require project write access, and any supplied
run/asset/job IDs must belong to the selected project.
`GET /jobs/events` lists event history newest first. `GET /jobs/stream` replays
the earliest page after the supplied cursor and closes, allowing reconnects with
`Last-Event-ID` without skipping a backlog. IDs remain JavaScript-safe integers
and do not reset during normal rotation or retention. Old database cursors are
accepted, but the pre-migration SQL history is not replayed. Clients should read
current job snapshots after reconnecting because diagnostic history can expire
or be lost after a process failure.

Recent-cursor reads skip sealed segments outside their ID range and reuse
bounded offsets into active files. Other historical filters still scan the
candidate files; the journal is not a replacement for an indexed analytics
database.

Raw file listing and reading (`/logs/files`) require administrator access because
a file may contain output from several projects. The API supports legacy `.log`,
new `.jsonl`, and `.jsonl.zst` files, rejects escaping paths and symlinks, and
limits returned tails. Compressed reads also have a 64 MiB decompressed scan
budget; inspect larger archives offline. For example:

```bash
zstd -dc logs/<sealed-segment>.jsonl.zst | tail -n 100
```

Retain the entire output directory, including hidden sequence metadata, if
backing up or moving logs. Stop producers before moving or deleting active files
or lock/sequence files. Back up PostgreSQL separately for authoritative state.
