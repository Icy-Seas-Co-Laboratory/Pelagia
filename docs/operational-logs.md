# Operational log retention

Pelagia rotates local text logs with `logging.max_bytes` and
`logging.backup_count`. PostgreSQL `logs` and `job_events` use age-based
retention. Every long-running worker attempts cleanup on startup and every
`logging.cleanup_interval_seconds` thereafter. A transaction advisory lock
allows only one worker to clean a schema at a time. Each pass deletes at most
`logging.cleanup_batch_size` rows from each category, so an old backlog may
need several passes. PostgreSQL autovacuum reuses freed space; the database
file may not shrink immediately.

| Category | Default age |
| --- | ---: |
| Job progress and legacy heartbeat/touch events | 7 days |
| Other job events | 90 days |
| Legacy high-frequency structured logs | 7 days |
| Routine structured logs | 90 days |
| Warning/error/critical logs and `jobs.deleted` | 365 days |

The live `processing_jobs.progress` and `worker_sessions` rows retain current
state. Heartbeats and worker touches no longer create history events; progress
events remain in `job_events` for live consumers but are not copied into
`logs`. Other lifecycle events still appear in both tables. Scientific source
and derived data are outside this cleanup.

Run a read-only count of expired rows before changing retention or running a
manual pass:

```bash
pelagia cleanup-logs --dry-run
pelagia cleanup-logs --no-dry-run
```

The manual command deletes one bounded batch. Set the retention periods and
batch interval under `[logging]` in `config.toml`, or use
`PELAGIA_LOG_HIGH_FREQUENCY_RETENTION_DAYS`,
`PELAGIA_LOG_JOB_EVENT_RETENTION_DAYS`,
`PELAGIA_LOG_ROUTINE_RETENTION_DAYS`,
`PELAGIA_LOG_IMPORTANT_RETENTION_DAYS`,
`PELAGIA_LOG_CLEANUP_INTERVAL_SECONDS`, and
`PELAGIA_LOG_CLEANUP_BATCH_SIZE`.
Keep the important-log period aligned with the project's operational and
record-retention requirements before shortening it.
