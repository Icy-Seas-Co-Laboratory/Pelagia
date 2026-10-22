# Queue status snapshots

Queue displays use `GET /jobs/status` as their authoritative read model.
`QueueStatusService` defines progress and estimate semantics, and PostgreSQL
supplies aggregates and bounded job rows from one read-only repeatable-read
transaction. Pagination of job rows does not change the aggregate counts.

## Scope and response

Authentication selects the project. The endpoint accepts `run_id`, `asset_id`,
`status`, `stage`, `ids`, and `worker_id`, using the same filter semantics as
`/jobs/summary`. Statuses, stages, and IDs accept repeated or comma-separated
values. At most 250 job IDs may be requested.

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/jobs/status?stage=segment&job_limit=50"
```

The response includes:

- `schema_version`: the status contract version, currently 1.
- `snapshot_id` and `as_of`: the identity and observation time of this response.
  Snapshot IDs are opaque identities, not ordered event cursors.
- `scope`: the authenticated project and applied filters.
- `total` and `by_stage`: complete job lifecycle counts for that scope.
- `work_groups`: progress and estimates grouped by stage and work unit.
- `health`: pending controls, expired leases, stale progress observations,
  active-job progress coverage, and the age of the oldest queued job.
- `jobs`: bounded normalized job rows, with `limit`, `matching_count`, and
  `truncated`. `job_limit` defaults to 50 and is bounded to 1–250.
- `recent_failures`: bounded failed/dead-lettered job rows within the same scope.
  `recent_failure_limit` defaults to 10 and is bounded to 0–100.

Active work includes queued, leased, working, and paused jobs. Historical jobs
are succeeded, failed, cancelled, or dead-lettered. A row limit only limits
display details; clients must use server aggregates for totals.
Stage monitors use `active_progress`; submitted-batch monitors use `progress`
for their explicit job-ID scope, including completed jobs. `historical_progress`
is available separately. These scopes are labelled in the UI.

## Progress and estimates

Units are explicit: frames, ROIs, bytes, phases, and estimated export work units
are different quantities. Clients must not sum them or average their
percentages. Stored frame coverage remains available through
[`/processing/status`](processing-status.md); it is not the percentage of the
current queue that has finished.

Progress observations identify the worker attempt, unit, phase, known/unknown
total, completed/failed/skipped counts, and observation time. Reporter schema 2
also records a recent sample window and elapsed time. Existing schema 1 job
records remain readable.

The worker reporter uses a recent window of approximately 30 seconds, retaining
one earlier observation when updates are sparse. A new reporter, counter reset,
unit change, or recovery of already committed outputs resets the rate baseline.
This prevents resumed work from appearing as newly processed throughput.

Rate and ETA objects contain an `available` flag and a machine-readable `reason`
when an estimate cannot be made. Unknown totals and insufficient progress must
remain unknown in the UI. ETA is an estimate for the stated scope and unit; it
does not predict unplanned downstream pipeline work. Completing a processing
loop is distinct from committing the job: `finalizing` progress remains active
until the persisted job lifecycle becomes terminal.
Progress observations become stale after 120 seconds independently of worker
lease heartbeats. Stale, expired-lease, and finalizing work cannot provide a
current throughput or ETA estimate.

## Client lifecycle

PelagiaView uses a shared, scope-aware polling store. Components subscribe to a
snapshot and format its fields; they do not derive rates, ETA, queue totals, or
progress percentages from job lists. Related widgets replace their snapshot
together. Refresh failures preserve the last successful snapshot and mark it
stale. Project/auth changes invalidate the previous subscription generation so
an obsolete request cannot populate the new view.
Subscribers with the same scope share one request and snapshot. Polling runs
every 2.5 seconds for active work, 12 seconds when idle, and 30 seconds in a
hidden tab. The last unsubscribe stops polling.

Queue mutations trigger a refresh. Job logs, event history, artifact metadata,
and frame coverage can still use their dedicated APIs; they are separate data
from the queue status snapshot.

[Durable processing submissions](processing-submissions.md) report acceptance
and planning before jobs exist. Once planning completes, their created jobs
appear in queue snapshots. A submission's successful planning status does not
mean its jobs have finished processing.

`/jobs`, `/jobs/summary`, and `/jobs/stream` remain available for existing clients.
The current status implementation uses polling. SSE can later invalidate these
same snapshots without moving metric calculations back into the browser.
