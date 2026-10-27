# Durable processing submissions

Use `POST /processing/submissions` to submit a processing batch or series without
waiting for the server to enumerate frames and create jobs. The request contains
selection criteria and processing options. A `202 Accepted` response means the
submission has been saved; it does not mean matching or processing has completed.

```json
{
  "request_id": "dcefe242-7560-4ce9-88c7-99755186069c",
  "kind": "queue",
  "request": {
    "stage": "preprocess_frames",
    "filters": {
      "collection": ["tow-01"],
      "preprocessing_status": ["unknown", "failed"]
    },
    "options": {"apply_mask": true}
  }
}
```

Authentication determines the project. Filters retain the processing queue's
asset, collection/tag, frame-range, and processing-state semantics. Explicit
frame IDs remain available for deliberately selected frames; callers do not
need to enumerate frames to submit a filtered dataset. A `series` submission
uses the existing processing-series request shape in `request`.

Keep the same `request_id` and request body when retrying an uncertain network
response. Within a project, an identical retry returns the original submission;
reusing that ID with different criteria or options returns a conflict. A new
request ID with identical active criteria and settings also returns the active
submission and sets `reused_active_submission: true`.

## Monitoring and selection timing

`GET /processing/submissions` lists recent project submissions, and
`GET /processing/submissions/{id}` retrieves one submission. The lifecycle is
`accepted`, `planning`, `processing`, then `succeeded`, `failed`, or `cancelled`.
Planning and processing overlap: each bounded selection page, worker job, frame
state, reservation, and seek cursor commit together. Workers can claim the
first job while later pages are still being prepared. `succeeded` means both
planning and all linked worker jobs completed successfully.

The additive `progress` object reports `phase`, `scheduled_count`, `job_count`,
live queued/running/paused/completed/failed/cancelled job counts, and
`more_jobs_being_prepared`. `phase: waiting` includes
`waiting_for_submission_id` when an older project request owns the current
planner pass. Counts grow as pages commit; planning does not run a full count
before the first job. A successful submission can match no eligible work.

Criteria and options are saved at acceptance. A queue submission freezes a
database-time creation boundary on its first planning pass, then uses durable
seek pagination. Units created after that boundary do not enter the request;
collection membership and eligibility of existing units are rechecked as each
page is reserved. A series
captures a separate boundary when each step becomes eligible, allowing later
steps to consume artifacts created by earlier steps without admitting newer,
unrelated inputs.
The created jobs retain their resolved frame or detection IDs for provenance.
For preprocessing and segmentation, omitted eligibility states default to
`unknown`, `failed`, and `cancelled` for the respective stage. Refinement defaults to
unrefined candidates. Preview frame coverage and the count of eligible ROI
candidates are different quantities, especially for partially refined frames.
Each series stage resolves its eligibility when that stage becomes ready and
cannot advance until planning is complete and every attached job is terminal.

## Job batch limits

Preprocessing and segmentation jobs contain at most 1,000 frames or an estimated
1,000,000,000 decoded input bytes, whichever limit is reached first. ROI
refinement jobs contain at most 10,000 candidate detections or an estimated
100,000,000 decoded input bytes. The same limits apply to durable submissions
and the synchronous queue endpoints. Byte estimates use the selected payload's
stored shape and dtype, not compressed blob size. Refinement counts the decoded
candidate crop and mask; optional source frames loaded for expansion are not
counted in the batch budget. Candidates without a stored ROI crop are excluded from refinement; they remain available as candidate records. Existing queued jobs skip such candidates when processed by an updated worker.
Inputs with missing or invalid size metadata, and individual inputs larger than
the cap, are scheduled alone so work is not lost. These are input-work budgets;
intermediate arrays and model allocations can make peak worker memory higher.

Preprocessing and candidate-detection batches are ordered by asset and frame index. Submissions already
in progress retain their previous seek order when resumed.

Refinement batches use candidate creation time, then asset, frame index, and
ROI index to approximate the order in which candidate payloads were written to
SkerryKV. This is a locality heuristic, not a physical disk-order guarantee;
creation timestamps are shared by a database transaction and storage can
deduplicate or reorganize payloads. In-progress submissions retain their
original cursor order when resumed.

The API's background dispatcher owns materialization independently of the
browser. Navigation after acceptance does not stop it. Pending submissions
survive API restarts; the saved boundary and cursor resume at the next
unreserved unit. Active per-stage reservations prevent retries and overlapping
submissions from scheduling the same work. Terminal reservations are released
in bounded cleanup passes. An API process must be running to dispatch pending
submissions. The dispatcher does not replace the workers that execute jobs.
Each dispatcher pass plans a small fair burst of at most four pages. Requests
rotate by their last planning time so one large request does not hold the
planner while other accepted requests wait.

`POST /processing/submissions/{id}/cancel` stops future planning immediately
and cancels already committed active jobs. Cleanup may finish shortly after the
response. Cancelled preprocessing and segmentation frames are eligible for an
ordinary later resubmission.

## Compatibility and operation

The existing synchronous `/processing/queue` and `/processing/series` endpoints
remain available, including dry-run previews. The durable endpoint is intended
for actual submissions. Those legacy synchronous endpoints do not participate
in durable submission fingerprinting or unit reservations; operators should not
use them concurrently with a durable request for the same selection. Install
the database migration before running the updated API with
`.venv/bin/pelagia migrate-db`, then run the matching PelagiaView version to use
quick acceptance.

Acceptance avoids work proportional to the number of matching frames, but it
still requires a network round trip and a database commit. There is no fixed
latency guarantee; background planning time still depends on selection size and
database load.
