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
reusing that ID with different criteria or options returns a conflict. Use a new
ID for a deliberately new submission.

## Monitoring and selection timing

`GET /processing/submissions` lists recent project submissions, and
`GET /processing/submissions/{id}` retrieves one submission. The lifecycle is
`accepted`, `planning`, then `succeeded` or `failed`. Here, `succeeded` means job
materialization finished. Follow the created jobs or series to monitor the
actual scientific processing. A successful submission can match no eligible
work; the result records zero matches and jobs.
Queue results include the full `matched_count` and `job_count`, plus up to 100
job IDs for immediate display. `job_ids_truncated` marks a shortened list; do not
derive total counts from that list. The recent-submission list prioritizes
accepted and planning requests, then recent terminal requests.

Criteria and options are saved at acceptance. Membership and eligibility are
evaluated when server-side planning runs. Changes to collections or processing
states between acceptance and planning can therefore change the selected work.
The created jobs retain their resolved frame or detection IDs for provenance.
For preprocessing and segmentation, omitted eligibility states default to
`unknown` and `failed` for the respective stage. Refinement defaults to
unrefined candidates. Preview frame coverage and the count of eligible ROI
candidates are different quantities, especially for partially refined frames.
Submitting a series preserves its existing behavior: each stage resolves its
eligibility when that stage is planned.

The API's background dispatcher owns materialization independently of the
browser. Navigation after acceptance does not stop it. Pending submissions
survive API restarts; interrupted planning can be recovered without creating a
second copy of the batch. An API process must be running to dispatch pending
submissions. The dispatcher does not replace the workers that execute jobs.

## Compatibility and operation

The existing synchronous `/processing/queue` and `/processing/series` endpoints
remain available, including dry-run previews. The durable endpoint is intended
for actual submissions. Install the database migration before running the
updated API with `.venv/bin/pelagia migrate-db`, then run the matching
PelagiaView version to use quick acceptance.

Acceptance avoids work proportional to the number of matching frames, but it
still requires a network round trip and a database commit. There is no fixed
latency guarantee; background planning time still depends on selection size and
database load.
