# Processing Status Projection

Pelagia maintains a project-scoped frame processing status projection so clients
can filter large frame sets without first compiling job, frame, detection, and
ROI state on the frontend.

This projection describes stored frame coverage, not queue throughput. For job
counts, active work, progress, rates, and ETA, use the authoritative
[`/jobs/status` snapshot](queue-status.md). A frame can already have a stored
artifact while another job is reprocessing it, so coverage and queue progress
are intentionally distinct metrics.

The projection is stored in PostgreSQL in `frame_processing_status`. It has one
row per project and frame. Each row tracks:

- `preprocessing_status`
- `candidate_detection_status`
- `roi_refinement_status`
- candidate, refined, and unrefined ROI counts
- refinement-queue counts for eligible candidates and refined candidate parents
- frame, asset, run, collection, and update metadata

Statuses use the same broad values as jobs:

```text
unknown, queued, leased, working, succeeded, failed, cancelled, dead_lettered
```

The refinement job status is separate from coverage: a successful job does
not imply that every candidate in a frame has been refined. Queue counts include
only candidates with a stored ROI payload. A candidate is counted as refined
once if it has at least one refined result, even if it has multiple outputs.
Unrefined queue candidates are the difference between those two counts. The
broader frame counts retain their original meanings and can differ from queue
totals.

## When It Updates

The projection is maintained automatically by normal processing paths:

- job creation marks target frames as `queued` when the target frame IDs can be
  resolved.
- workers mark preprocessing, candidate detection, and ROI refinement as
  `working`, `succeeded`, or `failed`.
- direct stored-frame preprocessing updates preprocessing status when `store`
  is true.
- direct stored-frame segmentation updates candidate detection status and
  detection counts.
- direct stored ROI refinement updates ROI refinement status and refined counts.
- frame extraction creates projection rows for newly stored frames.

If old data predates this projection, or if an operator suspects drift, rebuild
the projection with the API below.

## API Endpoints

All endpoints require an authenticated session with access to the active
project.

### Summary

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/summary"
```

Useful filtered summary:

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/summary?asset_id=$ASSET_ID&candidate_detection_status=succeeded&has_refined_rois=false"
```

The response includes `summary` and `snapshot`. Snapshots are project-wide. The
`status_version` changes when processing writes touch the project status state.
Unfiltered summaries are cached until that happens; filtered summary requests
do not rebuild or write the project snapshot.

### Faceted Summary

Use the faceted endpoint for queue pages that need counts for many assets or
filter options. It compiles all grouped counts in one backend request instead
of issuing one summary request per asset:

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/facets?preprocessing_status=succeeded"
```

The response contains the normal `summary` and `snapshot`, plus:

```json
{
  "facets": {
    "assets": {"asset-uuid": 1200},
    "collections": {"cruise-a": 5000},
    "preprocessing_status": {"succeeded": 4500, "failed": 20},
    "candidate_detection_status": {"unknown": 4520},
    "roi_refinement_status": {"unknown": 4520},
    "refinement_state": {"refined": 300, "unrefined": 45}
  }
}
```

`asset_id` and `collection` may be repeated or comma-separated on this endpoint.
Multiple values use OR semantics within their group. All returned facet counts
describe the same filtered frame set as `summary`.

Queue previews can request `count_mode=auto` on the facets and refinement-count
endpoints. The default `exact` mode retains exact counting. Auto mode reads
small tables exactly; for tables larger than 1,024 heap pages it uses a
repeatable PostgreSQL block sample targeting 256 pages and scales the observed
counts by the sampling fraction. This limits expected read work independently
of the number of ROIs. It is not a hard latency limit or a confidence interval:
block clustering, a small project within a shared table, and narrow filters can
make estimates noisy or leave no matching sample rows.

Auto responses add `counts` metadata: `exact`, `method` (`exact` or
`block_sample`), `sample_fraction`, and `matching_sample_rows`. Clients must
label sampled totals as estimates. Zero `matching_sample_rows` on a sampled
response means insufficient sample evidence, not an empty queue. Estimated
facets must never replace the exact project snapshot. Submission criteria,
candidate reservations, and worker inputs always use exact predicates.

### Refinement Queue Counts

`GET /processing/status/refinement-counts` returns counts of candidate detections
with a stored ROI crop, grouped into `refined` and `unrefined` by candidate.
Each candidate counts once even if multiple refined outputs exist. Repeated
`asset_id` and `collection` parameters use the same filtering as facets. These
are current counts, not reservations; a submission may select different work
if data changes or candidates are reserved before planning completes.

Pass `min_area` and/or `max_area` to count candidates by inclusive
bounding-box area in px², for example `?min_area=64&max_area=4096`. The same
limits can be supplied in an `roi_refinement` queue request and are applied
again when each candidate is reserved for a job. Limits must be finite,
non-negative numbers and `min_area` cannot exceed `max_area`.

The `0040_roi_bbox_area_indexes` migration adds expression indexes for this
metric to candidate and refined ROI tables. Migrations run inside one
transaction, so an automatic index build can block writes on a large existing
catalog. Before applying the migration in such an environment, create both
indexes concurrently with these exact names; the migration's `IF NOT EXISTS`
then recognizes them:

```sql
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pelagia_dc_area
    ON pelagia.detection_candidate ((bbox_w::double precision * bbox_h), id);
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pelagia_dr_area
    ON pelagia.detections_refined ((bbox_w::double precision * bbox_h), id);
```

These commands use the default `pelagia` schema; substitute the configured
schema in both the relation and index names when needed. Run concurrent builds
outside a transaction, verify `pg_index.indisvalid` for both indexes, and
resolve an interrupted invalid build before applying the migration. Area
browsing relies on the default B-tree null ordering to allow both scan
directions; bbox dimensions are non-null, so this preserves size ordering.

### Frame Rows

Use `/processing/status/frames` when the frontend needs per-frame status fields:

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/frames?collection=test&preprocessing_status=succeeded&limit=1000"
```

Use cursor pagination for large projects:

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/frames?limit=10000&cursor=$NEXT_CURSOR"
```

### Frame IDs Only

Use `/processing/status/frames/ids` when the frontend only needs the matching
frame IDs:

```bash
curl -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/frames/ids?asset_id=$ASSET_ID&candidate_detection_status=succeeded&roi_refinement_status=unknown&limit=50000"
```

This endpoint is intended for high-cardinality workflows. It returns IDs plus a
`next_cursor` and avoids sending row metadata.

## Filters

The summary, faceted summary, rows, and IDs endpoints support the same filters.
The faceted endpoint additionally accepts multiple `asset_id` and `collection`
values:

- `run_id`
- `asset_id`
- `collection`
- `preprocessing_status`
- `candidate_detection_status`
- `roi_refinement_status`
- `has_candidates`
- `has_refined_rois`
- `start_frame`
- `end_frame`
- `limit`, `cursor`, and `offset` for row/ID endpoints

Status filters can be repeated or comma-separated:

```text
?preprocessing_status=succeeded&preprocessing_status=working
?candidate_detection_status=succeeded,working
```

## Rebuild And Backfill

Rebuild the whole active project:

```bash
curl -X POST -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/rebuild"
```

Rebuild one asset:

```bash
curl -X POST -H "Authorization: Bearer $PELAGIA_TOKEN" \
  "http://127.0.0.1:8000/processing/status/rebuild?asset_id=$ASSET_ID"
```

Rebuild uses existing frames, stored preprocessed payloads, candidate
detections, and refined detections to reconstruct the projection. It preserves
non-derived status values where there is no stored artifact proving success.
After upgrading to the frame-level refinement queue counts, rebuild each
existing asset to backfill them. Until all selected frames have been backfilled,
the refinement-count endpoint falls back to the exact candidate query, which
can be slow on large projects. Rebuilds read candidate and refined data and
can contend with processing; schedule large backfills accordingly. The rebuild
endpoint accepts one asset per request, so assets can be reconciled separately.

## Frontend Usage Pattern

For PelagiaView or another client:

1. Fetch `/processing/status/facets` once for queue-page totals and option counts.
2. Do not issue one `/summary` request per asset or collection.
3. Use `/processing/status/frames/ids` for filtering large frame sets.
4. Page with `next_cursor` until no cursor is returned.
5. Re-fetch facets after job completion or when `snapshot.status_version` changes.

For projects around 1e6 frames, prefer the ID endpoint and cursor pagination.
Avoid unbounded requests from browser code.
