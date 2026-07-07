# Processing Status Projection

Pelagia maintains a project-scoped frame processing status projection so clients
can filter large frame sets without first compiling job, frame, detection, and
ROI state on the frontend.

The projection is stored in PostgreSQL in `frame_processing_status`. It has one
row per project and frame. Each row tracks:

- `preprocessing_status`
- `candidate_detection_status`
- `roi_refinement_status`
- candidate, refined, and unrefined ROI counts
- frame, asset, run, collection, and update metadata

Statuses use the same broad values as jobs:

```text
unknown, queued, leased, working, succeeded, failed, cancelled, dead_lettered
```

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

The response includes `summary` and `snapshot`. The snapshot contains a
`status_version` that changes when the project/session status summary changes or
when processing writes touch the project status state.

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

The summary, rows, and IDs endpoints support the same filters:

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

## Frontend Usage Pattern

For PelagiaView or another client:

1. Fetch `/processing/status/summary` when connecting to a project.
2. Use `/processing/status/frames/ids` for filtering large frame sets.
3. Page with `next_cursor` until no cursor is returned.
4. Re-fetch `/processing/status/summary` periodically or after job completion.
5. If `snapshot.status_version` changes, refresh any cached matching ID sets.

For projects around 1e6 frames, prefer the ID endpoint and cursor pagination.
Avoid unbounded requests from browser code.
