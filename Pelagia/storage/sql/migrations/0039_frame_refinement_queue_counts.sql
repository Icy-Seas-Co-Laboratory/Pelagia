-- NULL marks an existing frame that has not been backfilled.  Queue-count
-- readers must fall back to exact candidate rows until it has been refreshed.
ALTER TABLE {schema}.frame_processing_status
    ADD COLUMN IF NOT EXISTS refinement_queue_candidate_count integer,
    ADD COLUMN IF NOT EXISTS refinement_queue_refined_candidate_count integer;

-- NULL remains the default because a status row can be created for a frame
-- that already has detections. Reconciliation initializes its counts.
