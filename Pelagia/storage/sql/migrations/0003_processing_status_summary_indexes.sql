-- Adds selective indexes used by processing-status summaries and facets.

CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_run
    ON {schema}.frame_processing_status (project_id, run_id, asset_id, frame_index);

CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_frame_index
    ON {schema}.frame_processing_status (project_id, frame_index);

CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_has_candidates
    ON {schema}.frame_processing_status (project_id, asset_id)
    WHERE candidate_detection_count > 0;

CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_has_refined
    ON {schema}.frame_processing_status (project_id, asset_id)
    WHERE refined_detection_count > 0;
