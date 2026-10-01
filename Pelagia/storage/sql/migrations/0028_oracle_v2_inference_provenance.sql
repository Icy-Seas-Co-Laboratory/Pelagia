-- Record the immutable Oracle Builder V2 contract selected for a Pelagia job
-- and retain request/item correlation across safe retries.

ALTER TABLE {schema}.classification_inference_runs
    ADD COLUMN IF NOT EXISTS oracle_artifact_id text,
    ADD COLUMN IF NOT EXISTS oracle_artifact_fingerprint_sha256 text,
    ADD COLUMN IF NOT EXISTS oracle_output_schema_version text;

CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_classification_inference_runs_job
    ON {schema}.classification_inference_runs(job_id)
    WHERE job_id IS NOT NULL AND oracle_artifact_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS {schema}.classification_inference_batches (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    inference_run_id uuid NOT NULL REFERENCES {schema}.classification_inference_runs(id) ON DELETE CASCADE,
    request_id text NOT NULL,
    batch_index integer NOT NULL CHECK (batch_index >= 0),
    status text NOT NULL DEFAULT 'reserved'
        CHECK (status IN ('reserved', 'complete', 'failed')),
    error_detail text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    completed_at timestamptz,
    UNIQUE (inference_run_id, request_id),
    UNIQUE (inference_run_id, batch_index)
);

CREATE TABLE IF NOT EXISTS {schema}.classification_inference_batch_items (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    inference_run_id uuid NOT NULL REFERENCES {schema}.classification_inference_runs(id) ON DELETE CASCADE,
    batch_id uuid NOT NULL REFERENCES {schema}.classification_inference_batches(id) ON DELETE CASCADE,
    refined_detection_id uuid NOT NULL REFERENCES {schema}.detections_refined(id) ON DELETE CASCADE,
    item_id text NOT NULL,
    input_sha256 text NOT NULL,
    status text NOT NULL DEFAULT 'reserved'
        CHECK (status IN ('reserved', 'succeeded', 'failed')),
    detail jsonb NOT NULL DEFAULT '{}'::jsonb,
    completed_at timestamptz,
    UNIQUE (inference_run_id, item_id),
    UNIQUE (inference_run_id, refined_detection_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_classification_inference_batch_items_batch
    ON {schema}.classification_inference_batch_items(batch_id, status);
