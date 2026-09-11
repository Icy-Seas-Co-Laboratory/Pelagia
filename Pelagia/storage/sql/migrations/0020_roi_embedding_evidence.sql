-- Persist embedding-model outputs independently from legacy clustering evidence.
-- Embeddings describe a model-defined feature space; they do not assert clusters
-- or biological labels.

ALTER TABLE {schema}.classification_inference_runs
    DROP CONSTRAINT IF EXISTS classification_inference_runs_evidence_kind_check;

ALTER TABLE {schema}.classification_inference_runs
    ADD CONSTRAINT classification_inference_runs_evidence_kind_check
    CHECK (evidence_kind IN ('classification', 'clustering', 'embedding'));

CREATE TABLE IF NOT EXISTS {schema}.embedding_evidence (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    refined_detection_id uuid NOT NULL REFERENCES {schema}.detections_refined(id) ON DELETE CASCADE,
    inference_run_id uuid NOT NULL REFERENCES {schema}.classification_inference_runs(id) ON DELETE CASCADE,
    model_artifact_id uuid REFERENCES {schema}.model_artifacts(id) ON DELETE RESTRICT,
    embedding_payload_ref text NOT NULL,
    embedding_dtype text NOT NULL,
    embedding_shape jsonb NOT NULL,
    embedding_sha256 text NOT NULL,
    embedding_normalized boolean NOT NULL,
    output jsonb NOT NULL DEFAULT '{}'::jsonb,
    oracle_result jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (inference_run_id, refined_detection_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_embedding_evidence_roi_created
    ON {schema}.embedding_evidence(refined_detection_id, created_at DESC);
