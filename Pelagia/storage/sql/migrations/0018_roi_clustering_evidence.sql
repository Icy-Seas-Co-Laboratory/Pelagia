-- Persist Oracle Builder self-supervised clustering evidence separately from
-- classification evidence. Cluster IDs are run-local evidence, not labels.

ALTER TABLE {schema}.classification_inference_runs
    ADD COLUMN IF NOT EXISTS evidence_kind text NOT NULL DEFAULT 'classification';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'classification_inference_runs_evidence_kind_check'
          AND conrelid = '{schema}.classification_inference_runs'::regclass
    ) THEN
        ALTER TABLE {schema}.classification_inference_runs
            ADD CONSTRAINT classification_inference_runs_evidence_kind_check
            CHECK (evidence_kind IN ('classification', 'clustering'));
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {schema}.clustering_evidence (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    refined_detection_id uuid NOT NULL REFERENCES {schema}.detections_refined(id) ON DELETE CASCADE,
    inference_run_id uuid NOT NULL REFERENCES {schema}.classification_inference_runs(id) ON DELETE CASCADE,
    model_artifact_id uuid REFERENCES {schema}.model_artifacts(id) ON DELETE RESTRICT,
    embedding_payload_ref text,
    embedding_dtype text,
    embedding_shape jsonb,
    embedding_sha256 text,
    cluster_index integer,
    cluster_id text,
    similarity double precision,
    similarity_floor double precision,
    novelty_similarity_threshold double precision,
    novel boolean,
    abstained boolean,
    candidate_clusters jsonb NOT NULL DEFAULT '[]'::jsonb,
    nearest_neighbors jsonb NOT NULL DEFAULT '[]'::jsonb,
    evidence_packet jsonb NOT NULL DEFAULT '{}'::jsonb,
    oracle_result jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (inference_run_id, refined_detection_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_clustering_evidence_roi_created
    ON {schema}.clustering_evidence(refined_detection_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_clustering_evidence_project_cluster
    ON {schema}.clustering_evidence(project_id, cluster_id, created_at DESC);
