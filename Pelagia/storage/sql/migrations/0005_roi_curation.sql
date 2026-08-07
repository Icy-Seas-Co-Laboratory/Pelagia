-- Project-scoped ROI curation and derived classification evidence.
-- Human assertions and model evidence are deliberately separate.

CREATE TABLE IF NOT EXISTS {schema}.classification_labels (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    name text NOT NULL,
    display_name text,
    parent_label_id uuid REFERENCES {schema}.classification_labels(id) ON DELETE RESTRICT,
    rank text,
    description text,
    stable_concept_id text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    deprecated_at timestamptz,
    UNIQUE (project_id, name)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_classification_labels_project_active
    ON {schema}.classification_labels(project_id, deprecated_at, name);

CREATE TABLE IF NOT EXISTS {schema}.model_artifacts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    artifact_id uuid NOT NULL,
    run_id uuid,
    artifact_fingerprint text,
    task text NOT NULL,
    architecture text,
    contract_version text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, artifact_id)
);

CREATE TABLE IF NOT EXISTS {schema}.model_class_mappings (
    model_artifact_id uuid NOT NULL REFERENCES {schema}.model_artifacts(id) ON DELETE CASCADE,
    class_index integer NOT NULL,
    oracle_label_id text,
    oracle_label_name text,
    project_label_id uuid REFERENCES {schema}.classification_labels(id) ON DELETE SET NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (model_artifact_id, class_index)
);

CREATE TABLE IF NOT EXISTS {schema}.classification_inference_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    model_artifact_id uuid REFERENCES {schema}.model_artifacts(id) ON DELETE RESTRICT,
    model_selector text NOT NULL,
    status text NOT NULL DEFAULT 'running'
        CHECK (status IN ('running', 'complete', 'failed', 'cancelled')),
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    completed_at timestamptz
);

CREATE TABLE IF NOT EXISTS {schema}.classification_evidence (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    refined_detection_id uuid NOT NULL REFERENCES {schema}.detections_refined(id) ON DELETE CASCADE,
    inference_run_id uuid NOT NULL REFERENCES {schema}.classification_inference_runs(id) ON DELETE CASCADE,
    predicted_label_id uuid REFERENCES {schema}.classification_labels(id) ON DELETE SET NULL,
    predicted_class_index integer,
    predicted_label_name text,
    confidence double precision,
    entropy double precision,
    probability_margin double precision,
    prototype_class_index integer,
    prototype_similarity double precision,
    prototype_margin double precision,
    knn_class_index integer,
    knn_agreement double precision,
    knn_weighted_support double precision,
    knn_margin double precision,
    embedding_payload_ref text,
    embedding_dtype text,
    embedding_shape jsonb,
    embedding_sha256 text,
    probabilities jsonb NOT NULL DEFAULT '[]'::jsonb,
    evidence_packet jsonb NOT NULL DEFAULT '{}'::jsonb,
    oracle_result jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (inference_run_id, refined_detection_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_classification_evidence_roi_created
    ON {schema}.classification_evidence(refined_detection_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_classification_evidence_review_queue
    ON {schema}.classification_evidence(project_id, confidence, probability_margin, knn_margin);

CREATE TABLE IF NOT EXISTS {schema}.classification_evidence_neighbors (
    evidence_id uuid NOT NULL REFERENCES {schema}.classification_evidence(id) ON DELETE CASCADE,
    rank integer NOT NULL,
    exemplar_id text NOT NULL,
    class_index integer,
    label_name text,
    similarity double precision NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (evidence_id, rank)
);

CREATE TABLE IF NOT EXISTS {schema}.roi_label_annotations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    refined_detection_id uuid NOT NULL REFERENCES {schema}.detections_refined(id) ON DELETE CASCADE,
    label_id uuid NOT NULL REFERENCES {schema}.classification_labels(id) ON DELETE RESTRICT,
    actor_user_id uuid REFERENCES {schema}.users(id) ON DELETE SET NULL,
    actor_username text NOT NULL,
    method text NOT NULL DEFAULT 'human',
    status text NOT NULL DEFAULT 'accepted'
        CHECK (status IN ('accepted', 'ambiguous', 'deprecated')),
    is_current boolean NOT NULL DEFAULT true,
    parent_annotation_id uuid REFERENCES {schema}.roi_label_annotations(id) ON DELETE RESTRICT,
    suggested_by_evidence_id uuid REFERENCES {schema}.classification_evidence(id) ON DELETE SET NULL,
    notes text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_roi_label_annotation_current
    ON {schema}.roi_label_annotations(refined_detection_id) WHERE is_current;
CREATE INDEX IF NOT EXISTS idx_{schema}_roi_label_annotation_label
    ON {schema}.roi_label_annotations(project_id, label_id) WHERE is_current;

CREATE TABLE IF NOT EXISTS {schema}.roi_annotation_reviews (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    annotation_id uuid NOT NULL REFERENCES {schema}.roi_label_annotations(id) ON DELETE CASCADE,
    reviewer_user_id uuid REFERENCES {schema}.users(id) ON DELETE SET NULL,
    reviewer_username text NOT NULL,
    decision text NOT NULL CHECK (decision IN ('verified', 'rejected', 'needs_review')),
    notes text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_{schema}_roi_annotation_reviews_latest
    ON {schema}.roi_annotation_reviews(annotation_id, created_at DESC);
