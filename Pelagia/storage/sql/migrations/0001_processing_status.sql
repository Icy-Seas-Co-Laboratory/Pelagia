-- Adds the project/frame processing status projection used for scalable
-- frontend filtering. This migration is intentionally idempotent because older
-- development databases may already have received these objects via the base
-- schema bootstrap.

ALTER TYPE {schema}.job_status ADD VALUE IF NOT EXISTS 'working';

CREATE TABLE IF NOT EXISTS {schema}.project_processing_status_snapshots (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    session_id uuid REFERENCES {schema}.user_sessions(id) ON DELETE CASCADE,
    status_version bigint NOT NULL DEFAULT 0,
    generated_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    summary jsonb NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE {schema}.project_processing_status_snapshots
    ADD COLUMN IF NOT EXISTS session_id uuid REFERENCES {schema}.user_sessions(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS status_version bigint NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS generated_at timestamptz,
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS summary jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS {schema}.frame_processing_status (
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    frame_id uuid NOT NULL REFERENCES {schema}.frames(id) ON DELETE CASCADE,
    asset_id uuid NOT NULL REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    run_id uuid REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    frame_index integer NOT NULL,
    collections text[] NOT NULL DEFAULT ARRAY[]::text[],
    preprocessing_status text NOT NULL DEFAULT 'unknown',
    preprocessing_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    preprocessing_completed_at timestamptz,
    candidate_detection_status text NOT NULL DEFAULT 'unknown',
    candidate_detection_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    candidate_detection_completed_at timestamptz,
    candidate_detection_count integer NOT NULL DEFAULT 0,
    roi_refinement_status text NOT NULL DEFAULT 'unknown',
    roi_refinement_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    roi_refinement_completed_at timestamptz,
    refined_detection_count integer NOT NULL DEFAULT 0,
    unrefined_candidate_count integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (project_id, frame_id),
    CONSTRAINT frame_processing_status_preprocessing_known CHECK (
        preprocessing_status IN ('unknown', 'queued', 'leased', 'working', 'succeeded', 'failed', 'cancelled', 'dead_lettered')
    ),
    CONSTRAINT frame_processing_status_candidate_detection_known CHECK (
        candidate_detection_status IN ('unknown', 'queued', 'leased', 'working', 'succeeded', 'failed', 'cancelled', 'dead_lettered')
    ),
    CONSTRAINT frame_processing_status_roi_refinement_known CHECK (
        roi_refinement_status IN ('unknown', 'queued', 'leased', 'working', 'succeeded', 'failed', 'cancelled', 'dead_lettered')
    )
);

ALTER TABLE {schema}.frame_processing_status
    ADD COLUMN IF NOT EXISTS run_id uuid REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS frame_index integer,
    ADD COLUMN IF NOT EXISTS collections text[] NOT NULL DEFAULT ARRAY[]::text[],
    ADD COLUMN IF NOT EXISTS preprocessing_status text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS preprocessing_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS preprocessing_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS candidate_detection_status text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS candidate_detection_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS candidate_detection_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS candidate_detection_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS roi_refinement_status text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS roi_refinement_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS roi_refinement_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS refined_detection_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS unrefined_candidate_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW();

CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_project_processing_status_snapshots_session
    ON {schema}.project_processing_status_snapshots (project_id, session_id)
    WHERE session_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_project_processing_status_snapshots_project
    ON {schema}.project_processing_status_snapshots (project_id)
    WHERE session_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_asset_frame
    ON {schema}.frame_processing_status (project_id, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_preprocessing
    ON {schema}.frame_processing_status (project_id, preprocessing_status, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_candidate_detection
    ON {schema}.frame_processing_status (project_id, candidate_detection_status, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_roi_refinement
    ON {schema}.frame_processing_status (project_id, roi_refinement_status, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_updated
    ON {schema}.frame_processing_status (project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_collections
    ON {schema}.frame_processing_status USING gin (collections);
