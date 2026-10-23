-- Checkpoint criteria expansion so workers can consume committed batches while
-- the remainder of a large processing submission is still being planned.
ALTER TABLE {schema}.processing_submissions
    DROP CONSTRAINT IF EXISTS processing_submissions_status_check;

ALTER TABLE {schema}.processing_submissions
    ADD CONSTRAINT processing_submissions_status_check
        CHECK (status IN ('accepted', 'planning', 'processing', 'succeeded', 'failed', 'cancelled')),
    ADD COLUMN IF NOT EXISTS selection_boundary jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS planning_cursor jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS progress jsonb NOT NULL DEFAULT '{"phase":"finding_eligible","scheduled_count":0,"job_count":0,"queued_count":0,"running_count":0,"paused_count":0,"completed_count":0,"failed_count":0,"cancelled_count":0,"more_jobs_being_prepared":true}'::jsonb,
    ADD COLUMN IF NOT EXISTS last_planned_at timestamptz,
    ADD COLUMN IF NOT EXISTS planning_finished_at timestamptz,
    ADD COLUMN IF NOT EXISTS next_plan_at timestamptz,
    ADD COLUMN IF NOT EXISTS cancellation_reason text;

ALTER TABLE {schema}.processing_series
    ADD COLUMN IF NOT EXISTS submission_id uuid REFERENCES {schema}.processing_submissions(id) ON DELETE SET NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_ps_series_unique
    ON {schema}.processing_series (submission_id) WHERE submission_id IS NOT NULL;

ALTER TABLE {schema}.processing_series_steps
    ADD COLUMN IF NOT EXISTS planning_complete boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS selection_boundary jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS planning_cursor jsonb NOT NULL DEFAULT '{}'::jsonb;

UPDATE {schema}.processing_series_steps
SET planning_complete = true
WHERE status IN ('active', 'skipped', 'succeeded', 'failed', 'cancelled');

CREATE TABLE IF NOT EXISTS {schema}.processing_submission_units (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    submission_id uuid NOT NULL REFERENCES {schema}.processing_submissions(id) ON DELETE CASCADE,
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    stage text NOT NULL CHECK (stage IN ('preprocess_frames', 'segment', 'roi_refinement')),
    step_id uuid REFERENCES {schema}.processing_series_steps(id) ON DELETE CASCADE,
    unit_id uuid NOT NULL,
    job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    released_at timestamptz,
    UNIQUE (submission_id, stage, unit_id)
);

CREATE TABLE IF NOT EXISTS {schema}.processing_submission_jobs (
    submission_id uuid NOT NULL REFERENCES {schema}.processing_submissions(id) ON DELETE CASCADE,
    job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    step_id uuid REFERENCES {schema}.processing_series_steps(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (submission_id, job_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_psu_active
    ON {schema}.processing_submission_units (project_id, stage, unit_id)
    WHERE released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_{schema}_psu_submission
    ON {schema}.processing_submission_units (submission_id, job_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_psu_unreleased
    ON {schema}.processing_submission_units (created_at, id) WHERE released_at IS NULL;
CREATE INDEX IF NOT EXISTS idx_{schema}_psj_submission
    ON {schema}.processing_submission_jobs (submission_id, created_at);
CREATE INDEX IF NOT EXISTS idx_{schema}_ps_fair
    ON {schema}.processing_submissions (COALESCE(last_planned_at, created_at), created_at)
    WHERE status IN ('accepted', 'planning');
CREATE INDEX IF NOT EXISTS idx_{schema}_frames_seek
    ON {schema}.frames (created_at, id);
CREATE INDEX IF NOT EXISTS idx_{schema}_dc_seek
    ON {schema}.detection_candidate (created_at, id);
