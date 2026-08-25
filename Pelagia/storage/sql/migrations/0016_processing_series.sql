-- Durable, project-scoped orchestration for multi-stage processing queues.
CREATE TABLE IF NOT EXISTS {schema}.processing_series (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'active', 'paused', 'succeeded', 'failed', 'cancelled')),
    failure_policy text NOT NULL DEFAULT 'fail_fast' CHECK (failure_policy IN ('fail_fast', 'continue')),
    priority integer NOT NULL DEFAULT 100,
    submitted_by_user_id text,
    submitted_by_username text,
    control_reason text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS {schema}.processing_series_steps (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    series_id uuid NOT NULL REFERENCES {schema}.processing_series(id) ON DELETE CASCADE,
    step_index integer NOT NULL CHECK (step_index >= 0),
    stage {schema}.stage_name NOT NULL,
    filters jsonb NOT NULL DEFAULT '{}'::jsonb,
    options jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'planning', 'active', 'skipped', 'succeeded', 'failed', 'cancelled')),
    matched_count bigint NOT NULL DEFAULT 0,
    job_count bigint NOT NULL DEFAULT 0,
    failure_policy text CHECK (failure_policy IN ('fail_fast', 'continue')),
    started_at timestamptz,
    finished_at timestamptz,
    UNIQUE (series_id, step_index)
);

CREATE TABLE IF NOT EXISTS {schema}.processing_work_units (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    series_id uuid NOT NULL REFERENCES {schema}.processing_series(id) ON DELETE CASCADE,
    step_id uuid NOT NULL REFERENCES {schema}.processing_series_steps(id) ON DELETE CASCADE,
    job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE RESTRICT,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (job_id),
    UNIQUE (step_id, job_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_processing_series_project ON {schema}.processing_series (project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_series_steps_series ON {schema}.processing_series_steps (series_id, step_index);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_work_units_step ON {schema}.processing_work_units (step_id, job_id);
