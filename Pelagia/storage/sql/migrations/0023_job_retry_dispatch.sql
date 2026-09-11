-- Scheduled retry metadata, idempotent jobs, and durable successor dispatches.
ALTER TABLE {schema}.processing_jobs
    ADD COLUMN IF NOT EXISTS available_at timestamptz NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS failure_category text,
    ADD COLUMN IF NOT EXISTS idempotency_key text;

UPDATE {schema}.processing_jobs SET available_at = COALESCE(available_at, created_at, NOW());

CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_idempotency
    ON {schema}.processing_jobs (project_id, idempotency_key) WHERE idempotency_key IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_available
    ON {schema}.processing_jobs (available_at, priority, created_at) WHERE status = 'queued';

CREATE TABLE IF NOT EXISTS {schema}.processing_job_dispatches (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    parent_job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    idempotency_key text NOT NULL,
    job_spec jsonb NOT NULL,
    status text NOT NULL DEFAULT 'pending',
    child_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    error_message text,
    available_at timestamptz NOT NULL DEFAULT NOW(),
    materialized_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, idempotency_key)
);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_job_dispatches_pending
    ON {schema}.processing_job_dispatches (available_at, created_at) WHERE status = 'pending';
