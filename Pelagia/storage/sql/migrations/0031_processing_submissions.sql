-- Persist criteria-based processing requests before any frame expansion.
CREATE TABLE IF NOT EXISTS {schema}.processing_submissions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    request_id uuid NOT NULL,
    kind text NOT NULL CHECK (kind IN ('queue', 'series')),
    request jsonb NOT NULL,
    request_hash text NOT NULL,
    status text NOT NULL DEFAULT 'accepted'
        CHECK (status IN ('accepted', 'planning', 'succeeded', 'failed', 'cancelled')),
    claim_token uuid,
    claim_expires_at timestamptz,
    result jsonb,
    error text,
    submitted_by_user_id text,
    submitted_by_username text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    finished_at timestamptz,
    UNIQUE (project_id, request_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_processing_submissions_dispatch
    ON {schema}.processing_submissions (status, created_at)
    WHERE status IN ('accepted', 'planning');

DROP TRIGGER IF EXISTS trg_processing_submissions_updated_at ON {schema}.processing_submissions;
CREATE TRIGGER trg_processing_submissions_updated_at
BEFORE UPDATE ON {schema}.processing_submissions
FOR EACH ROW EXECUTE FUNCTION {schema}.set_updated_at();
