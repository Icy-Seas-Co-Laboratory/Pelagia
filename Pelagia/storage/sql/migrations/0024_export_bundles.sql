-- Durable, project-scoped records for reproducible export bundles.
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'export_bundle';

CREATE TABLE IF NOT EXISTS {schema}.export_artifacts (
    id uuid PRIMARY KEY DEFAULT {schema}.uuidv7(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    job_id uuid UNIQUE REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    requested_by_user_id uuid REFERENCES {schema}.users(id) ON DELETE SET NULL,
    requested_by_username text,
    status text NOT NULL DEFAULT 'queued',
    request jsonb NOT NULL DEFAULT '{}'::jsonb,
    manifest jsonb,
    snapshot_at timestamptz,
    artifact_path text,
    artifact_sha256 text,
    size_bytes bigint,
    expires_at timestamptz,
    failure_message text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    CONSTRAINT export_artifacts_status_check CHECK (status IN ('queued', 'working', 'succeeded', 'failed', 'cancelled', 'expired')),
    CONSTRAINT export_artifacts_size_check CHECK (size_bytes IS NULL OR size_bytes >= 0)
);
CREATE INDEX IF NOT EXISTS idx_{schema}_export_artifacts_project_created
    ON {schema}.export_artifacts (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_export_artifacts_expiry
    ON {schema}.export_artifacts (expires_at) WHERE expires_at IS NOT NULL;
DROP TRIGGER IF EXISTS export_artifacts_updated_at ON {schema}.export_artifacts;
CREATE TRIGGER export_artifacts_updated_at
    BEFORE UPDATE ON {schema}.export_artifacts
    FOR EACH ROW EXECUTE FUNCTION {schema}.set_updated_at();
