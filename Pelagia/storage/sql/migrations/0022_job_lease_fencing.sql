-- A lease token fences a worker that continues after its lease has been reclaimed.
-- It is intentionally nullable for historical/never-claimed rows.
ALTER TABLE {schema}.processing_jobs
    ADD COLUMN IF NOT EXISTS lease_token uuid;

ALTER TABLE {schema}.processing_series_steps
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW();

DROP TRIGGER IF EXISTS trg_processing_series_steps_updated_at ON {schema}.processing_series_steps;
CREATE TRIGGER trg_processing_series_steps_updated_at
BEFORE UPDATE ON {schema}.processing_series_steps
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_active_lease
    ON {schema}.processing_jobs (id, worker_id, lease_token)
    WHERE status IN ('leased', 'working');
