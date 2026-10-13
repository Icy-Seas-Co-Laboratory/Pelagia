-- Index event age for bounded PostgreSQL operational log cleanup.
CREATE INDEX IF NOT EXISTS idx_{schema}_job_events_created_at
    ON {schema}.job_events (created_at, id);
