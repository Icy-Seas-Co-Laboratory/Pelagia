-- Durable source snapshots and retry-safe identity for telemetry imports.

ALTER TABLE {schema}.telemetry_sources
    ADD COLUMN IF NOT EXISTS import_key text,
    ADD COLUMN IF NOT EXISTS source_payload_key text;

CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_telemetry_sources_import_key
    ON {schema}.telemetry_sources(project_id, run_id, import_key)
    WHERE import_key IS NOT NULL;
