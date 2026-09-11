-- Immutable input membership, preflight estimates, and durable retry history.
ALTER TABLE {schema}.export_artifacts
    ADD COLUMN IF NOT EXISTS input_snapshot jsonb,
    ADD COLUMN IF NOT EXISTS estimate jsonb,
    ADD COLUMN IF NOT EXISTS progress jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS attempts jsonb NOT NULL DEFAULT '[]'::jsonb;
