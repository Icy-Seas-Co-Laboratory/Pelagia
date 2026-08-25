-- Preserve the exact selection and resolved preset used for a submitted series.
ALTER TABLE {schema}.processing_series
    ADD COLUMN IF NOT EXISTS selection jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS preset_snapshot jsonb NOT NULL DEFAULT '{}'::jsonb;

ALTER TABLE {schema}.processing_series_steps
    ADD COLUMN IF NOT EXISTS skip_reason text;
