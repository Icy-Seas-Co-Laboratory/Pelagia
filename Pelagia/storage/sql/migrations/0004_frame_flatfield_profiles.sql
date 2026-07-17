ALTER TABLE {schema}.frames
    ADD COLUMN IF NOT EXISTS flatfield_profile real[],
    ADD COLUMN IF NOT EXISTS flatfield_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;
