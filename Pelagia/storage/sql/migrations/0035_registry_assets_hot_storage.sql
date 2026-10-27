-- Registry datasets can contain ROI images and masks. Keep their bytes in the
-- hot object store while PostgreSQL retains the portable contract metadata.
ALTER TABLE {schema}.registry_assets
    ADD COLUMN IF NOT EXISTS payload_ref text,
    ADD COLUMN IF NOT EXISTS payload_bytes bigint;

CREATE INDEX IF NOT EXISTS idx_{schema}_registry_assets_hot_pending
    ON {schema}.registry_assets (workspace_id, asset_id)
    WHERE payload IS NOT NULL AND payload_ref IS NULL;
