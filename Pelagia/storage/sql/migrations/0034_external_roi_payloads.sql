-- Store candidate and refined ROI bytes in the configured hot object store.
ALTER TABLE {schema}.detection_candidate
    ADD COLUMN IF NOT EXISTS roi_payload_ref text,
    ADD COLUMN IF NOT EXISTS mask_payload_ref text,
    ADD COLUMN IF NOT EXISTS roi_payload_bytes bigint,
    ADD COLUMN IF NOT EXISTS mask_payload_bytes bigint;

ALTER TABLE {schema}.detections_refined
    ADD COLUMN IF NOT EXISTS roi_payload_ref text,
    ADD COLUMN IF NOT EXISTS mask_payload_ref text,
    ADD COLUMN IF NOT EXISTS roi_payload_bytes bigint,
    ADD COLUMN IF NOT EXISTS mask_payload_bytes bigint;
