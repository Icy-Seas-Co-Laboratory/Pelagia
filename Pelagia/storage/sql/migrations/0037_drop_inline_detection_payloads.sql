-- Refuse to discard bytes that have not been copied and verified in SkerryKV.
DO $$
DECLARE
    v_table text;
    v_column text;
    has_payload boolean;
BEGIN
    FOREACH v_table IN ARRAY ARRAY['detection_candidate', 'detections_refined'] LOOP
        FOREACH v_column IN ARRAY ARRAY['roi_payload', 'mask_payload'] LOOP
            IF EXISTS (SELECT 1 FROM information_schema.columns c
                       WHERE c.table_schema = '{schema}' AND c.table_name = v_table
                         AND c.column_name = v_column) THEN
                EXECUTE format('SELECT EXISTS (SELECT 1 FROM %I.%I WHERE %I IS NOT NULL)',
                               '{schema}', v_table, v_column) INTO has_payload;
                IF has_payload THEN
                    RAISE EXCEPTION 'Migrate all inline candidate and refined ROI payloads to SkerryKV before applying this migration';
                END IF;
            END IF;
        END LOOP;
    END LOOP;
END $$;

ALTER TABLE {schema}.detection_candidate
    DROP COLUMN IF EXISTS roi_payload,
    DROP COLUMN IF EXISTS mask_payload;
ALTER TABLE {schema}.detections_refined
    DROP COLUMN IF EXISTS roi_payload,
    DROP COLUMN IF EXISTS mask_payload;
