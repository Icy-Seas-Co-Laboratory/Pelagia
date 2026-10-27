-- Queue detached object hashes for safe, reference-checked garbage collection.
CREATE TABLE IF NOT EXISTS {schema}.object_gc_candidates (
    hash text PRIMARY KEY,
    enqueued_at timestamptz NOT NULL DEFAULT NOW()
);

-- Keep candidate discovery in the transaction that removed a database reference.
-- The GC worker must still scan every reference column before deleting a hash,
-- because the same content-addressed object can be referenced by many rows and
-- by both configured storage roles.
CREATE OR REPLACE FUNCTION {schema}.enqueue_object_gc_candidates()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    column_name text;
    old_hash text;
    new_hash text;
BEGIN
    FOREACH column_name IN ARRAY TG_ARGV LOOP
        old_hash := to_jsonb(OLD) ->> column_name;
        IF old_hash IS NULL OR btrim(old_hash) = '' THEN
            CONTINUE;
        END IF;

        IF TG_OP = 'UPDATE' THEN
            new_hash := to_jsonb(NEW) ->> column_name;
            IF old_hash IS NOT DISTINCT FROM new_hash THEN
                CONTINUE;
            END IF;
        END IF;

        INSERT INTO {schema}.object_gc_candidates (hash)
        VALUES (old_hash)
        ON CONFLICT (hash) DO UPDATE
        SET enqueued_at = EXCLUDED.enqueued_at;
    END LOOP;

    IF TG_OP = 'DELETE' THEN
        RETURN OLD;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS frames_object_gc_candidates ON {schema}.frames;
CREATE TRIGGER frames_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.frames
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates(
    'kvstore_hash', 'payload_ref',
    'preprocessed_kvstore_hash', 'preprocessed_payload_ref',
    'background_kvstore_hash', 'background_payload_ref'
);

DROP TRIGGER IF EXISTS detection_candidate_object_gc_candidates ON {schema}.detection_candidate;
CREATE TRIGGER detection_candidate_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.detection_candidate
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates(
    'roi_payload_ref', 'mask_payload_ref'
);

DROP TRIGGER IF EXISTS detections_refined_object_gc_candidates ON {schema}.detections_refined;
CREATE TRIGGER detections_refined_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.detections_refined
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates(
    'roi_payload_ref', 'mask_payload_ref'
);

DROP TRIGGER IF EXISTS registry_assets_object_gc_candidates ON {schema}.registry_assets;
CREATE TRIGGER registry_assets_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.registry_assets
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates('payload_ref');

DROP TRIGGER IF EXISTS classification_evidence_object_gc_candidates ON {schema}.classification_evidence;
CREATE TRIGGER classification_evidence_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.classification_evidence
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates('embedding_payload_ref');

DROP TRIGGER IF EXISTS clustering_evidence_object_gc_candidates ON {schema}.clustering_evidence;
CREATE TRIGGER clustering_evidence_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.clustering_evidence
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates('embedding_payload_ref');

DROP TRIGGER IF EXISTS embedding_evidence_object_gc_candidates ON {schema}.embedding_evidence;
CREATE TRIGGER embedding_evidence_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.embedding_evidence
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates('embedding_payload_ref');

DROP TRIGGER IF EXISTS telemetry_sources_object_gc_candidates ON {schema}.telemetry_sources;
CREATE TRIGGER telemetry_sources_object_gc_candidates
AFTER UPDATE OR DELETE ON {schema}.telemetry_sources
FOR EACH ROW EXECUTE FUNCTION {schema}.enqueue_object_gc_candidates('source_payload_key');
