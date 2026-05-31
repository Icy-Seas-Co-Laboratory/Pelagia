CREATE SCHEMA IF NOT EXISTS {schema};
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'asset_kind' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.asset_kind AS ENUM ('video', 'image', 'image_sequence');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'stage_name' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.stage_name AS ENUM ('ingest_run', 'extract_frames', 'segment', 'classify', 'publish', 'train_model', 'io_import', 'io_export', 'io_upload', 'io_download');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'job_status' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.job_status AS ENUM ('queued', 'leased', 'paused', 'succeeded', 'failed', 'cancelled', 'dead_lettered');
    END IF;
END $$;

ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'train_model';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_import';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_export';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_upload';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_download';

ALTER TYPE {schema}.job_status ADD VALUE IF NOT EXISTS 'paused';

CREATE OR REPLACE FUNCTION {schema}.set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TABLE IF NOT EXISTS {schema}.runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_key text NOT NULL UNIQUE,
    instrument text NOT NULL,
    source_path text NOT NULL,
    source_type text NOT NULL,
    status text NOT NULL DEFAULT 'registered',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.raw_assets (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    asset_key text NOT NULL,
    path text NOT NULL,
    kind {schema}.asset_kind NOT NULL,
    checksum text NOT NULL,
    size_bytes bigint NOT NULL,
    media_count integer NOT NULL DEFAULT 1,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, asset_key)
);

CREATE TABLE IF NOT EXISTS {schema}.frames (
    id bigserial PRIMARY KEY,
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    asset_id uuid NOT NULL REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    frame_index integer NOT NULL,
    captured_at timestamptz,
    width integer NOT NULL,
    height integer NOT NULL,
    bbox_x integer NOT NULL DEFAULT 0,
    bbox_y integer NOT NULL DEFAULT 0,
    parent_frame_id bigint REFERENCES {schema}.frames(id) ON DELETE SET NULL,
    source_ref text,
    frame_hash text NOT NULL,
    frame_png bytea NOT NULL,
    payload_ref text,
    payload_encoding text,
    payload_format text,
    payload_dtype text,
    payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (asset_id, frame_index)
);

ALTER TABLE {schema}.frames
    ADD COLUMN IF NOT EXISTS bbox_x integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS bbox_y integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS parent_frame_id bigint REFERENCES {schema}.frames(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS payload_ref text,
    ADD COLUMN IF NOT EXISTS payload_encoding text,
    ADD COLUMN IF NOT EXISTS payload_format text,
    ADD COLUMN IF NOT EXISTS payload_dtype text,
    ADD COLUMN IF NOT EXISTS payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb;

UPDATE {schema}.frames
SET
    bbox_x = COALESCE((metadata->>'bbox_x')::integer, bbox_x, 0),
    bbox_y = COALESCE((metadata->>'bbox_y')::integer, bbox_y, 0),
    parent_frame_id = COALESCE((metadata->>'parent_frame_id')::bigint, parent_frame_id),
    payload_ref = COALESCE(payload_ref, metadata->>'kvstore_key', frame_hash),
    payload_encoding = COALESCE(payload_encoding, metadata->>'kvstore_encoding', metadata->>'array_encoding'),
    payload_format = COALESCE(payload_format, metadata->>'kvstore_format'),
    payload_dtype = COALESCE(payload_dtype, metadata->>'dtype'),
    payload_shape = CASE
        WHEN payload_shape = '[]'::jsonb AND metadata ? 'shape' THEN metadata->'shape'
        ELSE payload_shape
    END;

CREATE TABLE IF NOT EXISTS {schema}.detections (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    frame_id bigint NOT NULL REFERENCES {schema}.frames(id) ON DELETE CASCADE,
    roi_index integer NOT NULL,
    bbox_x integer NOT NULL,
    bbox_y integer NOT NULL,
    bbox_w integer NOT NULL,
    bbox_h integer NOT NULL,
    crop_bbox_x integer,
    crop_bbox_y integer,
    crop_bbox_w integer,
    crop_bbox_h integer,
    area double precision,
    perimeter double precision,
    major_axis_length double precision,
    minor_axis_length double precision,
    min_gray_value integer,
    mean_gray_value double precision,
    roi_payload bytea,
    mask_payload bytea,
    roi_encoding text,
    roi_format text,
    roi_dtype text,
    roi_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    mask_encoding text,
    mask_format text,
    mask_dtype text,
    mask_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    CONSTRAINT detections_bbox_positive CHECK (bbox_w > 0 AND bbox_h > 0),
    CONSTRAINT detections_crop_bbox_positive CHECK (
        (
            crop_bbox_x IS NULL AND crop_bbox_y IS NULL
            AND crop_bbox_w IS NULL AND crop_bbox_h IS NULL
        )
        OR (
            crop_bbox_x IS NOT NULL AND crop_bbox_y IS NOT NULL
            AND crop_bbox_w > 0 AND crop_bbox_h > 0
        )
    ),
    UNIQUE (frame_id, roi_index)
);

ALTER TABLE {schema}.detections
    ADD COLUMN IF NOT EXISTS crop_bbox_x integer,
    ADD COLUMN IF NOT EXISTS crop_bbox_y integer,
    ADD COLUMN IF NOT EXISTS crop_bbox_w integer,
    ADD COLUMN IF NOT EXISTS crop_bbox_h integer,
    ADD COLUMN IF NOT EXISTS roi_payload bytea,
    ADD COLUMN IF NOT EXISTS mask_payload bytea,
    ADD COLUMN IF NOT EXISTS roi_encoding text,
    ADD COLUMN IF NOT EXISTS roi_format text,
    ADD COLUMN IF NOT EXISTS roi_dtype text,
    ADD COLUMN IF NOT EXISTS roi_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS mask_encoding text,
    ADD COLUMN IF NOT EXISTS mask_format text,
    ADD COLUMN IF NOT EXISTS mask_dtype text,
    ADD COLUMN IF NOT EXISTS mask_shape jsonb NOT NULL DEFAULT '[]'::jsonb;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = 'detections'
          AND column_name = 'crop_png'
    ) THEN
        EXECUTE 'UPDATE {schema}.detections SET roi_payload = COALESCE(roi_payload, crop_png) WHERE roi_payload IS NULL';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = 'detections'
          AND column_name = 'mask_png'
    ) THEN
        EXECUTE 'UPDATE {schema}.detections SET mask_payload = COALESCE(mask_payload, mask_png) WHERE mask_payload IS NULL';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'detections_bbox_positive'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.detections
            ADD CONSTRAINT detections_bbox_positive CHECK (bbox_w > 0 AND bbox_h > 0);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'detections_crop_bbox_positive'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.detections
            ADD CONSTRAINT detections_crop_bbox_positive CHECK (
                (
                    crop_bbox_x IS NULL AND crop_bbox_y IS NULL
                    AND crop_bbox_w IS NULL AND crop_bbox_h IS NULL
                )
                OR (
                    crop_bbox_x IS NOT NULL AND crop_bbox_y IS NOT NULL
                    AND crop_bbox_w > 0 AND crop_bbox_h > 0
                )
            );
    END IF;
END $$;

UPDATE {schema}.detections
SET
    roi_encoding = COALESCE(roi_encoding, metadata->>'roi_encoding', metadata->>'array_encoding'),
    roi_format = COALESCE(roi_format, metadata->>'roi_format'),
    roi_dtype = COALESCE(roi_dtype, metadata->>'dtype'),
    roi_shape = CASE
        WHEN roi_shape = '[]'::jsonb AND metadata ? 'shape' THEN metadata->'shape'
        ELSE roi_shape
    END,
    mask_encoding = COALESCE(mask_encoding, metadata->>'mask_encoding'),
    mask_format = COALESCE(mask_format, metadata->>'mask_format'),
    mask_dtype = COALESCE(mask_dtype, metadata->>'mask_dtype'),
    mask_shape = CASE
        WHEN mask_shape = '[]'::jsonb AND metadata ? 'mask_shape' THEN metadata->'mask_shape'
        ELSE mask_shape
    END,
    crop_bbox_x = COALESCE(
        crop_bbox_x,
        CASE WHEN metadata ? 'roi_bbox' THEN ((metadata->'roi_bbox')->>0)::integer END
    ),
    crop_bbox_y = COALESCE(
        crop_bbox_y,
        CASE WHEN metadata ? 'roi_bbox' THEN ((metadata->'roi_bbox')->>1)::integer END
    ),
    crop_bbox_w = COALESCE(
        crop_bbox_w,
        CASE WHEN metadata ? 'roi_bbox' THEN ((metadata->'roi_bbox')->>2)::integer END
    ),
    crop_bbox_h = COALESCE(
        crop_bbox_h,
        CASE WHEN metadata ? 'roi_bbox' THEN ((metadata->'roi_bbox')->>3)::integer END
    );

CREATE TABLE IF NOT EXISTS {schema}.models (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_key text NOT NULL UNIQUE,
    model_name text NOT NULL,
    version text NOT NULL,
    task text NOT NULL DEFAULT 'classification',
    artifact_uri text,
    labels jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.classification_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id uuid NOT NULL REFERENCES {schema}.detections(id) ON DELETE CASCADE,
    model_id uuid NOT NULL REFERENCES {schema}.models(id) ON DELETE CASCADE,
    label text,
    score double precision,
    scores jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (detection_id, model_id)
);

CREATE TABLE IF NOT EXISTS {schema}.processing_jobs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    asset_id uuid REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    stage {schema}.stage_name NOT NULL,
    status {schema}.job_status NOT NULL DEFAULT 'queued',
    priority integer NOT NULL DEFAULT 100,
    attempt_count integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 3,
    lease_expires_at timestamptz,
    worker_id text,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    result jsonb,
    progress jsonb NOT NULL DEFAULT '{}'::jsonb,
    logs_tail jsonb NOT NULL DEFAULT '[]'::jsonb,
    summary text,
    control_reason text,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    started_at timestamptz,
    finished_at timestamptz
);

CREATE TABLE IF NOT EXISTS {schema}.processing_job_dependencies (
    job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    depends_on_job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, depends_on_job_id)
);

CREATE TABLE IF NOT EXISTS {schema}.worker_sessions (
    worker_id text PRIMARY KEY,
    pid integer,
    status text NOT NULL DEFAULT 'idle',
    leased_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    capabilities jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    shutdown_requested boolean NOT NULL DEFAULT false,
    last_heartbeat timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW()
);

ALTER TABLE {schema}.worker_sessions
    ADD COLUMN IF NOT EXISTS pid integer,
    ADD COLUMN IF NOT EXISTS shutdown_requested boolean NOT NULL DEFAULT false;

CREATE TABLE IF NOT EXISTS {schema}.job_events (
    id bigserial PRIMARY KEY,
    job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    event_type text NOT NULL,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_{schema}_raw_assets_run_id ON {schema}.raw_assets (run_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frames_asset_id ON {schema}.frames (asset_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_{schema}_detections_frame_id ON {schema}.detections (frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_classification_results_detection_id ON {schema}.classification_results (detection_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_status ON {schema}.processing_jobs (status, stage, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_run_id ON {schema}.processing_jobs (run_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_job_dependencies_depends_on ON {schema}.processing_job_dependencies (depends_on_job_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_job_events_job_id ON {schema}.job_events (job_id, id);

DROP TRIGGER IF EXISTS trg_runs_updated_at ON {schema}.runs;
CREATE TRIGGER trg_runs_updated_at
BEFORE UPDATE ON {schema}.runs
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

DROP TRIGGER IF EXISTS trg_processing_jobs_updated_at ON {schema}.processing_jobs;
CREATE TRIGGER trg_processing_jobs_updated_at
BEFORE UPDATE ON {schema}.processing_jobs
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

DROP TRIGGER IF EXISTS trg_worker_sessions_updated_at ON {schema}.worker_sessions;
CREATE TRIGGER trg_worker_sessions_updated_at
BEFORE UPDATE ON {schema}.worker_sessions
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();
