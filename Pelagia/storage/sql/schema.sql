CREATE SCHEMA IF NOT EXISTS {schema};
CREATE EXTENSION IF NOT EXISTS pgcrypto;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'asset_kind' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.asset_kind AS ENUM ('video', 'image', 'image_sequence');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'stage_name' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.stage_name AS ENUM ('ingest_run', 'extract_frames', 'background_frames', 'preprocess_frames', 'segment', 'roi_refinement', 'classify', 'publish', 'train_model', 'io_import', 'io_export', 'io_upload', 'io_download');
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_type t JOIN pg_namespace n ON n.oid = t.typnamespace WHERE t.typname = 'job_status' AND n.nspname = '{schema}') THEN
        CREATE TYPE {schema}.job_status AS ENUM ('queued', 'leased', 'working', 'paused', 'succeeded', 'failed', 'cancelled', 'dead_lettered');
    END IF;
END $$;

ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'train_model';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'background_frames';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'preprocess_frames';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'roi_refinement';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_import';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_export';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_upload';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'io_download';

ALTER TYPE {schema}.job_status ADD VALUE IF NOT EXISTS 'paused';
ALTER TYPE {schema}.job_status ADD VALUE IF NOT EXISTS 'working';

CREATE OR REPLACE FUNCTION {schema}.set_updated_at()
RETURNS trigger AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION {schema}.uuidv7()
RETURNS uuid AS $$
DECLARE
    unix_ts_ms bigint;
    hex_ts text;
    rand_a integer;
    rand_b text;
BEGIN
    unix_ts_ms := floor(extract(epoch from clock_timestamp()) * 1000)::bigint;
    hex_ts := lpad(to_hex(unix_ts_ms), 12, '0');
    rand_a := floor(random() * 4096)::integer;
    rand_b := encode(gen_random_bytes(8), 'hex');
    RETURN (
        substr(hex_ts, 1, 8) || '-' ||
        substr(hex_ts, 9, 4) || '-' ||
        '7' || lpad(to_hex(rand_a), 3, '0') || '-' ||
        substr('89ab', floor(random() * 4)::integer + 1, 1) || substr(rand_b, 1, 3) || '-' ||
        substr(rand_b, 4, 12)
    )::uuid;
END;
$$ LANGUAGE plpgsql VOLATILE;

CREATE TABLE IF NOT EXISTS {schema}.schema_migrations (
    migration_id text PRIMARY KEY,
    checksum text NOT NULL,
    description text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    applied_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.users (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    username text NOT NULL UNIQUE,
    display_name text,
    password_hash text,
    is_active boolean NOT NULL DEFAULT true,
    is_admin boolean NOT NULL DEFAULT false,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.projects (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_key text NOT NULL UNIQUE,
    project_name text NOT NULL,
    description text,
    kvstore_root_path text,
    is_active boolean NOT NULL DEFAULT true,
    settings jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.project_memberships (
    user_id uuid NOT NULL REFERENCES {schema}.users(id) ON DELETE CASCADE,
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    role text NOT NULL DEFAULT 'viewer',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (user_id, project_id),
    CONSTRAINT project_memberships_role_known CHECK (role IN ('viewer', 'editor', 'manager', 'admin'))
);

CREATE TABLE IF NOT EXISTS {schema}.user_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES {schema}.users(id) ON DELETE CASCADE,
    project_id uuid REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    token_hash text NOT NULL UNIQUE,
    user_agent text,
    remote_addr text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    expires_at timestamptz NOT NULL,
    revoked_at timestamptz,
    last_seen_at timestamptz NOT NULL DEFAULT NOW(),
    created_at timestamptz NOT NULL DEFAULT NOW()
);

ALTER TABLE {schema}.users
    ADD COLUMN IF NOT EXISTS display_name text,
    ADD COLUMN IF NOT EXISTS password_hash text,
    ADD COLUMN IF NOT EXISTS is_active boolean NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS is_admin boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW();

ALTER TABLE {schema}.projects
    ADD COLUMN IF NOT EXISTS description text,
    ADD COLUMN IF NOT EXISTS kvstore_root_path text,
    ADD COLUMN IF NOT EXISTS is_active boolean NOT NULL DEFAULT true,
    ADD COLUMN IF NOT EXISTS settings jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW();

UPDATE {schema}.projects AS projects
SET settings = jsonb_set(
    projects.settings,
    '{storage,frame}',
    jsonb_strip_nulls(
        jsonb_build_object(
            'encoding', legacy.frame_storage->'image_encoding',
            'quality', legacy.frame_storage->'image_quality'
        )
    ),
    true
)
FROM (
    SELECT
        id,
        COALESCE(metadata #> '{processing,frame_storage}', metadata->'frame_storage') AS frame_storage
    FROM {schema}.projects
) AS legacy
WHERE projects.id = legacy.id
  AND legacy.frame_storage IS NOT NULL
  AND projects.settings #> '{storage,frame}' IS NULL;

ALTER TABLE {schema}.project_memberships
    ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW();

ALTER TABLE {schema}.user_sessions
    ALTER COLUMN project_id DROP NOT NULL,
    ADD COLUMN IF NOT EXISTS user_agent text,
    ADD COLUMN IF NOT EXISTS remote_addr text,
    ADD COLUMN IF NOT EXISTS metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS revoked_at timestamptz,
    ADD COLUMN IF NOT EXISTS last_seen_at timestamptz NOT NULL DEFAULT NOW();

CREATE TABLE IF NOT EXISTS {schema}.runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    run_key text NOT NULL UNIQUE,
    instrument text NOT NULL,
    source_path text NOT NULL,
    source_type text NOT NULL,
    status text NOT NULL DEFAULT 'registered',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW()
);

ALTER TABLE {schema}.runs
    ADD COLUMN IF NOT EXISTS project_id uuid;

UPDATE {schema}.runs
SET project_id = '00000000-0000-0000-0000-000000000001'::uuid
WHERE project_id IS NULL;

ALTER TABLE {schema}.runs
    ALTER COLUMN project_id DROP DEFAULT,
    ALTER COLUMN project_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'runs_project_id_fkey'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.runs
            ADD CONSTRAINT runs_project_id_fkey
            FOREIGN KEY (project_id) REFERENCES {schema}.projects(id) ON DELETE RESTRICT;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {schema}.raw_assets (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    filename text NOT NULL,
    path text NOT NULL,
    kind {schema}.asset_kind NOT NULL,
    checksum text NOT NULL,
    size_bytes bigint NOT NULL,
    collections text[] NOT NULL DEFAULT ARRAY['none']::text[],
    media_count integer NOT NULL DEFAULT 1,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    CONSTRAINT raw_assets_collections_nonempty CHECK (cardinality(collections) > 0),
    UNIQUE (run_id, filename)
);

ALTER TABLE {schema}.raw_assets
    ADD COLUMN IF NOT EXISTS project_id uuid;

UPDATE {schema}.raw_assets assets
SET project_id = runs.project_id
FROM {schema}.runs runs
WHERE assets.run_id = runs.id
  AND assets.project_id IS NULL;

UPDATE {schema}.raw_assets
SET project_id = '00000000-0000-0000-0000-000000000001'::uuid
WHERE project_id IS NULL;

ALTER TABLE {schema}.raw_assets
    ALTER COLUMN project_id DROP DEFAULT,
    ALTER COLUMN project_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'raw_assets_project_id_fkey'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.raw_assets
            ADD CONSTRAINT raw_assets_project_id_fkey
            FOREIGN KEY (project_id) REFERENCES {schema}.projects(id) ON DELETE RESTRICT;
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = 'raw_assets' AND column_name = 'asset_key'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = 'raw_assets' AND column_name = 'filename'
    ) THEN
        ALTER TABLE {schema}.raw_assets RENAME COLUMN asset_key TO filename;
    END IF;
END $$;

ALTER TABLE {schema}.raw_assets
    ADD COLUMN IF NOT EXISTS filename text;

UPDATE {schema}.raw_assets
SET filename = COALESCE(filename, metadata->>'filename', path)
WHERE filename IS NULL;

ALTER TABLE {schema}.raw_assets
    ALTER COLUMN filename SET NOT NULL;

ALTER TABLE {schema}.raw_assets
    ADD COLUMN IF NOT EXISTS collections text[] NOT NULL DEFAULT ARRAY['none']::text[];

UPDATE {schema}.raw_assets
SET collections = ARRAY['none']::text[]
WHERE collections IS NULL OR cardinality(collections) = 0;

ALTER TABLE {schema}.raw_assets
    ALTER COLUMN collections SET DEFAULT ARRAY['none']::text[],
    ALTER COLUMN collections SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'raw_assets_collections_nonempty'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.raw_assets
            ADD CONSTRAINT raw_assets_collections_nonempty CHECK (cardinality(collections) > 0);
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'raw_assets_run_id_asset_key_key'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.raw_assets DROP CONSTRAINT raw_assets_run_id_asset_key_key;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'raw_assets_run_id_filename_key'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.raw_assets
            ADD CONSTRAINT raw_assets_run_id_filename_key UNIQUE (run_id, filename);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {schema}.frames (
    id uuid PRIMARY KEY DEFAULT {schema}.uuidv7(),
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    asset_id uuid NOT NULL REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    frame_index integer NOT NULL,
    captured_at timestamptz,
    width integer NOT NULL,
    height integer NOT NULL,
    bbox_x integer NOT NULL DEFAULT 0,
    bbox_y integer NOT NULL DEFAULT 0,
    parent_frame_id uuid REFERENCES {schema}.frames(id) ON DELETE SET NULL,
    source_ref text,
    kvstore_hash text NOT NULL,
    preview_thumbhash bytea NOT NULL,
    payload_ref text,
    payload_encoding text,
    payload_format text,
    payload_dtype text,
    payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    preprocessed_kvstore_hash text,
    preprocessed_preview_thumbhash bytea,
    preprocessed_payload_ref text,
    preprocessed_payload_encoding text,
    preprocessed_payload_format text,
    preprocessed_payload_dtype text,
    preprocessed_payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    preprocessed_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    background_kvstore_hash text,
    background_payload_ref text,
    background_payload_encoding text,
    background_payload_format text,
    background_payload_dtype text,
    background_payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    background_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (asset_id, frame_index)
);

ALTER TABLE {schema}.frames
    ADD COLUMN IF NOT EXISTS bbox_x integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS bbox_y integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS parent_frame_id uuid REFERENCES {schema}.frames(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS kvstore_hash text,
    ADD COLUMN IF NOT EXISTS preview_thumbhash bytea,
    ADD COLUMN IF NOT EXISTS payload_ref text,
    ADD COLUMN IF NOT EXISTS payload_encoding text,
    ADD COLUMN IF NOT EXISTS payload_format text,
    ADD COLUMN IF NOT EXISTS payload_dtype text,
    ADD COLUMN IF NOT EXISTS payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS preprocessed_kvstore_hash text,
    ADD COLUMN IF NOT EXISTS preprocessed_preview_thumbhash bytea,
    ADD COLUMN IF NOT EXISTS preprocessed_payload_ref text,
    ADD COLUMN IF NOT EXISTS preprocessed_payload_encoding text,
    ADD COLUMN IF NOT EXISTS preprocessed_payload_format text,
    ADD COLUMN IF NOT EXISTS preprocessed_payload_dtype text,
    ADD COLUMN IF NOT EXISTS preprocessed_payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS preprocessed_metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS background_kvstore_hash text,
    ADD COLUMN IF NOT EXISTS background_payload_ref text,
    ADD COLUMN IF NOT EXISTS background_payload_encoding text,
    ADD COLUMN IF NOT EXISTS background_payload_format text,
    ADD COLUMN IF NOT EXISTS background_payload_dtype text,
    ADD COLUMN IF NOT EXISTS background_payload_shape jsonb NOT NULL DEFAULT '[]'::jsonb,
    ADD COLUMN IF NOT EXISTS background_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

UPDATE {schema}.frames
SET
    bbox_x = COALESCE((metadata->>'bbox_x')::integer, bbox_x, 0),
    bbox_y = COALESCE((metadata->>'bbox_y')::integer, bbox_y, 0),
    kvstore_hash = COALESCE(kvstore_hash, metadata->>'kvstore_key', metadata->>'kvstore_hash'),
    payload_ref = COALESCE(payload_ref, metadata->>'kvstore_key', kvstore_hash),
    payload_encoding = COALESCE(payload_encoding, metadata->>'kvstore_encoding', metadata->>'array_encoding'),
    payload_format = COALESCE(payload_format, metadata->>'kvstore_format'),
    payload_dtype = COALESCE(payload_dtype, metadata->>'dtype'),
    payload_shape = CASE
        WHEN payload_shape = '[]'::jsonb AND metadata ? 'shape' THEN metadata->'shape'
        ELSE payload_shape
    END
WHERE
    kvstore_hash IS NULL
    OR payload_ref IS NULL
    OR payload_encoding IS NULL
    OR payload_format IS NULL
    OR payload_dtype IS NULL
    OR (payload_shape = '[]'::jsonb AND metadata ? 'shape')
    OR (metadata ? 'bbox_x' AND bbox_x IS DISTINCT FROM (metadata->>'bbox_x')::integer)
    OR (metadata ? 'bbox_y' AND bbox_y IS DISTINCT FROM (metadata->>'bbox_y')::integer);

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = 'frames' AND column_name = 'frame_hash'
    ) THEN
        EXECUTE 'UPDATE {schema}.frames SET kvstore_hash = COALESCE(kvstore_hash, frame_hash) WHERE kvstore_hash IS NULL';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = '{schema}' AND table_name = 'frames' AND column_name = 'frame_png'
    ) THEN
        EXECUTE 'UPDATE {schema}.frames SET preview_thumbhash = COALESCE(preview_thumbhash, frame_png) WHERE preview_thumbhash IS NULL';
    END IF;
END $$;

UPDATE {schema}.frames
SET preview_thumbhash = ''::bytea
WHERE preview_thumbhash IS NULL;

ALTER TABLE {schema}.frames
    ALTER COLUMN kvstore_hash SET NOT NULL,
    ALTER COLUMN preview_thumbhash SET NOT NULL;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = 'frames'
          AND column_name = 'id'
          AND udt_name = 'uuid'
    ) THEN
        ALTER TABLE {schema}.frames ALTER COLUMN id SET DEFAULT {schema}.uuidv7();
    END IF;
END $$;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = '{schema}' AND table_name = 'detections'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.tables
        WHERE table_schema = '{schema}' AND table_name = 'detection_candidate'
    ) THEN
        ALTER TABLE {schema}.detections RENAME TO detection_candidate;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {schema}.detection_candidate (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    frame_id uuid NOT NULL REFERENCES {schema}.frames(id) ON DELETE CASCADE,
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

ALTER TABLE {schema}.detection_candidate
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
          AND table_name = 'detection_candidate'
          AND column_name = 'crop_png'
    ) THEN
        EXECUTE 'UPDATE {schema}.detection_candidate SET roi_payload = COALESCE(roi_payload, crop_png) WHERE roi_payload IS NULL';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name = 'detection_candidate'
          AND column_name = 'mask_png'
    ) THEN
        EXECUTE 'UPDATE {schema}.detection_candidate SET mask_payload = COALESCE(mask_payload, mask_png) WHERE mask_payload IS NULL';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'detections_bbox_positive'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.detection_candidate
            ADD CONSTRAINT detections_bbox_positive CHECK (bbox_w > 0 AND bbox_h > 0);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'detections_crop_bbox_positive'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.detection_candidate
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

UPDATE {schema}.detection_candidate
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
    )
WHERE
    roi_encoding IS NULL
    OR roi_format IS NULL
    OR roi_dtype IS NULL
    OR mask_encoding IS NULL
    OR mask_format IS NULL
    OR mask_dtype IS NULL
    OR (roi_shape = '[]'::jsonb AND metadata ? 'shape')
    OR (mask_shape = '[]'::jsonb AND metadata ? 'mask_shape')
    OR (crop_bbox_x IS NULL AND metadata ? 'roi_bbox')
    OR (crop_bbox_y IS NULL AND metadata ? 'roi_bbox')
    OR (crop_bbox_w IS NULL AND metadata ? 'roi_bbox')
    OR (crop_bbox_h IS NULL AND metadata ? 'roi_bbox');

CREATE TABLE IF NOT EXISTS {schema}.detections_refined (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    candidate_detection_id uuid NOT NULL REFERENCES {schema}.detection_candidate(id) ON DELETE CASCADE,
    job_id uuid,
    run_id uuid NOT NULL REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    frame_id uuid NOT NULL REFERENCES {schema}.frames(id) ON DELETE CASCADE,
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
    refinement_method text NOT NULL DEFAULT 'identity',
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

ALTER TABLE {schema}.detections_refined
    ADD COLUMN IF NOT EXISTS job_id uuid;

ALTER TABLE {schema}.detections_refined
    DROP CONSTRAINT IF EXISTS detections_refined_candidate_detection_id_key;

CREATE TABLE IF NOT EXISTS {schema}.models (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    model_key text NOT NULL UNIQUE,
    model_name text NOT NULL,
    version text NOT NULL,
    task text NOT NULL DEFAULT 'classification',
    artifact_uri text,
    labels jsonb NOT NULL DEFAULT '[]'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

ALTER TABLE {schema}.models
    ADD COLUMN IF NOT EXISTS project_id uuid;

UPDATE {schema}.models
SET project_id = '00000000-0000-0000-0000-000000000001'::uuid
WHERE project_id IS NULL;

ALTER TABLE {schema}.models
    ALTER COLUMN project_id DROP DEFAULT,
    ALTER COLUMN project_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'models_project_id_fkey'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.models
            ADD CONSTRAINT models_project_id_fkey
            FOREIGN KEY (project_id) REFERENCES {schema}.projects(id) ON DELETE RESTRICT;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {schema}.classification_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    detection_id uuid NOT NULL REFERENCES {schema}.detection_candidate(id) ON DELETE CASCADE,
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
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
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

ALTER TABLE {schema}.processing_jobs
    ADD COLUMN IF NOT EXISTS project_id uuid;

UPDATE {schema}.processing_jobs jobs
SET project_id = COALESCE(
    (SELECT runs.project_id FROM {schema}.runs runs WHERE runs.id = jobs.run_id),
    (SELECT assets.project_id FROM {schema}.raw_assets assets WHERE assets.id = jobs.asset_id),
    '00000000-0000-0000-0000-000000000001'::uuid
)
WHERE jobs.project_id IS NULL;

ALTER TABLE {schema}.processing_jobs
    ALTER COLUMN project_id DROP DEFAULT,
    ALTER COLUMN project_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'processing_jobs_project_id_fkey'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.processing_jobs
            ADD CONSTRAINT processing_jobs_project_id_fkey
            FOREIGN KEY (project_id) REFERENCES {schema}.projects(id) ON DELETE RESTRICT;
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'detections_refined_job_id_fkey'
          AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.detections_refined
            ADD CONSTRAINT detections_refined_job_id_fkey
            FOREIGN KEY (job_id) REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {schema}.processing_job_dependencies (
    job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    depends_on_job_id uuid NOT NULL REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE,
    PRIMARY KEY (job_id, depends_on_job_id)
);

CREATE TABLE IF NOT EXISTS {schema}.project_processing_status_snapshots (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    session_id uuid REFERENCES {schema}.user_sessions(id) ON DELETE CASCADE,
    status_version bigint NOT NULL DEFAULT 0,
    generated_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    summary jsonb NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE {schema}.project_processing_status_snapshots
    ADD COLUMN IF NOT EXISTS session_id uuid REFERENCES {schema}.user_sessions(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS status_version bigint NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS generated_at timestamptz,
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW(),
    ADD COLUMN IF NOT EXISTS summary jsonb NOT NULL DEFAULT '{}'::jsonb;

CREATE TABLE IF NOT EXISTS {schema}.frame_processing_status (
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    frame_id uuid NOT NULL REFERENCES {schema}.frames(id) ON DELETE CASCADE,
    asset_id uuid NOT NULL REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    run_id uuid REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    frame_index integer NOT NULL,
    collections text[] NOT NULL DEFAULT ARRAY[]::text[],
    preprocessing_status text NOT NULL DEFAULT 'unknown',
    preprocessing_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    preprocessing_completed_at timestamptz,
    candidate_detection_status text NOT NULL DEFAULT 'unknown',
    candidate_detection_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    candidate_detection_completed_at timestamptz,
    candidate_detection_count integer NOT NULL DEFAULT 0,
    roi_refinement_status text NOT NULL DEFAULT 'unknown',
    roi_refinement_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    roi_refinement_completed_at timestamptz,
    refined_detection_count integer NOT NULL DEFAULT 0,
    unrefined_candidate_count integer NOT NULL DEFAULT 0,
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    PRIMARY KEY (project_id, frame_id),
    CONSTRAINT frame_processing_status_preprocessing_known CHECK (
        preprocessing_status IN ('unknown', 'queued', 'leased', 'working', 'succeeded', 'failed', 'cancelled', 'dead_lettered')
    ),
    CONSTRAINT frame_processing_status_candidate_detection_known CHECK (
        candidate_detection_status IN ('unknown', 'queued', 'leased', 'working', 'succeeded', 'failed', 'cancelled', 'dead_lettered')
    ),
    CONSTRAINT frame_processing_status_roi_refinement_known CHECK (
        roi_refinement_status IN ('unknown', 'queued', 'leased', 'working', 'succeeded', 'failed', 'cancelled', 'dead_lettered')
    )
);

ALTER TABLE {schema}.frame_processing_status
    ADD COLUMN IF NOT EXISTS run_id uuid REFERENCES {schema}.runs(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS frame_index integer,
    ADD COLUMN IF NOT EXISTS collections text[] NOT NULL DEFAULT ARRAY[]::text[],
    ADD COLUMN IF NOT EXISTS preprocessing_status text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS preprocessing_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS preprocessing_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS candidate_detection_status text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS candidate_detection_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS candidate_detection_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS candidate_detection_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS roi_refinement_status text NOT NULL DEFAULT 'unknown',
    ADD COLUMN IF NOT EXISTS roi_refinement_job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    ADD COLUMN IF NOT EXISTS roi_refinement_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS refined_detection_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS unrefined_candidate_count integer NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS updated_at timestamptz NOT NULL DEFAULT NOW();

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

CREATE TABLE IF NOT EXISTS {schema}.logs (
    id bigserial PRIMARY KEY,
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE RESTRICT,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    level text NOT NULL DEFAULT 'info',
    logger text NOT NULL DEFAULT 'pelagia',
    event_type text NOT NULL,
    message text,
    run_id uuid REFERENCES {schema}.runs(id) ON DELETE SET NULL,
    asset_id uuid REFERENCES {schema}.raw_assets(id) ON DELETE SET NULL,
    job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    worker_id text,
    request_id text,
    duration_ms double precision,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE {schema}.logs
    ADD COLUMN IF NOT EXISTS project_id uuid;

UPDATE {schema}.logs logs
SET project_id = COALESCE(
    (SELECT runs.project_id FROM {schema}.runs runs WHERE runs.id = logs.run_id),
    (SELECT assets.project_id FROM {schema}.raw_assets assets WHERE assets.id = logs.asset_id),
    (SELECT jobs.project_id FROM {schema}.processing_jobs jobs WHERE jobs.id = logs.job_id),
    '00000000-0000-0000-0000-000000000001'::uuid
)
WHERE logs.project_id IS NULL;

ALTER TABLE {schema}.logs
    ALTER COLUMN project_id SET NOT NULL;

ALTER TABLE {schema}.logs
    DROP CONSTRAINT IF EXISTS logs_project_id_fkey;

ALTER TABLE {schema}.logs
    ADD CONSTRAINT logs_project_id_fkey
    FOREIGN KEY (project_id) REFERENCES {schema}.projects(id) ON DELETE RESTRICT;

CREATE INDEX IF NOT EXISTS idx_{schema}_users_username ON {schema}.users (username);
CREATE INDEX IF NOT EXISTS idx_{schema}_projects_project_key ON {schema}.projects (project_key);
CREATE INDEX IF NOT EXISTS idx_{schema}_project_memberships_project_id ON {schema}.project_memberships (project_id, role);
CREATE INDEX IF NOT EXISTS idx_{schema}_user_sessions_user_id ON {schema}.user_sessions (user_id, expires_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_user_sessions_project_id ON {schema}.user_sessions (project_id, expires_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_raw_assets_run_id ON {schema}.raw_assets (run_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_runs_project_id ON {schema}.runs (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_raw_assets_project_id ON {schema}.raw_assets (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_raw_assets_collections ON {schema}.raw_assets USING gin (collections);
CREATE INDEX IF NOT EXISTS idx_{schema}_frames_asset_id ON {schema}.frames (asset_id, frame_index);
CREATE INDEX IF NOT EXISTS idx_{schema}_detection_candidate_frame_id ON {schema}.detection_candidate (frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_detections_refined_candidate_id ON {schema}.detections_refined (candidate_detection_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_detections_refined_job_id ON {schema}.detections_refined (job_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_detections_refined_frame_id ON {schema}.detections_refined (frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_classification_results_detection_id ON {schema}.classification_results (detection_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_models_project_id ON {schema}.models (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_project_id ON {schema}.processing_jobs (project_id, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_status ON {schema}.processing_jobs (status, stage, priority, created_at);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_run_id ON {schema}.processing_jobs (run_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_stage_status_updated ON {schema}.processing_jobs (stage, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_updated ON {schema}.processing_jobs (updated_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_jobs_asset_stage ON {schema}.processing_jobs (asset_id, stage, status, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_processing_job_dependencies_depends_on ON {schema}.processing_job_dependencies (depends_on_job_id);
CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_project_processing_status_snapshots_session
    ON {schema}.project_processing_status_snapshots (project_id, session_id)
    WHERE session_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_project_processing_status_snapshots_project
    ON {schema}.project_processing_status_snapshots (project_id)
    WHERE session_id IS NULL;
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_asset_frame
    ON {schema}.frame_processing_status (project_id, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_preprocessing
    ON {schema}.frame_processing_status (project_id, preprocessing_status, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_candidate_detection
    ON {schema}.frame_processing_status (project_id, candidate_detection_status, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_roi_refinement
    ON {schema}.frame_processing_status (project_id, roi_refinement_status, asset_id, frame_index, frame_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_updated
    ON {schema}.frame_processing_status (project_id, updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_frame_processing_status_collections
    ON {schema}.frame_processing_status USING gin (collections);
CREATE INDEX IF NOT EXISTS idx_{schema}_job_events_job_id ON {schema}.job_events (job_id, id);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_created_at ON {schema}.logs (created_at DESC, id DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_project_id ON {schema}.logs (project_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_event_type ON {schema}.logs (event_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_level ON {schema}.logs (level, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_run_id ON {schema}.logs (run_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_job_id ON {schema}.logs (job_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_worker_id ON {schema}.logs (worker_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_{schema}_logs_request_id ON {schema}.logs (request_id, created_at DESC);

DROP TRIGGER IF EXISTS trg_runs_updated_at ON {schema}.runs;
CREATE TRIGGER trg_runs_updated_at
BEFORE UPDATE ON {schema}.runs
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

DROP TRIGGER IF EXISTS trg_users_updated_at ON {schema}.users;
CREATE TRIGGER trg_users_updated_at
BEFORE UPDATE ON {schema}.users
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

DROP TRIGGER IF EXISTS trg_projects_updated_at ON {schema}.projects;
CREATE TRIGGER trg_projects_updated_at
BEFORE UPDATE ON {schema}.projects
FOR EACH ROW
EXECUTE FUNCTION {schema}.set_updated_at();

DROP TRIGGER IF EXISTS trg_project_memberships_updated_at ON {schema}.project_memberships;
CREATE TRIGGER trg_project_memberships_updated_at
BEFORE UPDATE ON {schema}.project_memberships
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
