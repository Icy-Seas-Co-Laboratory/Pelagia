-- Logical, auditable line-scan assemblies. Raster ROI segments remain frame-local.
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'roi_continuity';

CREATE TABLE IF NOT EXISTS {schema}.roi_continuity_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    job_id uuid REFERENCES {schema}.processing_jobs(id) ON DELETE SET NULL,
    run_id uuid REFERENCES {schema}.runs(id) ON DELETE SET NULL,
    asset_id uuid NOT NULL REFERENCES {schema}.raw_assets(id) ON DELETE CASCADE,
    method text NOT NULL,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS {schema}.roi_continuity_assemblies (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    continuity_run_id uuid NOT NULL REFERENCES {schema}.roi_continuity_runs(id) ON DELETE CASCADE,
    assembly_key text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (continuity_run_id, assembly_key)
);

CREATE TABLE IF NOT EXISTS {schema}.roi_continuity_links (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    continuity_run_id uuid NOT NULL REFERENCES {schema}.roi_continuity_runs(id) ON DELETE CASCADE,
    assembly_id uuid REFERENCES {schema}.roi_continuity_assemblies(id) ON DELETE SET NULL,
    source_refined_detection_id uuid NOT NULL REFERENCES {schema}.detections_refined(id) ON DELETE CASCADE,
    target_refined_detection_id uuid NOT NULL REFERENCES {schema}.detections_refined(id) ON DELETE CASCADE,
    score double precision NOT NULL,
    accepted boolean NOT NULL,
    reason text,
    features jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (continuity_run_id, source_refined_detection_id, target_refined_detection_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_roi_continuity_links_source
    ON {schema}.roi_continuity_links(source_refined_detection_id, created_at DESC);
