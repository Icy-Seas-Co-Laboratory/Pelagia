-- PostgreSQL workspaces for portable Oracle Dataset contract revisions.

ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'registry_load';
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'registry_export';

CREATE TABLE IF NOT EXISTS {schema}.registry_workspaces (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    owner_username text NOT NULL,
    dataset_id text NOT NULL,
    revision_id text NOT NULL,
    parent_revision_id text,
    dataset_type text NOT NULL CHECK (dataset_type IN ('classification', 'mask_refinement')),
    name text NOT NULL,
    title text,
    description text,
    dataset_version text,
    dataset_lifecycle text NOT NULL,
    contract_schema_name text NOT NULL,
    contract_schema_version text NOT NULL,
    source_path text NOT NULL,
    source_sha256 text NOT NULL,
    source_size_bytes bigint NOT NULL,
    status text NOT NULL DEFAULT 'loaded'
        CHECK (status IN ('loading', 'loaded', 'exporting', 'failed', 'purged')),
    is_active boolean NOT NULL DEFAULT true,
    dirty_at timestamptz,
    loaded_at timestamptz NOT NULL DEFAULT NOW(),
    exported_at timestamptz,
    last_export_path text,
    last_export_sha256 text,
    last_export_operation_id text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (project_id, owner_username, dataset_id, revision_id)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_registry_workspaces_active
    ON {schema}.registry_workspaces(project_id, owner_username, is_active, loaded_at DESC);
CREATE UNIQUE INDEX IF NOT EXISTS uq_{schema}_registry_one_active_workspace
    ON {schema}.registry_workspaces(project_id, owner_username) WHERE is_active;

CREATE TABLE IF NOT EXISTS {schema}.registry_assets (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    asset_id text NOT NULL,
    content_sha256 text NOT NULL,
    payload bytea,
    external_uri text,
    encoding text NOT NULL,
    media_type text,
    shape jsonb,
    dtype text,
    original_filename text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at text NOT NULL,
    PRIMARY KEY (workspace_id, asset_id)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_items (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    item_id text NOT NULL,
    ordinal bigint NOT NULL,
    sample_weight double precision,
    source_key text,
    image_asset_id text NOT NULL,
    candidate_mask_asset_id text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at text NOT NULL,
    updated_at text NOT NULL,
    PRIMARY KEY (workspace_id, item_id),
    UNIQUE (workspace_id, source_key)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_labels (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    label_id text NOT NULL,
    origin text NOT NULL CHECK (origin IN ('classification', 'workspace')),
    class_index integer,
    name text NOT NULL,
    display_name text,
    parent_label_id text,
    rank text,
    description text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at text,
    deprecated_at text,
    PRIMARY KEY (workspace_id, label_id),
    UNIQUE (workspace_id, origin, name)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_annotations (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    annotation_id text NOT NULL,
    item_id text NOT NULL,
    label_id text NOT NULL,
    origin text NOT NULL CHECK (origin IN ('classification', 'workspace')),
    created_at text NOT NULL,
    annotator text,
    method text,
    confidence double precision,
    status text NOT NULL,
    is_current boolean NOT NULL,
    parent_annotation_id text,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    notes text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (workspace_id, annotation_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_registry_annotation_current
    ON {schema}.registry_annotations(workspace_id, item_id, origin) WHERE is_current;

CREATE TABLE IF NOT EXISTS {schema}.registry_reviews (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    review_id text NOT NULL,
    annotation_id text NOT NULL,
    origin text NOT NULL CHECK (origin IN ('classification', 'workspace')),
    reviewer text NOT NULL,
    decision text NOT NULL CHECK (decision IN ('verified', 'rejected', 'needs_review')),
    created_at text NOT NULL,
    notes text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (workspace_id, review_id)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_descriptors (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    descriptor_id text NOT NULL,
    scope text NOT NULL CHECK (scope IN ('target', 'image')),
    name text NOT NULL,
    parent_descriptor_id text,
    concept_id text,
    concept_type text,
    selectable boolean NOT NULL,
    exclusive_within_parent boolean NOT NULL,
    preferred boolean NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at text NOT NULL,
    deprecated_at text,
    PRIMARY KEY (workspace_id, descriptor_id),
    UNIQUE (workspace_id, scope, name)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_descriptor_annotations (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    annotation_id text NOT NULL,
    item_id text NOT NULL,
    descriptor_id text NOT NULL,
    created_at text NOT NULL,
    annotator text,
    status text NOT NULL,
    is_current boolean NOT NULL,
    parent_annotation_id text,
    notes text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (workspace_id, annotation_id)
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_registry_descriptor_current
    ON {schema}.registry_descriptor_annotations(workspace_id, item_id, descriptor_id) WHERE is_current;

CREATE TABLE IF NOT EXISTS {schema}.registry_mask_annotations (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    annotation_id text NOT NULL,
    item_id text NOT NULL,
    mask_asset_id text NOT NULL,
    created_at text NOT NULL,
    annotator text,
    method text,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    validation jsonb NOT NULL DEFAULT '{}'::jsonb,
    status text NOT NULL,
    is_current boolean NOT NULL,
    parent_annotation_id text,
    notes text,
    PRIMARY KEY (workspace_id, annotation_id)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_inference_runs (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    inference_run_id text NOT NULL,
    dataset_fingerprint_sha256 text NOT NULL,
    model_artifact_id text,
    model_run_id text,
    model_artifact_fingerprint_sha256 text,
    name text,
    status text NOT NULL,
    created_at text NOT NULL,
    completed_at text,
    input_contract jsonb NOT NULL DEFAULT '{}'::jsonb,
    parameters jsonb NOT NULL DEFAULT '{}'::jsonb,
    software_environment jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (workspace_id, inference_run_id)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_evidence_arrays (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    array_id text NOT NULL,
    content_sha256 text NOT NULL,
    payload bytea NOT NULL,
    encoding text NOT NULL,
    media_type text NOT NULL,
    shape jsonb NOT NULL,
    dtype text NOT NULL,
    created_at text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (workspace_id, array_id),
    UNIQUE (workspace_id, content_sha256)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_model_evidence (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    evidence_id text NOT NULL,
    inference_run_id text NOT NULL,
    item_id text NOT NULL,
    predicted_label_id text,
    prediction_confidence double precision,
    nearest_neighbor_similarity double precision,
    top_k_label_agreement double precision,
    weighted_label_support double precision,
    label_margin double precision,
    logits_array_id text,
    embedding_array_id text,
    output_array_id text,
    packet jsonb NOT NULL DEFAULT '{}'::jsonb,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at text NOT NULL,
    PRIMARY KEY (workspace_id, evidence_id),
    UNIQUE (workspace_id, inference_run_id, item_id)
);

CREATE TABLE IF NOT EXISTS {schema}.registry_dataset_events (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    event_id text NOT NULL,
    revision_id text NOT NULL,
    event_type text NOT NULL,
    created_at text NOT NULL,
    actor text,
    details jsonb NOT NULL DEFAULT '{}'::jsonb,
    PRIMARY KEY (workspace_id, event_id)
);

-- Lossless storage for contract relations that Registry does not edit directly.
-- Keeping these rows avoids dropping provenance/taxonomy material on export while
-- allowing dedicated relational tables to be introduced later when Pelagia needs
-- to query them.
CREATE TABLE IF NOT EXISTS {schema}.registry_contract_records (
    workspace_id uuid NOT NULL REFERENCES {schema}.registry_workspaces(id) ON DELETE CASCADE,
    relation text NOT NULL,
    ordinal bigint NOT NULL,
    record jsonb NOT NULL,
    PRIMARY KEY (workspace_id, relation, ordinal)
);
