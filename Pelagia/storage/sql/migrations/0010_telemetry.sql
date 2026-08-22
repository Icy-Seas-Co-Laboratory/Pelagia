-- Project-scoped telemetry, native-resolution observations, and timeline events.

ALTER TYPE {schema}.asset_kind ADD VALUE IF NOT EXISTS 'telemetry';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'runs_id_project_id_key' AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.runs ADD CONSTRAINT runs_id_project_id_key UNIQUE (id, project_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'raw_assets_id_project_id_run_id_key' AND connamespace = '{schema}'::regnamespace
    ) THEN
        ALTER TABLE {schema}.raw_assets
            ADD CONSTRAINT raw_assets_id_project_id_run_id_key UNIQUE (id, project_id, run_id);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS {schema}.telemetry_sources (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    run_id uuid NOT NULL,
    raw_asset_id uuid NOT NULL UNIQUE,
    format text NOT NULL,
    parser_name text NOT NULL,
    parser_version text NOT NULL,
    import_status text NOT NULL DEFAULT 'importing'
        CHECK (import_status IN ('importing', 'ready', 'failed')),
    observed_start_at timestamptz,
    observed_end_at timestamptz,
    observation_count bigint NOT NULL DEFAULT 0 CHECK (observation_count >= 0),
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    imported_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (id, project_id, run_id),
    FOREIGN KEY (run_id, project_id)
        REFERENCES {schema}.runs(id, project_id) ON DELETE CASCADE,
    FOREIGN KEY (raw_asset_id, project_id, run_id)
        REFERENCES {schema}.raw_assets(id, project_id, run_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_{schema}_telemetry_sources_project_run
    ON {schema}.telemetry_sources(project_id, run_id, created_at DESC);

CREATE TABLE IF NOT EXISTS {schema}.telemetry_sensors (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    sensor_key text NOT NULL,
    display_name text,
    manufacturer text,
    model text,
    serial_number text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, sensor_key),
    UNIQUE (id, project_id)
);

CREATE TABLE IF NOT EXISTS {schema}.telemetry_parameters (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    parameter_key text NOT NULL,
    display_name text,
    definition text,
    standard_name text,
    canonical_unit text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, parameter_key),
    UNIQUE (id, project_id)
);

CREATE TABLE IF NOT EXISTS {schema}.telemetry_streams (
    id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    public_id uuid NOT NULL DEFAULT gen_random_uuid() UNIQUE,
    project_id uuid NOT NULL,
    run_id uuid NOT NULL,
    source_id uuid NOT NULL,
    sensor_id uuid NOT NULL,
    parameter_id uuid NOT NULL,
    stream_key text NOT NULL,
    native_unit text NOT NULL,
    sampling_rate_hz double precision CHECK (sampling_rate_hz IS NULL OR sampling_rate_hz > 0),
    interpolation text NOT NULL DEFAULT 'none'
        CHECK (interpolation IN ('linear', 'nearest', 'previous', 'none')),
    max_gap interval CHECK (max_gap IS NULL OR max_gap > interval '0 seconds'),
    priority integer NOT NULL DEFAULT 100,
    is_default boolean NOT NULL DEFAULT false,
    qc_scheme text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    CONSTRAINT telemetry_streams_interpolation_gap CHECK (
        interpolation = 'none' OR max_gap IS NOT NULL
    ),
    FOREIGN KEY (source_id, project_id, run_id)
        REFERENCES {schema}.telemetry_sources(id, project_id, run_id) ON DELETE CASCADE,
    FOREIGN KEY (sensor_id, project_id)
        REFERENCES {schema}.telemetry_sensors(id, project_id) ON DELETE RESTRICT,
    FOREIGN KEY (parameter_id, project_id)
        REFERENCES {schema}.telemetry_parameters(id, project_id) ON DELETE RESTRICT,
    UNIQUE (source_id, stream_key),
    UNIQUE (id, project_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_telemetry_streams_default_parameter
    ON {schema}.telemetry_streams(run_id, parameter_id) WHERE is_default;
CREATE INDEX IF NOT EXISTS idx_{schema}_telemetry_streams_project_run
    ON {schema}.telemetry_streams(project_id, run_id, priority, stream_key);

CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations (
    stream_id bigint NOT NULL REFERENCES {schema}.telemetry_streams(id) ON DELETE CASCADE,
    observed_at timestamptz NOT NULL,
    value double precision NOT NULL CHECK (
        value > '-Infinity'::double precision AND value < 'Infinity'::double precision
    ),
    qc_flag smallint,
    PRIMARY KEY (stream_id, observed_at)
) PARTITION BY HASH (stream_id);

CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p0 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 0);
CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p1 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 1);
CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p2 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 2);
CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p3 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 3);
CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p4 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 4);
CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p5 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 5);
CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p6 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 6);
CREATE TABLE IF NOT EXISTS {schema}.telemetry_observations_p7 PARTITION OF {schema}.telemetry_observations FOR VALUES WITH (MODULUS 8, REMAINDER 7);

CREATE TABLE IF NOT EXISTS {schema}.timeline_event_types (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL REFERENCES {schema}.projects(id) ON DELETE CASCADE,
    event_type_key text NOT NULL,
    display_name text,
    description text,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    UNIQUE (project_id, event_type_key),
    UNIQUE (id, project_id)
);

CREATE TABLE IF NOT EXISTS {schema}.timeline_events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id uuid NOT NULL,
    run_id uuid NOT NULL,
    event_type_id uuid NOT NULL,
    source_id uuid,
    start_at timestamptz NOT NULL,
    end_at timestamptz,
    value text,
    created_by uuid REFERENCES {schema}.users(id) ON DELETE SET NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    created_at timestamptz NOT NULL DEFAULT NOW(),
    updated_at timestamptz NOT NULL DEFAULT NOW(),
    FOREIGN KEY (event_type_id, project_id)
        REFERENCES {schema}.timeline_event_types(id, project_id) ON DELETE RESTRICT,
    FOREIGN KEY (run_id, project_id)
        REFERENCES {schema}.runs(id, project_id) ON DELETE CASCADE,
    FOREIGN KEY (source_id, project_id, run_id)
        REFERENCES {schema}.telemetry_sources(id, project_id, run_id) ON DELETE RESTRICT,
    CONSTRAINT timeline_events_time_order CHECK (end_at IS NULL OR end_at >= start_at)
);

CREATE INDEX IF NOT EXISTS idx_{schema}_timeline_events_run_time
    ON {schema}.timeline_events(project_id, run_id, start_at, end_at);
