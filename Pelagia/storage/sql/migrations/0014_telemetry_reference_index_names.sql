-- Ensure the telemetry reference indexes exist even when a schema name is
-- long enough for PostgreSQL's 63-byte identifier limit to truncate the
-- names emitted by the original reference-index migration.

CREATE INDEX IF NOT EXISTS telemetry_streams_sensor_project_idx
    ON {schema}.telemetry_streams(sensor_id, project_id);
CREATE INDEX IF NOT EXISTS telemetry_streams_parameter_project_idx
    ON {schema}.telemetry_streams(parameter_id, project_id);
CREATE INDEX IF NOT EXISTS telemetry_streams_source_project_run_idx
    ON {schema}.telemetry_streams(source_id, project_id, run_id);
CREATE INDEX IF NOT EXISTS timeline_events_type_project_idx
    ON {schema}.timeline_events(event_type_id, project_id);
CREATE INDEX IF NOT EXISTS timeline_events_source_project_run_idx
    ON {schema}.timeline_events(source_id, project_id, run_id);
