-- Keep project-scoped foreign-key checks and catalog lookups bounded as telemetry grows.

CREATE INDEX IF NOT EXISTS idx_{schema}_telemetry_streams_sensor_project
    ON {schema}.telemetry_streams(sensor_id, project_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_telemetry_streams_parameter_project
    ON {schema}.telemetry_streams(parameter_id, project_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_telemetry_streams_source_project_run
    ON {schema}.telemetry_streams(source_id, project_id, run_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_timeline_events_type_project
    ON {schema}.timeline_events(event_type_id, project_id);
CREATE INDEX IF NOT EXISTS idx_{schema}_timeline_events_source_project_run
    ON {schema}.timeline_events(source_id, project_id, run_id);
