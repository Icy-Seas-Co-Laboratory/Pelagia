-- Add centroid and major-axis orientation measurements to candidate and refined ROIs.
ALTER TABLE {schema}.detection_candidate
    ADD COLUMN IF NOT EXISTS centroid_x double precision,
    ADD COLUMN IF NOT EXISTS centroid_y double precision,
    ADD COLUMN IF NOT EXISTS major_axis_center_x double precision,
    ADD COLUMN IF NOT EXISTS major_axis_center_y double precision,
    ADD COLUMN IF NOT EXISTS major_axis_orientation_degrees double precision,
    ADD COLUMN IF NOT EXISTS major_axis_direction_degrees double precision;

ALTER TABLE {schema}.detections_refined
    ADD COLUMN IF NOT EXISTS centroid_x double precision,
    ADD COLUMN IF NOT EXISTS centroid_y double precision,
    ADD COLUMN IF NOT EXISTS major_axis_center_x double precision,
    ADD COLUMN IF NOT EXISTS major_axis_center_y double precision,
    ADD COLUMN IF NOT EXISTS major_axis_orientation_degrees double precision,
    ADD COLUMN IF NOT EXISTS major_axis_direction_degrees double precision;
