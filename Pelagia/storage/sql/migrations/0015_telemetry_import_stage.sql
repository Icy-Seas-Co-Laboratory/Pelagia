-- Add the worker stage used for queued telemetry imports.
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'telemetry_import';
