-- Generate portable Registry datasets from Pelagia curation selections.
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'registry_generate';
