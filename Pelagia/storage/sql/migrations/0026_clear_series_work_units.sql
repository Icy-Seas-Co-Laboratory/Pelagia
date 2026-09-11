-- Clearing a terminal job must also clear its series work-unit association.
-- The association is only an orchestration record; retaining it blocks the
-- operator-facing job-clear workflow with a foreign-key violation.
ALTER TABLE {schema}.processing_work_units
    DROP CONSTRAINT IF EXISTS processing_work_units_job_id_fkey;
ALTER TABLE {schema}.processing_work_units
    ADD CONSTRAINT processing_work_units_job_id_fkey
    FOREIGN KEY (job_id) REFERENCES {schema}.processing_jobs(id) ON DELETE CASCADE;
