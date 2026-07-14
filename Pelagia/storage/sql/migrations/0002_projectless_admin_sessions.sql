-- Allows user admins to authenticate before the first project is created.

ALTER TABLE {schema}.user_sessions
    ALTER COLUMN project_id DROP NOT NULL;
