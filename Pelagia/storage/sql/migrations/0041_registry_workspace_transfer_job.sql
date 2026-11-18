-- Direct workspace copy/move jobs use the same project-scoped worker queue.
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'registry_workspace_transfer';
