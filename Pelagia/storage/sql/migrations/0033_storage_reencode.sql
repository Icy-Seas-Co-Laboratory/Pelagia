-- Project-scoped, restartable storage-policy rewrites use the durable job queue.
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'storage_reencode';
