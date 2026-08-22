-- Index inclusive timeline intervals for bounded overlap and point lookups.

-- Migrations run in one transaction, so CREATE INDEX CONCURRENTLY is not valid
-- here. This expression index is additive and avoids both a table rewrite and a
-- btree_gist extension requirement; the existing project/run btree index remains
-- available for scoped filtering and PostgreSQL may combine the two indexes.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM {schema}.timeline_events
        WHERE end_at IS NOT NULL AND end_at < start_at
    ) THEN
        RAISE EXCEPTION
            'Cannot index timeline intervals: %.timeline_events contains end_at values before start_at.',
            '{schema}';
    END IF;
END $$;

-- Put the distinguishing part first so PostgreSQL's 63-byte identifier limit
-- cannot collapse this into another timeline index for a long schema name.
CREATE INDEX IF NOT EXISTS idx_tl_period_{schema}
    ON {schema}.timeline_events USING gist (
        tstzrange(start_at, COALESCE(end_at, start_at), '[]')
    );
