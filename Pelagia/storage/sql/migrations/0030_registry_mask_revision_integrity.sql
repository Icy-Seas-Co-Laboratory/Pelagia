-- A Registry ROI can expose one active editable mask revision.  Candidate masks
-- remain on registry_items and are immutable input assets.
CREATE UNIQUE INDEX IF NOT EXISTS idx_{schema}_registry_mask_annotation_current
    ON {schema}.registry_mask_annotations(workspace_id, item_id)
    WHERE is_current;
