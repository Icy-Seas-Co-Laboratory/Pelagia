-- Canonical Registry ROI crop and object bounding-box geometry.

ALTER TABLE {schema}.registry_items
    ADD COLUMN IF NOT EXISTS coordinate_space text,
    ADD COLUMN IF NOT EXISTS bbox_x integer,
    ADD COLUMN IF NOT EXISTS bbox_y integer,
    ADD COLUMN IF NOT EXISTS bbox_w integer,
    ADD COLUMN IF NOT EXISTS bbox_h integer,
    ADD COLUMN IF NOT EXISTS crop_bbox_x integer,
    ADD COLUMN IF NOT EXISTS crop_bbox_y integer,
    ADD COLUMN IF NOT EXISTS crop_bbox_w integer,
    ADD COLUMN IF NOT EXISTS crop_bbox_h integer,
    ADD COLUMN IF NOT EXISTS spatial_metadata jsonb NOT NULL DEFAULT '{}'::jsonb;

-- Older loaded workspaces did not have canonical geometry. Their stored ROI image
-- extent is the compatible crop, so the object bbox intentionally defaults to it.
UPDATE {schema}.registry_items item
SET coordinate_space = COALESCE(item.coordinate_space, 'image_pixels'),
    bbox_x = COALESCE(item.bbox_x, 0),
    bbox_y = COALESCE(item.bbox_y, 0),
    bbox_w = COALESCE(item.bbox_w, NULLIF(asset.shape->>1, '')::integer),
    bbox_h = COALESCE(item.bbox_h, NULLIF(asset.shape->>0, '')::integer),
    crop_bbox_x = COALESCE(item.crop_bbox_x, 0),
    crop_bbox_y = COALESCE(item.crop_bbox_y, 0),
    crop_bbox_w = COALESCE(item.crop_bbox_w, NULLIF(asset.shape->>1, '')::integer),
    crop_bbox_h = COALESCE(item.crop_bbox_h, NULLIF(asset.shape->>0, '')::integer),
    spatial_metadata = item.spatial_metadata || jsonb_build_object(
        'normalization', 'pelagia.registry_item_geometry.v1',
        'fallback', 'bbox_and_crop_from_image_extent'
    )
FROM {schema}.registry_assets asset
WHERE asset.workspace_id=item.workspace_id
  AND asset.asset_id=item.image_asset_id
  AND (item.bbox_w IS NULL OR item.bbox_h IS NULL
       OR item.crop_bbox_w IS NULL OR item.crop_bbox_h IS NULL);

ALTER TABLE {schema}.registry_items
    DROP CONSTRAINT IF EXISTS registry_items_geometry_positive,
    ADD CONSTRAINT registry_items_geometry_positive CHECK (
        (bbox_x IS NULL AND bbox_y IS NULL AND bbox_w IS NULL AND bbox_h IS NULL
         AND crop_bbox_x IS NULL AND crop_bbox_y IS NULL
         AND crop_bbox_w IS NULL AND crop_bbox_h IS NULL)
        OR
        (bbox_x IS NOT NULL AND bbox_y IS NOT NULL AND bbox_w > 0 AND bbox_h > 0
         AND crop_bbox_x IS NOT NULL AND crop_bbox_y IS NOT NULL
         AND crop_bbox_w > 0 AND crop_bbox_h > 0)
    ),
    DROP CONSTRAINT IF EXISTS registry_items_bbox_within_crop,
    ADD CONSTRAINT registry_items_bbox_within_crop CHECK (
        bbox_x IS NULL OR (
            bbox_x >= crop_bbox_x AND bbox_y >= crop_bbox_y
            AND bbox_x + bbox_w <= crop_bbox_x + crop_bbox_w
            AND bbox_y + bbox_h <= crop_bbox_y + crop_bbox_h
        )
    );
