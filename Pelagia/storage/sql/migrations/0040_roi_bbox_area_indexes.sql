-- Bounding-box area is the queue and browsing size metric (px²).  Keep the
-- expression identical to the storage predicates so PostgreSQL can use these
-- indexes for range gates and area-first scans.  The id provides a stable
-- order for equal-area ROIs.
CREATE INDEX IF NOT EXISTS idx_{schema}_dc_area
    ON {schema}.detection_candidate
    ((bbox_w::double precision * bbox_h), id);

CREATE INDEX IF NOT EXISTS idx_{schema}_dr_area
    ON {schema}.detections_refined
    ((bbox_w::double precision * bbox_h), id);
