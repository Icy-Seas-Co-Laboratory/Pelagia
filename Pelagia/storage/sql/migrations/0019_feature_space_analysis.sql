-- Ephemeral UMAP/HDBSCAN analyses use the ordinary queue for inspectable
-- progress and cancellation, but their results are cache entries, not data products.
ALTER TYPE {schema}.stage_name ADD VALUE IF NOT EXISTS 'feature_space_analysis';
