-- Register complete pelagia_interchange packages as first-class raw assets.

ALTER TYPE {schema}.asset_kind ADD VALUE IF NOT EXISTS 'interchange';
