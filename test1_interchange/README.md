# test1

This directory is a self-contained Scientific Image Interchange 1.0 dataset. Its SQLite files under `data/` are the authoritative retained encoded images and frame-level provenance. Treat finalized shards as immutable archival artifacts.

## Layout

- `manifest.json`: physical inventory and shard index
- `metadata.toml`: human-editable scientific and collection metadata
- `history.jsonl`: append-only processing provenance (one JSON event per line)
- `checksums.sha256`: SHA-256 checksums for every finalized package file except itself
- `data/*.sqlite`: standalone SQLite image shards
- `calibration/`: ordinary calibration resources
- `preview/`: non-authoritative derivatives
- `tools/`: standard-library inspection, extraction, and verification scripts

Each shard contains `frames`, `source_files`, `storage_formats`, and `shard_metadata`. The exact encoded image bytes are stored in `frames.blob`; ordinary extraction copies them without decoding or re-encoding. A null BLOB is an explicit missing/failed/removed record, never an implicit sequence collapse.

Inspect manually with `sqlite3 data/SHARD.sqlite '.tables'` or run:

```bash
python tools/inspect.py .
python tools/inspect.py . --json
python tools/extract.py . --frame 12345 --output extracted/one
python tools/extract.py . --camera CAMERA --frames 1000:2000 --output extracted
python tools/verify.py . --level full
python tools/verify.py . --level archival --json
```

Frame and timestamp ranges are inclusive. JPEG and PNG payloads are written with conventional extensions; unknown representations use `.bin`. Null-payload status records are skipped rather than converted into invented images. The extractor also accepts one shard directly: `python tools/extract.py data/SHARD.sqlite --all --output extracted`.

Verification levels become progressively more expensive: `quick` checks the package inventory and file hashes; `structural` also checks every SQLite database and its declared structure; `full` streams all frame records and hashes; `archival` adds strict source-count and completion-provenance requirements. An archival success is a mechanical integrity result, not authorization to delete original acquisitions. Retention policy, backups, scientific acceptance, and responsible-person approval remain separate decisions. These tools never delete source data.

An interrupted writer leaves `data/*.sqlite.partial` files unlisted by the manifest. With the package installed, list them using `pii shards . --partials` or move them intact to a recovery directory using `pii shards . --quarantine-partials RECOVERY_DIR`. They are never deleted automatically.

Hashes always have an algorithm and semantic target. Frame hashes cover exact stored BLOB bytes; shard hashes cover complete finalized SQLite files. The checksum file covers every regular finalized package file except itself, so there is no self-reference. Editing the manifest, metadata, or history therefore requires checksum regeneration. Manifest shard hashes remain authoritative for shard-file integrity.

Timestamp values do not imply accuracy: provenance fields record source, clock, timezone/UTC conversion, precision, synchronization, offset, drift, and interpolation where known. Scientific interpretation belongs in `metadata.toml`; processing actions belong in `history.jsonl`.

The normative reference is **Scientific Image Interchange Format 1.0**, maintained with the `pelagia_interchange` source at `docs/interchange-specification.md`. These files use ordinary JSON, TOML, JSON Lines, SHA-256 text, and SQLite so they remain accessible without Pelagia.
