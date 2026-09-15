# Export bundles

Pelagia creates exports asynchronously. Submit a project-scoped request to
`POST /exports`, inspect it with `GET /exports/{export_id}`, and download the
completed ZIP from `GET /exports/{export_id}/download`.

Use `POST /exports/estimate` with the same request body before submitting a
large job. It returns resolved counts and conservative file/byte estimates as
planning information. Export size never rejects a request or publication:
`POST /exports` only persists and queues the request, and the worker performs
the potentially expensive snapshot before writing. XLSX output rolls over into
numbered worksheets when it exceeds Excel's per-sheet row capacity.

Every archive has one UUID root directory and includes `manifest.json`,
`versions.json`, `export.log`, `checksums.sha256`, `README.md`, and a data
dictionary. Product files use the same asset, frame, ROI, run, and telemetry
source UUIDs that appear in their tables and sidecars, so products can be
joined without relying on filenames.

At worker start Pelagia freezes the selected ROI and telemetry-source membership,
their source/payload fingerprints, and the normalized selection parameters.
The worker verifies those inputs before publishing; a changed or missing input
fails the attempt instead of silently producing a different release. Artifact
records retain product-level attempt history. Retryable worker failures return
the artifact to `queued`; only the final exhausted attempt is marked failed.

Supported products are `raw_roi_statistics`, `binned_roi_statistics`,
`roi_evidence`, and `telemetry`. The first release exports refined ROIs only.
ROI statistics are analysis products: their principal tables use documented
scientific columns. Non-core metadata is emitted once in `asset_metadata`,
`frame_metadata`, and `roi_metadata` tables and joined through UUIDs; it is not
repeated for every ROI. Oversized XLSX tables are partitioned into numbered
worksheets in the same per-asset workbook. Telemetry bundles also include original source bytes, the import profile,
catalog records, normalized observations, and timeline context for a future
verified import workflow.

ROI evidence uses a compact, UUID-addressed layout:
`products/roi-evidence/{asset_uuid}/{frame_uuid}/{roi_uuid}.png`, with a
same-named `.json` sidecar containing the ROI, frame, asset, and ML-evidence
metadata.

The binned ROI product is one project-level time-series table rather than one
file per asset. Each row represents an exact frame capture time and includes a
count column for every bounding-box-area bin, plus ROI/frame/asset counts,
concurrent data-stream count, source frame dimensions, declared scan rate, and
available instrument/deployment context. A data stream uses its declared
`data_stream_id`, `stream_id`, or `camera_id`; when absent, the source asset is
treated as one stream. Mixed frame dimensions or scan rates are preserved as
semicolon-separated values, while their scalar columns are null.

While an export is running, its associated job reports estimated work units.
Input freezing reports every 1,000 selected records. ROI products then report
their read and write phases in 1,000-ROI batches; telemetry reports completed
sources; and archive construction reports checksum and packaging work in
100-file batches. The final visible phases are archive verification and
publication. Clients may derive an approximate ETA from the reported observed
rate; it is not a guarantee because serialization cost varies with image and
source-file size.

For example:

```json
{
  "products": ["raw_roi_statistics", "roi_evidence"],
  "formats": {"raw_roi_statistics": "xlsx"},
  "asset_ids": ["<asset-uuid>"],
  "roi_stage": "refined"
}
```

Run a worker with the `export_bundle` capability to process requests. Artifacts
are written beneath the configured `artifacts.local_root`, not into immutable
project KVStores.
