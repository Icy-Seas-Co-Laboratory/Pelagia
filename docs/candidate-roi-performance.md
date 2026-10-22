# Candidate ROI performance

Candidate generation uses a fast connected-component gate: `area` is
bounding-box width × height, and perimeter is 2 × (width + height). Both
connected-component and contour assembly apply that gate without a second
contour-perimeter pass during candidate generation. Standalone filtering of
materialized ROIs keeps bounding-box area but applies the external-contour
perimeter. Threshold equality is accepted. Candidate records produced by either
assembly method store bounding-box area and external-contour perimeter.

Connected-component statistics are filtered before allocating local masks and
extracting contours. Component traversal order and final one-based ROI indices
remain stable. Counts distinguish source components, materialized candidates,
filter survivors, and records retained after the detection limit.

Recording reuses resolved encoding options. Standalone calls still resolve
missing options from configuration; there is no process-global configuration
cache. Mask orientation uses binary-image moments for the centroid and
population covariance, with each row’s foreground endpoints for the projected
extent. Its coordinate conventions, degeneracy tolerances, and provenance method
remain unchanged.

Refined ROI records store the sum of OpenCV external-contour areas and closed
arc lengths. They do not use foreground-pixel count as the displayed area.
`metadata.roi_measurements` records version `1` and the area/perimeter method so
new measurements can be distinguished from historical rows. The external-contour convention does not subtract holes from area and does
not include inner hole boundaries in perimeter.

## Reproducible processing benchmark

Run from the repository root using the managed environment:

```bash
.venv/bin/python scripts/benchmark_candidate_rois.py --iterations 25 --output /tmp/roi-current.json
.venv/bin/python scripts/benchmark_candidate_rois.py --checkout /path/to/baseline-checkout --iterations 25 --output /tmp/roi-baseline.json
```

The same script can import either checkout. It uses four deterministic 1024 ×
1024 images, one OpenCV thread, three warm-up frames per scenario, and packaged
configuration defaults without local overrides or environment configuration.
Each frame uses manual thresholding, minimum width/height of five pixels, zero
padding, and raw ROI/mask encoding. A resolved application configuration is
provided, while any fallback configuration loads still perform real parsing.

The report includes mean, median, p95, throughput, configuration loads, a
separate instrumented phase sample, and a separate `tracemalloc` peak sample.
Memory figures describe traced allocations, not total process RSS. Geometry and
payload hashes must match across versions; floating-point orientation agreement
is tested separately with explicit tolerances.

These measurements exclude input decoding, preprocessing, mask augmentation,
database publication, and worker orchestration. They are synthetic code-path
benchmarks, not replacements for a production job measurement. Re-run production
frames with their original configuration to assess the reported 533 ms/frame
baseline.

## Measured processing results

Baseline commit: `9141ded752729d3f02dabf09f6be59fad34585e8`. Results use Python
3.12.2, NumPy 2.4.6, OpenCV 4.13.0, macOS ARM64, and one OpenCV thread. Each
scenario has 100 timed frames after three warmups. See the
[raw results](candidate-roi-benchmark-results.json) for means, throughput,
phase samples, allocation measurements, and matching output hashes.

| Scenario | Components → ROIs | Median before → after | Speedup | p95 before → after |
| --- | ---: | ---: | ---: | ---: |
| Many rejected components | 1,647 → 7 | 24.990 → 4.642 ms | 5.38× | 26.933 → 4.822 ms |
| Few components | 7 → 7 | 14.994 → 3.130 ms | 4.79× | 15.428 → 3.350 ms |
| Large object | 1 → 1 | 20.540 → 5.572 ms | 3.69× | 22.205 → 6.121 ms |
| Dense mask | 1 → 1 | 33.864 → 8.801 ms | 3.85× | 35.362 → 9.063 ms |

Configuration loads fell from 14 to zero for seven-ROI frames and from two to
zero for one-ROI frames. Geometry and raw payload hashes match in all four
scenarios. Peak traced allocations for the large-object scenario fell from
29.09 MB to 9.62 MB, and for the dense mask from 55.79 MB to 10.49 MB.

The separate orientation benchmark uses 500 timed measurements per case:

```bash
.venv/bin/python scripts/benchmark_roi_orientation.py --iterations 500
```

| Mask | Original median | Optimized median | Speedup |
| --- | ---: | ---: | ---: |
| Small 32 × 48 candidate | 0.0553 ms | 0.0424 ms | 1.31× |
| Large filled mask | 8.1613 ms | 0.9237 ms | 8.84× |
| Sparse disconnected mask | 1.7868 ms | 0.9545 ms | 1.87× |
| 6,550 isolated components | 1.8225 ms | 0.9430 ms | 1.93× |
| Ring with nested island | 2.1157 ms | 0.9496 ms | 2.23× |

## Timing interpretation

The job timing collector records disjoint phases for configuration loading,
connected-component labeling, component materialization, record preparation,
encoding, shape/gray measurements, orientation, and record finalization.
Assembly leaf timings are requested by the candidate path only, avoiding
nested measurements inside refinement's existing assembly timer.

Per-record `stage_durations_ms` still includes aggregate `roi_assembly` and
`roi_recording` durations for compatibility. These aggregate durations should
not be added to their leaf subphases. `source_component_count` records the raw
number of components before filtering; `materialized_candidate_count` records
masks/contours actually constructed; `filtered_candidate_count` remains the
count before the detection limit.

## Database publication

Publication uses Psycopg's pipelined `executemany` with a stable parameterized
INSERT statement, grouped by at most 250 rows and approximately 8 MiB of
parameter values. A single larger detection is allowed in its own group. The
row and byte bounds limit transient buffering; they do not split the
transaction. Delete-and-replace remains atomic across every group, with the
existing project-scope checks and database-generated IDs.

`replace_frame_detections(..., returning="full")` retains the existing full-row
return contract. Workers request `returning="ids"`, which avoids returning ROI
and mask payloads. `returning="none"` is available for callers that need no rows.
Groups use a fixed number of parameters per statement, so growing the group
size does not grow the statement's bind-parameter count.

Run the isolated local PostgreSQL benchmark with:

```bash
.venv/bin/python scripts/benchmark_detection_publication.py --baseline-checkout /path/to/baseline-checkout --iterations 100 --warmups 5
```

The script uses `PELAGIA_TEST_DATABASE_DSN` or the repository's local test DSN.
It creates and removes a uniquely named schema for each variant. Each timed
publication atomically replaces 27 detections across four frames, with 16 KiB
ROI plus 4 KiB mask payloads per detection. Payloads are seeded opaque random
bytes to resist database compression; this benchmark does not decode them.

| Publication path | Median/publication | Median/frame | p95/publication |
| --- | ---: | ---: | ---: |
| Baseline individual inserts, full rows | 5.885 ms | 1.471 ms | 7.292 ms |
| Pipelined groups, full rows | 4.924 ms | 1.231 ms | 5.665 ms |
| Pipelined groups, ID-only worker | 3.399 ms | 0.850 ms | 4.411 ms |

The final captured run shows a 42.3% lower worker publication median and a 39.5%
lower p95. Repeated comparisons with both baseline-first and baseline-last
execution orders also improved worker medians (38–42%). These local-database
numbers do not predict savings against the production 42 ms/frame publication
measurement; network latency, payload sizes, storage, and contention differ.

## Validation

Final full-suite run: `.venv/bin/pytest -q -rs` — **730 passed, 1 skipped**.
The skip is the existing JPEG XS codec test because the installed `imagecodecs`
build lacks JPEG XS support. PostgreSQL integration tests ran successfully,
including rollback when a later publication group fails. The suite also covers
project isolation, API full-row compatibility, worker ID-only returns, geometric
filter boundaries, stable candidate indices and limits, standalone configuration
fallback, and orientation agreement with the prior coordinate-array algorithm.
