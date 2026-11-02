# Queue preview and ROI size performance

Measurements on 2026-11-02 compared local PostgreSQL 18.4 with a remote
PostgreSQL 16 deployment containing approximately 26.5 million candidates and
864,000 frame-status rows. Local data contained approximately 346,000
candidates and 8,225 status rows. Cardinalities are catalog estimates.

The read-only harness executes SQL with an eight-second statement timeout.
Three runs include connection setup and result fetching, but exclude HTTP,
serialization by FastAPI, and browser rendering. The reported p95 is simply
the highest observed value in these three runs. Raw timing summaries are in
[queue-performance-benchmark.json](queue-performance-benchmark.json).

| Query | Local median | Remote median | Remote highest |
| --- | ---: | ---: | ---: |
| Full exact facets, matching endpoint SQL | 15.5 ms | 2,701.2 ms | 2,811.1 ms |
| Auto facets | 14.0 ms | 42.3 ms | 45.4 ms |
| Exact area-filtered candidate/refined-parent count | 520.2 ms | >8,000 ms, cancelled | >8,000 ms |
| Auto area-filtered candidate/refined-parent count | 5.7 ms | 69.1 ms | 72.0 ms |
| Initialized refinement totals | 5.0 ms | Not available on deployed schema | — |

The cheaper status-only GROUP BY is recorded separately in the JSON; it does
not represent the full facets request, which also builds the summary and
asset/collection options. The two auto modes sample their own source tables
independently. The local status table is small enough to remain exact.

The remote candidate sample targeted 256 of 6,617,600 heap pages and contained
810 matching candidates for the 100–10,000 px² area gate, with no refined
parents. This is sample evidence, not an accuracy guarantee. Sparse project or
area selections can produce no estimate. See [processing-status.md](processing-status.md)
for response metadata and client behavior.

## Area index experiment

The production remote area sort exceeded eight seconds. To test the proposed
index without changing production tables, a connection created a temporary
geometry-only copy of one million existing candidates and joined it to the
existing frames/assets for project scoping. The same top-100 query took
1,061.6 ms before indexing. Adding the index while retaining explicit
`DESC NULLS LAST` still took 990.3 ms. Omitting the unnecessary null clause
allowed a backward expression-index scan and incremental sort, taking 0.7 ms.
Both bbox dimensions are non-null, so size ordering is preserved. The connection
closed and removed its temporary table and index.

These single-run geometry measurements establish that the query can use the
index. They do not measure full ROI payload delivery, a production index over
all 26.5 million rows, or cold-cache/multi-project performance. The migration
adds indexes to both candidate and refined tables; large deployments should
follow the concurrent-build procedure in [processing-status.md](processing-status.md).

## Repeating measurements

Set `PELAGIA_TEST_DATABASE_DSN` privately in the environment, then run:

```bash
.venv/bin/python scripts/benchmark_queue_queries.py \
  --schema pelagia --project-id PROJECT_UUID --iterations 3 \
  --timeout-ms 8000 --output /tmp/queue-benchmark.json
```

The harness performs no schema initialization or persistent data writes.
It detects older schemas without refinement totals and records that query as
unavailable. It does not log connection strings or exception messages that
could contain credentials. The index-copy experiment above is separate from
this read-only harness.

## Rollout and follow-up

Deploy the backend count-mode contract before enabling the updated frontend.
Prebuild valid area indexes concurrently for large existing catalogs, apply
migrations, and reconcile existing frame refinement totals asset by asset.
Auto previews remain useful during the backfill through sampled candidates.

After rollout, measure complete HTTP requests and browser rendering under
concurrent processing, including narrow filters and projects that own a small
fraction of a shared table. The measured SQL supports the sub-second page
target, but does not establish an end-to-end latency guarantee. If operators
need accurate previews for sparse selections, add per-asset summaries or an
explicit asynchronous exact-count action rather than increasing polling work.
Deep ROI browsing is a separate candidate for cursor pagination; offset-based
pages still become more expensive as the offset increases.
