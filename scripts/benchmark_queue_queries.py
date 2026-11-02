#!/usr/bin/env python3
"""Run read-only queue-query benchmarks against a Pelagia PostgreSQL database.

The DSN is read from an environment variable and is never included in output.
Every connection uses a read-only transaction and a bounded statement timeout.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

try:
    import psycopg
except ImportError as exc:  # pragma: no cover
    raise SystemExit("psycopg is required") from exc


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction)))]


def safe_error(exc: BaseException) -> str:
    sqlstate = getattr(exc, "sqlstate", None)
    kind = type(exc).__name__
    if sqlstate:
        return f"{kind} (sqlstate {sqlstate})"
    return kind


def run_query(dsn: str, sql: str, params: tuple[Any, ...], timeout_ms: int, iterations: int) -> dict[str, Any]:
    timings: list[float] = []
    errors: list[str] = []
    sample: Any = None
    for _ in range(iterations):
        started = time.perf_counter()
        try:
            with psycopg.connect(dsn, autocommit=False) as connection:
                with connection.cursor() as cursor:
                    cursor.execute("SET TRANSACTION READ ONLY")
                    cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(timeout_ms),))
                    cursor.execute(sql, params)
                    rows = cursor.fetchall()
                    if sample is None:
                        sample = rows[:3]
            timings.append((time.perf_counter() - started) * 1000)
        except Exception as exc:  # benchmark should continue to other queries
            errors.append(safe_error(exc))
    return {
        "iterations": iterations,
        "completed": len(timings),
        "median_ms": statistics.median(timings) if timings else None,
        "p95_ms": percentile(timings, 0.95),
        "errors": errors,
        "sample": sample,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn-env", default="PELAGIA_TEST_DATABASE_DSN")
    parser.add_argument("--schema", default="pelagia")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--timeout-ms", type=int, default=8000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.timeout_ms < 1:
        parser.error("iterations and timeout-ms must be positive")
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", args.schema):
        raise SystemExit("schema must contain only letters, numbers, and underscores")
    dsn = os.environ.get(args.dsn_env)
    if not dsn:
        raise SystemExit(f"missing DSN environment variable: {args.dsn_env}")
    schema = args.schema
    project = args.project_id
    status = f"{schema}.frame_processing_status"
    detections = f"{schema}.detection_candidate"
    frames = f"{schema}.frames"
    assets = f"{schema}.raw_assets"
    queries: dict[str, tuple[str, tuple[Any, ...]]] = {
        "status_groups": (f"""SELECT preprocessing_status, candidate_detection_status,
            roi_refinement_status, count(*)::bigint
            FROM {status} WHERE project_id = %s GROUP BY 1,2,3""", (project,)),
        "area_top100": (f"""SELECT d.id, d.bbox_w::double precision*d.bbox_h AS bbox_area
            FROM {detections} d JOIN {frames} f ON f.id=d.frame_id
            JOIN {assets} a ON a.id=f.asset_id WHERE a.project_id=%s
            ORDER BY d.bbox_w::double precision*d.bbox_h DESC,
              f.frame_index DESC, d.roi_index ASC LIMIT 100""", (project,)),
        "candidate_count_area_100_10000": (f"""SELECT count(*)::bigint,
            count(*) FILTER (WHERE EXISTS (SELECT 1 FROM {schema}.detections_refined r
              WHERE r.candidate_detection_id=d.id))::bigint
            FROM {detections} d
            JOIN {frames} f ON f.id=d.frame_id JOIN {assets} a ON a.id=f.asset_id
            WHERE a.project_id=%s AND d.roi_payload_ref IS NOT NULL
              AND d.bbox_w::double precision*d.bbox_h BETWEEN 100 AND 10000""", (project,)),
    }
    result: dict[str, Any] = {"schema": schema, "project_id": project, "timeout_ms": args.timeout_ms}
    result["queries"] = {name: run_query(dsn, sql, params, args.timeout_ms, args.iterations) for name, (sql, params) in queries.items()}
    with psycopg.connect(dsn, autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(args.timeout_ms),))
            cursor.execute(f"SELECT EXISTS (SELECT 1 FROM information_schema.columns WHERE table_schema=%s AND table_name='frame_processing_status' AND column_name='refinement_queue_candidate_count')", (schema,))
            has_summary = bool(cursor.fetchone()[0])
    result["refinement_summary_columns_present"] = has_summary
    with psycopg.connect(dsn, autocommit=False) as connection:
        with connection.cursor() as cursor:
            cursor.execute("SET TRANSACTION READ ONLY")
            cursor.execute("SELECT set_config('statement_timeout', %s, true)", (str(args.timeout_ms),))
            cursor.execute("SELECT pg_relation_size(%s::regclass), current_setting('block_size')::bigint", (f"{schema}.detection_candidate",))
            bytes_size, block_size = cursor.fetchone()
            candidate_pages = int(bytes_size // block_size)
            cursor.execute("SELECT pg_relation_size(%s::regclass)", (f"{schema}.frame_processing_status",))
            status_pages = int(cursor.fetchone()[0] // block_size)
    result["candidate_pages"] = candidate_pages
    result["status_pages"] = status_pages
    sample_percent = 100.0 if candidate_pages <= 1024 else 25600.0 / candidate_pages
    status_sample_percent = 100.0 if status_pages <= 1024 else 25600.0 / status_pages
    result["auto_sample_percent"] = sample_percent
    result["auto_status_sample_percent"] = status_sample_percent
    result["queries"]["auto_candidate_area_refinement"] = run_query(
        dsn,
        f"""SELECT count(*)::bigint, count(*) FILTER (WHERE refined)::bigint,
            count(*) FILTER (WHERE NOT refined)::bigint
            FROM (SELECT d.id, EXISTS (SELECT 1 FROM {schema}.detections_refined r
              WHERE r.candidate_detection_id=d.id) AS refined
              FROM {detections} d TABLESAMPLE SYSTEM (%s) REPEATABLE(0)
              JOIN {status} s ON s.frame_id=d.frame_id
              WHERE s.project_id=%s AND d.roi_payload_ref IS NOT NULL
                AND d.bbox_w::double precision*d.bbox_h BETWEEN 100 AND 10000) sampled""",
        (sample_percent, project), args.timeout_ms, args.iterations,
    )
    result["queries"]["auto_status_groups"] = run_query(
        dsn,
        f"""SELECT preprocessing_status, candidate_detection_status,
            roi_refinement_status, count(*)::bigint
            FROM {status} TABLESAMPLE SYSTEM (%s) REPEATABLE(0)
            WHERE project_id=%s GROUP BY 1,2,3""",
        (status_sample_percent, project), args.timeout_ms, args.iterations,
    )
    # Match the facets endpoint rather than treating a simpler GROUP BY as a
    # page benchmark. Import only the existing summary-column builder; this
    # also works against an older deployed backend without sampling helpers.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from Pelagia.storage.postgres import PostgresRepository

    for mode, percentage in (("exact", 100.0), ("auto", status_sample_percent)):
        sampling = "" if percentage == 100 else f" TABLESAMPLE SYSTEM ({percentage:.12g}) REPEATABLE(0)"
        facet_sql = f"""WITH filtered AS MATERIALIZED (
            SELECT asset_id, collections, preprocessing_status, candidate_detection_status,
                   roi_refinement_status, candidate_detection_count, refined_detection_count,
                   unrefined_candidate_count, updated_at
            FROM {status} status{sampling} WHERE status.project_id = %s
        ), summary AS (
            SELECT {PostgresRepository._frame_status_summary_columns()} FROM filtered status
        ), asset_counts AS (
            SELECT asset_id::text AS value, COUNT(*)::bigint AS unit_count
            FROM filtered GROUP BY asset_id
        ), collection_counts AS (
            SELECT collection AS value, COUNT(*)::bigint AS unit_count
            FROM filtered CROSS JOIN LATERAL unnest(collections) AS collection GROUP BY collection
        ) SELECT summary.*,
            (SELECT jsonb_object_agg(value, unit_count) FROM asset_counts),
            (SELECT jsonb_object_agg(value, unit_count) FROM collection_counts)
        FROM summary"""
        result["queries"][f"{mode}_facets"] = run_query(
            dsn, facet_sql, (project,), args.timeout_ms, args.iterations,
        )
    if has_summary:
        result["queries"]["refinement_summary"] = run_query(dsn, f"SELECT COALESCE(sum(refinement_queue_candidate_count),0)::bigint, COALESCE(sum(refinement_queue_refined_candidate_count),0)::bigint FROM {status} WHERE project_id=%s", (project,), args.timeout_ms, args.iterations)
    else:
        result["queries"]["refinement_summary"] = {"skipped": True, "reason": "deployed schema lacks refinement_queue_candidate_count; use candidate_count_area query"}
    payload = json.dumps(result, default=str, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
