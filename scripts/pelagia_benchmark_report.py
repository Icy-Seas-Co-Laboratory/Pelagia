#!/usr/bin/env python
"""Generate a PDF benchmark report from Pelagia PostgreSQL timing data.

Example:
    python scripts/pelagia_benchmark_report.py \
        --database-dsn postgresql://postgres:postgres@localhost:5432/pelagia \
        --schema pelagia \
        --output reports/pelagia-benchmark.pdf \
        --since "2026-06-01"
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Pelagia.config import CoreConfig
from Pelagia.utils.validation import validate_schema_name

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - exercised when postgres extras are absent.
    psycopg = None
    dict_row = None


PAGE_WIDTH = 612
PAGE_HEIGHT = 792
MARGIN = 48
LINE_HEIGHT = 12


@dataclass(frozen=True)
class QueryWindow:
    since: str | None
    until: str | None
    run_id: str | None
    asset_id: str | None
    collection: str | None


class SimplePdf:
    """Very small PDF writer for dependency-free benchmark reports."""

    def __init__(self, title: str) -> None:
        self.title = title
        self.pages: list[list[str]] = []
        self.current: list[str] = []
        self.y = PAGE_HEIGHT - MARGIN
        self._new_page()

    def _new_page(self) -> None:
        if self.current:
            self.pages.append(self.current)
        self.current = []
        self.y = PAGE_HEIGHT - MARGIN

    def _ensure_space(self, lines: int = 1) -> None:
        if self.y - lines * LINE_HEIGHT < MARGIN:
            self._new_page()

    def text(
        self,
        value: str,
        *,
        x: float = MARGIN,
        size: int = 10,
        font: str = "F1",
        leading: float | None = None,
    ) -> None:
        leading = leading or max(LINE_HEIGHT, size + 3)
        self._ensure_space(1)
        escaped = _pdf_escape(value)
        self.current.append(f"BT /{font} {size} Tf 1 0 0 1 {x:.2f} {self.y:.2f} Tm ({escaped}) Tj ET")
        self.y -= leading

    def heading(self, value: str) -> None:
        self._ensure_space(3)
        self.y -= 6
        self.text(value, size=14, font="F2", leading=18)

    def paragraph(self, value: str, *, width_chars: int = 95) -> None:
        for line in _wrap(value, width_chars):
            self.text(line, size=10, leading=13)

    def table(self, headers: list[str], rows: Iterable[Iterable[Any]], *, widths: list[int]) -> None:
        rows = list(rows)
        self._ensure_space(3)
        self.text(_fixed_row(headers, widths), size=8, font="F3", leading=10)
        self.text(_fixed_row(["-" * min(width, 18) for width in widths], widths), size=8, font="F3", leading=10)
        for row in rows:
            cells = [_format_cell(cell) for cell in row]
            wrapped_cells = [_wrap_cell(cell, width) for cell, width in zip(cells, widths)]
            row_height = max(len(lines) for lines in wrapped_cells)
            self._ensure_space(row_height)
            for line_index in range(row_height):
                line_cells = [
                    lines[line_index] if line_index < len(lines) else ""
                    for lines in wrapped_cells
                ]
                self.text(_fixed_row(line_cells, widths), size=8, font="F3", leading=10)
        self.y -= 6

    def write(self, path: Path) -> None:
        if self.current:
            self.pages.append(self.current)
            self.current = []

        objects: list[bytes] = []
        objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
        page_refs = " ".join(f"{3 + index * 2} 0 R" for index in range(len(self.pages)))
        objects.append(f"<< /Type /Pages /Kids [{page_refs}] /Count {len(self.pages)} >>".encode("ascii"))
        font_object_id = 3 + len(self.pages) * 2
        for index, page in enumerate(self.pages):
            page_id = 3 + index * 2
            content_id = page_id + 1
            objects.append(
                (
                    f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                    f"/Resources << /Font << /F1 {font_object_id} 0 R /F2 {font_object_id + 1} 0 R "
                    f"/F3 {font_object_id + 2} 0 R >> >> /Contents {content_id} 0 R >>"
                ).encode("ascii")
            )
            stream = "\n".join(page).encode("latin-1", errors="replace")
            objects.append(b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream")
        objects.extend(
            [
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
                b"<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>",
            ]
        )

        payload = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
        offsets = [0]
        for object_id, obj in enumerate(objects, start=1):
            offsets.append(len(payload))
            payload.extend(f"{object_id} 0 obj\n".encode("ascii"))
            payload.extend(obj)
            payload.extend(b"\nendobj\n")
        xref_offset = len(payload)
        payload.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
        payload.extend(b"0000000000 65535 f \n")
        for offset in offsets[1:]:
            payload.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
        payload.extend(
            (
                f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
                f"startxref\n{xref_offset}\n%%EOF\n"
            ).encode("ascii")
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(payload))


def main() -> None:
    args = parse_args()
    if psycopg is None:
        raise SystemExit("psycopg is required. Install with: python -m pip install 'psycopg[binary]'")

    config = CoreConfig.load(config_path=args.config) if args.config else CoreConfig.load()
    dsn = args.database_dsn or config.database.dsn
    schema = validate_schema_name(args.schema or config.database.schema_name)
    window = QueryWindow(
        since=args.since,
        until=args.until,
        run_id=args.run_id,
        asset_id=args.asset_id,
        collection=args.collection,
    )
    report = collect_benchmark_data(dsn, schema, window, top_n=args.top_n)
    write_pdf_report(report, Path(args.output), schema=schema, window=window)
    if args.json_output:
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"Wrote benchmark report: {args.output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-dsn", help="PostgreSQL DSN. Defaults to Pelagia config.")
    parser.add_argument("--schema", help="PostgreSQL schema. Defaults to Pelagia config.")
    parser.add_argument("--config", help="Optional config.toml path.")
    parser.add_argument("--output", default="reports/pelagia-benchmark.pdf", help="Output PDF path.")
    parser.add_argument("--json-output", help="Optional JSON export path for the collected summary.")
    parser.add_argument("--since", help="Include rows at or after this timestamp.")
    parser.add_argument("--until", help="Include rows before or at this timestamp.")
    parser.add_argument("--run-id", help="Filter by run_id.")
    parser.add_argument("--asset-id", help="Filter by asset_id.")
    parser.add_argument("--collection", help="Filter assets/logs/jobs by collection membership.")
    parser.add_argument("--top-n", type=int, default=15, help="Number of slowest rows/failures to include.")
    return parser.parse_args()


def collect_benchmark_data(dsn: str, schema: str, window: QueryWindow, *, top_n: int) -> dict[str, Any]:
    with psycopg.connect(dsn, row_factory=dict_row, autocommit=True) as connection:
        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "window": window.__dict__,
            "overview": query_overview(connection, schema, window),
            "duration_by_event": query_duration_by_event(connection, schema, window, top_n=25),
            "job_lifecycle": query_job_lifecycle(connection, schema, window),
            "worker_handlers": query_worker_handlers(connection, schema, window),
            "throughput": query_throughput(connection, schema, window),
            "slow_logs": query_slow_logs(connection, schema, window, limit=top_n),
            "slow_jobs": query_slow_jobs(connection, schema, window, limit=top_n),
            "warnings_errors": query_warnings_errors(connection, schema, window, limit=top_n),
        }


def query_overview(connection, schema: str, window: QueryWindow) -> dict[str, Any]:
    log_where, log_params = log_filter(schema, window)
    job_where, job_params = job_filter(schema, window)
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                MIN(created_at) AS first_log_at,
                MAX(created_at) AS last_log_at,
                COUNT(*) AS log_count,
                COUNT(duration_ms) AS timed_log_count,
                COUNT(*) FILTER (WHERE level = 'error') AS error_log_count,
                COUNT(*) FILTER (WHERE level = 'warning') AS warning_log_count
            FROM {schema}.logs logs
            {log_where}
            """,
            log_params,
        )
        overview = dict(cursor.fetchone() or {})
        cursor.execute(
            f"""
            SELECT
                COUNT(*) AS job_count,
                COUNT(*) FILTER (WHERE status = 'succeeded') AS succeeded_jobs,
                COUNT(*) FILTER (WHERE status = 'failed') AS failed_jobs,
                COUNT(*) FILTER (WHERE status = 'queued') AS queued_jobs,
                COUNT(*) FILTER (WHERE started_at IS NOT NULL AND finished_at IS NOT NULL) AS timed_jobs
            FROM {schema}.processing_jobs jobs
            {job_where}
            """,
            job_params,
        )
        overview.update(dict(cursor.fetchone() or {}))
        cursor.execute(f"SELECT COUNT(*) AS count FROM {schema}.worker_sessions")
        overview["worker_session_count"] = cursor.fetchone()["count"]
        cursor.execute(*asset_count_sql(schema, window))
        overview["asset_count"] = cursor.fetchone()["count"]
        cursor.execute(*frame_count_sql(schema, window))
        overview["frame_count"] = cursor.fetchone()["count"]
        cursor.execute(*detection_count_sql(schema, window))
        overview["detection_candidate_count"] = cursor.fetchone()["count"]
    return overview


def query_duration_by_event(connection, schema: str, window: QueryWindow, *, top_n: int) -> list[dict[str, Any]]:
    where, params = log_filter(schema, window, extra=["logs.duration_ms IS NOT NULL"])
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                logs.event_type,
                logs.logger,
                COUNT(*) AS count,
                AVG(logs.duration_ms) AS avg_ms,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY logs.duration_ms) AS p50_ms,
                percentile_cont(0.90) WITHIN GROUP (ORDER BY logs.duration_ms) AS p90_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY logs.duration_ms) AS p95_ms,
                MAX(logs.duration_ms) AS max_ms,
                SUM(logs.duration_ms) AS total_ms
            FROM {schema}.logs logs
            {where}
            GROUP BY logs.event_type, logs.logger
            ORDER BY total_ms DESC NULLS LAST
            LIMIT %s
            """,
            [*params, top_n],
        )
        return cursor.fetchall()


def query_job_lifecycle(connection, schema: str, window: QueryWindow) -> list[dict[str, Any]]:
    where, params = job_filter(
        schema,
        window,
        extra=["jobs.started_at IS NOT NULL", "jobs.finished_at IS NOT NULL"],
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                jobs.stage,
                jobs.status,
                COUNT(*) AS count,
                AVG(EXTRACT(EPOCH FROM (jobs.started_at - jobs.created_at)) * 1000) AS avg_queue_ms,
                percentile_cont(0.50) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (jobs.started_at - jobs.created_at)) * 1000
                ) AS p50_queue_ms,
                AVG(EXTRACT(EPOCH FROM (jobs.finished_at - jobs.started_at)) * 1000) AS avg_run_ms,
                percentile_cont(0.50) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (jobs.finished_at - jobs.started_at)) * 1000
                ) AS p50_run_ms,
                percentile_cont(0.95) WITHIN GROUP (
                    ORDER BY EXTRACT(EPOCH FROM (jobs.finished_at - jobs.started_at)) * 1000
                ) AS p95_run_ms,
                MAX(EXTRACT(EPOCH FROM (jobs.finished_at - jobs.started_at)) * 1000) AS max_run_ms
            FROM {schema}.processing_jobs jobs
            {where}
            GROUP BY jobs.stage, jobs.status
            ORDER BY jobs.stage, jobs.status
            """,
            params,
        )
        return cursor.fetchall()


def query_worker_handlers(connection, schema: str, window: QueryWindow) -> list[dict[str, Any]]:
    where, params = log_filter(
        schema,
        window,
        extra=["logs.event_type = 'job.handler_completed'", "logs.duration_ms IS NOT NULL"],
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                COALESCE(logs.payload->>'stage', 'unknown') AS stage,
                COALESCE(logs.worker_id, 'unknown') AS worker_id,
                COUNT(*) AS count,
                AVG(logs.duration_ms) AS avg_ms,
                percentile_cont(0.50) WITHIN GROUP (ORDER BY logs.duration_ms) AS p50_ms,
                percentile_cont(0.95) WITHIN GROUP (ORDER BY logs.duration_ms) AS p95_ms,
                MAX(logs.duration_ms) AS max_ms
            FROM {schema}.logs logs
            {where}
            GROUP BY COALESCE(logs.payload->>'stage', 'unknown'), COALESCE(logs.worker_id, 'unknown')
            ORDER BY stage, avg_ms DESC NULLS LAST
            """,
            params,
        )
        return cursor.fetchall()


def query_throughput(connection, schema: str, window: QueryWindow) -> list[dict[str, Any]]:
    where, params = log_filter(schema, window, extra=["logs.duration_ms IS NOT NULL", "logs.duration_ms > 0"])
    unit_expr = """
        COALESCE(
            NULLIF(logs.payload->>'stored_frame_count', '')::double precision,
            NULLIF(logs.payload->>'frame_count', '')::double precision,
            NULLIF(logs.payload->>'detection_count', '')::double precision,
            NULLIF(logs.payload->>'source_frame_count', '')::double precision
        )
    """
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                logs.event_type,
                COUNT(*) AS count,
                SUM({unit_expr}) AS unit_count,
                SUM(logs.duration_ms) AS total_ms,
                AVG(logs.duration_ms) AS avg_ms
            FROM {schema}.logs logs
            {where}
            GROUP BY logs.event_type
            HAVING SUM({unit_expr}) IS NOT NULL
            ORDER BY unit_count DESC NULLS LAST
            """,
            params,
        )
        rows = cursor.fetchall()
    for row in rows:
        total_ms = float(row.get("total_ms") or 0)
        units = float(row.get("unit_count") or 0)
        row["units_per_second"] = None if total_ms <= 0 else units / (total_ms / 1000)
    return rows


def query_slow_logs(connection, schema: str, window: QueryWindow, *, limit: int) -> list[dict[str, Any]]:
    where, params = log_filter(schema, window, extra=["logs.duration_ms IS NOT NULL"])
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                logs.created_at,
                logs.event_type,
                logs.logger,
                logs.level,
                logs.duration_ms,
                logs.worker_id,
                logs.job_id,
                logs.message,
                logs.payload
            FROM {schema}.logs logs
            {where}
            ORDER BY logs.duration_ms DESC
            LIMIT %s
            """,
            [*params, limit],
        )
        return cursor.fetchall()


def query_slow_jobs(connection, schema: str, window: QueryWindow, *, limit: int) -> list[dict[str, Any]]:
    where, params = job_filter(
        schema,
        window,
        extra=["jobs.started_at IS NOT NULL", "jobs.finished_at IS NOT NULL"],
    )
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                jobs.id,
                jobs.stage,
                jobs.status,
                jobs.worker_id,
                EXTRACT(EPOCH FROM (jobs.started_at - jobs.created_at)) * 1000 AS queue_ms,
                EXTRACT(EPOCH FROM (jobs.finished_at - jobs.started_at)) * 1000 AS run_ms,
                jobs.summary,
                jobs.error_message
            FROM {schema}.processing_jobs jobs
            {where}
            ORDER BY run_ms DESC NULLS LAST
            LIMIT %s
            """,
            [*params, limit],
        )
        return cursor.fetchall()


def query_warnings_errors(connection, schema: str, window: QueryWindow, *, limit: int) -> list[dict[str, Any]]:
    where, params = log_filter(schema, window, extra=["logs.level IN ('warning', 'error')"])
    with connection.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT
                logs.created_at,
                logs.level,
                logs.event_type,
                logs.message,
                logs.worker_id,
                logs.job_id,
                logs.payload
            FROM {schema}.logs logs
            {where}
            ORDER BY logs.created_at DESC, logs.id DESC
            LIMIT %s
            """,
            [*params, limit],
        )
        return cursor.fetchall()


def log_filter(schema: str, window: QueryWindow, extra: list[str] | None = None) -> tuple[str, list[Any]]:
    clauses = list(extra or [])
    params: list[Any] = []
    if window.since:
        clauses.append("logs.created_at >= %s")
        params.append(window.since)
    if window.until:
        clauses.append("logs.created_at <= %s")
        params.append(window.until)
    if window.run_id:
        clauses.append("logs.run_id = %s")
        params.append(window.run_id)
    if window.asset_id:
        clauses.append("logs.asset_id = %s")
        params.append(window.asset_id)
    if window.collection:
        clauses.append(
            f"""
            EXISTS (
                SELECT 1 FROM {schema}.raw_assets assets
                WHERE assets.id = logs.asset_id AND %s = ANY(assets.collections)
            )
            """
        )
        params.append(window.collection)
    return where_sql(clauses), params


def job_filter(schema: str, window: QueryWindow, extra: list[str] | None = None) -> tuple[str, list[Any]]:
    clauses = list(extra or [])
    params: list[Any] = []
    if window.since:
        clauses.append("jobs.created_at >= %s")
        params.append(window.since)
    if window.until:
        clauses.append("jobs.created_at <= %s")
        params.append(window.until)
    if window.run_id:
        clauses.append("jobs.run_id = %s")
        params.append(window.run_id)
    if window.asset_id:
        clauses.append("jobs.asset_id = %s")
        params.append(window.asset_id)
    if window.collection:
        clauses.append(
            f"""
            EXISTS (
                SELECT 1 FROM {schema}.raw_assets assets
                WHERE assets.id = jobs.asset_id AND %s = ANY(assets.collections)
            )
            """
        )
        params.append(window.collection)
    return where_sql(clauses), params


def where_sql(clauses: list[str]) -> str:
    return "" if not clauses else "WHERE " + " AND ".join(f"({clause})" for clause in clauses)


def asset_count_sql(schema: str, window: QueryWindow) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if window.run_id:
        clauses.append("run_id = %s")
        params.append(window.run_id)
    if window.asset_id:
        clauses.append("id = %s")
        params.append(window.asset_id)
    if window.collection:
        clauses.append("%s = ANY(collections)")
        params.append(window.collection)
    return f"SELECT COUNT(*) AS count FROM {schema}.raw_assets {where_sql(clauses)}", params


def frame_count_sql(schema: str, window: QueryWindow) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if window.run_id:
        clauses.append("frames.run_id = %s")
        params.append(window.run_id)
    if window.asset_id:
        clauses.append("frames.asset_id = %s")
        params.append(window.asset_id)
    if window.collection:
        clauses.append("%s = ANY(assets.collections)")
        params.append(window.collection)
    join = f"JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id" if window.collection else ""
    return f"SELECT COUNT(*) AS count FROM {schema}.frames frames {join} {where_sql(clauses)}", params


def detection_count_sql(schema: str, window: QueryWindow) -> tuple[str, list[Any]]:
    clauses = []
    params: list[Any] = []
    if window.asset_id:
        clauses.append("frames.asset_id = %s")
        params.append(window.asset_id)
    if window.run_id:
        clauses.append("detections.run_id = %s")
        params.append(window.run_id)
    if window.collection:
        clauses.append("%s = ANY(assets.collections)")
        params.append(window.collection)
    joins = [
        f"JOIN {schema}.frames frames ON frames.id = detections.frame_id",
    ]
    if window.collection:
        joins.append(f"JOIN {schema}.raw_assets assets ON assets.id = frames.asset_id")
    return (
        f"SELECT COUNT(*) AS count FROM {schema}.detection_candidate detections {' '.join(joins)} {where_sql(clauses)}",
        params,
    )


def write_pdf_report(report: dict[str, Any], output: Path, *, schema: str, window: QueryWindow) -> None:
    pdf = SimplePdf("Pelagia Benchmark Report")
    overview = report["overview"]
    pdf.text("Pelagia Benchmark Report", size=20, font="F2", leading=26)
    pdf.text(f"Generated: {report['generated_at']}", size=9)
    pdf.text(f"Schema: {schema}", size=9)
    filters = ", ".join(f"{key}={value}" for key, value in window.__dict__.items() if value) or "none"
    pdf.text(f"Filters: {filters}", size=9)
    pdf.heading("Summary")
    pdf.table(
        ["Metric", "Value"],
        [
            ["First log", overview.get("first_log_at")],
            ["Last log", overview.get("last_log_at")],
            ["Logs", overview.get("log_count")],
            ["Timed logs", overview.get("timed_log_count")],
            ["Warnings", overview.get("warning_log_count")],
            ["Errors", overview.get("error_log_count")],
            ["Jobs", overview.get("job_count")],
            ["Succeeded jobs", overview.get("succeeded_jobs")],
            ["Failed jobs", overview.get("failed_jobs")],
            ["Queued jobs", overview.get("queued_jobs")],
            ["Timed jobs", overview.get("timed_jobs")],
            ["Assets", overview.get("asset_count")],
            ["Frames", overview.get("frame_count")],
            ["Candidate detections", overview.get("detection_candidate_count")],
            ["Worker sessions", overview.get("worker_session_count")],
        ],
        widths=[28, 60],
    )
    pdf.paragraph(
        "Timing comes from logs.duration_ms for processing and worker events, plus processing_jobs "
        "created_at/started_at/finished_at for queue wait and job runtime. Percentiles are computed in PostgreSQL."
    )

    pdf.heading("Duration by Event")
    pdf.table(
        ["event_type", "logger", "n", "avg", "p50", "p90", "p95", "max", "total"],
        [
            [
                row.get("event_type"),
                row.get("logger"),
                row.get("count"),
                ms(row.get("avg_ms")),
                ms(row.get("p50_ms")),
                ms(row.get("p90_ms")),
                ms(row.get("p95_ms")),
                ms(row.get("max_ms")),
                seconds(row.get("total_ms")),
            ]
            for row in report["duration_by_event"]
        ],
        widths=[26, 24, 5, 8, 8, 8, 8, 8, 9],
    )

    pdf.heading("Job Lifecycle by Stage")
    pdf.table(
        ["stage", "status", "n", "avg wait", "p50 wait", "avg run", "p50 run", "p95 run", "max run"],
        [
            [
                row.get("stage"),
                row.get("status"),
                row.get("count"),
                ms(row.get("avg_queue_ms")),
                ms(row.get("p50_queue_ms")),
                ms(row.get("avg_run_ms")),
                ms(row.get("p50_run_ms")),
                ms(row.get("p95_run_ms")),
                ms(row.get("max_run_ms")),
            ]
            for row in report["job_lifecycle"]
        ],
        widths=[18, 12, 5, 10, 10, 10, 10, 10, 10],
    )

    pdf.heading("Worker Handler Timing")
    pdf.table(
        ["stage", "worker", "n", "avg", "p50", "p95", "max"],
        [
            [
                row.get("stage"),
                row.get("worker_id"),
                row.get("count"),
                ms(row.get("avg_ms")),
                ms(row.get("p50_ms")),
                ms(row.get("p95_ms")),
                ms(row.get("max_ms")),
            ]
            for row in report["worker_handlers"]
        ],
        widths=[18, 22, 5, 10, 10, 10, 10],
    )

    pdf.heading("Throughput Estimates")
    pdf.table(
        ["event_type", "n", "units", "total time", "units/s", "avg event"],
        [
            [
                row.get("event_type"),
                row.get("count"),
                integer(row.get("unit_count")),
                seconds(row.get("total_ms")),
                rate(row.get("units_per_second")),
                ms(row.get("avg_ms")),
            ]
            for row in report["throughput"]
        ],
        widths=[30, 5, 10, 12, 10, 10],
    )

    pdf.heading("Slowest Timed Logs")
    pdf.table(
        ["created_at", "event", "logger", "duration", "worker", "message"],
        [
            [
                short_time(row.get("created_at")),
                row.get("event_type"),
                row.get("logger"),
                ms(row.get("duration_ms")),
                row.get("worker_id"),
                row.get("message"),
            ]
            for row in report["slow_logs"]
        ],
        widths=[18, 26, 22, 10, 14, 35],
    )

    pdf.heading("Slowest Jobs")
    pdf.table(
        ["job_id", "stage", "status", "worker", "queue", "runtime", "summary"],
        [
            [
                row.get("id"),
                row.get("stage"),
                row.get("status"),
                row.get("worker_id"),
                ms(row.get("queue_ms")),
                ms(row.get("run_ms")),
                row.get("summary") or row.get("error_message"),
            ]
            for row in report["slow_jobs"]
        ],
        widths=[18, 16, 10, 14, 10, 10, 38],
    )

    pdf.heading("Recent Warnings and Errors")
    pdf.table(
        ["created_at", "level", "event", "worker", "message"],
        [
            [
                short_time(row.get("created_at")),
                row.get("level"),
                row.get("event_type"),
                row.get("worker_id"),
                row.get("message") or _payload_error(row.get("payload")),
            ]
            for row in report["warnings_errors"]
        ],
        widths=[18, 8, 28, 14, 48],
    )

    pdf.write(output)


def ms(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    if abs(number) >= 1000:
        return f"{number / 1000:.2f}s"
    return f"{number:.1f}ms"


def seconds(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number / 1000:.2f}s"


def rate(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return f"{number:.2f}"


def integer(value: Any) -> str:
    number = _float(value)
    if number is None:
        return "-"
    return str(int(round(number)))


def short_time(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    return text.replace("+00:00", "Z")[:19]


def _payload_error(payload: Any) -> str:
    if isinstance(payload, dict):
        return str(payload.get("error_message") or payload.get("error_type") or "")
    return ""


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, dict):
        return json.dumps(value, default=str)[:160]
    return str(value)


def _wrap_cell(value: str, width: int) -> list[str]:
    return _wrap(value, width) or [""]


def _fixed_row(values: Iterable[Any], widths: list[int]) -> str:
    cells = []
    for value, width in zip(values, widths):
        text = _format_cell(value).replace("\n", " ")
        cells.append(text[:width].ljust(width))
    return "  ".join(cells)


def _wrap(value: str, width: int) -> list[str]:
    words = str(value).replace("\n", " ").split()
    if not words:
        return [""]
    lines: list[str] = []
    current = ""
    for word in words:
        if len(word) > width:
            if current:
                lines.append(current)
                current = ""
            lines.extend(word[i : i + width] for i in range(0, len(word), width))
        elif not current:
            current = word
        elif len(current) + 1 + len(word) <= width:
            current += " " + word
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _pdf_escape(value: str) -> str:
    text = str(value)
    text = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    return text.encode("latin-1", errors="replace").decode("latin-1")


if __name__ == "__main__":
    main()
