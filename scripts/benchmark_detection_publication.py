#!/usr/bin/env python3
"""Benchmark candidate-detection replacement against a baseline checkout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import random
from pathlib import Path
import statistics
import subprocess
import sys
import time
import uuid


DEFAULT_TEST_DSN = "postgresql://postgres:postgres@localhost:5432/pelagia"


def _child_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--child-source", type=Path)
    parser.add_argument("--variant", choices=("baseline_full", "batched_full", "batched_ids"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=5)
    return parser


def _run_child(args: argparse.Namespace) -> None:
    sys.path.insert(0, str(args.child_source.resolve()))

    from Pelagia.config import CoreConfig
    from Pelagia.domain import (
        AssetKind,
        DetectionRecord,
        FrameRecord,
        PlannedRun,
        RawAssetManifest,
        RunManifest,
    )
    from Pelagia.storage.postgres import DEFAULT_PROJECT_ID, PostgresRepository

    schema = f"pelagia_publication_bench_{uuid.uuid4().hex}"
    config = CoreConfig()
    config.database.dsn = os.getenv("PELAGIA_TEST_DATABASE_DSN", DEFAULT_TEST_DSN)
    config.database.schema_name = schema
    config.database.pool_min_size = 1
    config.database.pool_max_size = 2
    repository = PostgresRepository(config)

    try:
        repository.initialize_schema()
        repository.ensure_default_project()
        run_id = str(uuid.uuid4())
        asset_id = str(uuid.uuid4())
        repository.register_planned_run(
            PlannedRun(
                manifest=RunManifest(
                    run_id=run_id,
                    run_key=f"publication-benchmark-{uuid.uuid4().hex}",
                    instrument="benchmark",
                    source_path="/tmp/benchmark.avi",
                    source_type=AssetKind.VIDEO.value,
                    created_at=datetime.now(timezone.utc),
                    assets=[
                        RawAssetManifest(
                            asset_id=asset_id,
                            filename="benchmark.avi",
                            path="/tmp/benchmark.avi",
                            kind=AssetKind.VIDEO,
                            size_bytes=1,
                            checksum="sha256:benchmark",
                            collections=["benchmark"],
                            media_count=4,
                        )
                    ],
                ),
                jobs=[],
            ),
            project_id=DEFAULT_PROJECT_ID,
        )
        frames = repository.replace_frames(
            run_id,
            [
                FrameRecord(
                    asset_id=asset_id,
                    frame_index=index,
                    width=1024,
                    height=1024,
                    preview_thumbhash=b"",
                    kvstore_hash=f"frame-{index}",
                )
                for index in range(4)
            ],
        )
        frame_ids = [str(frame["id"]) for frame in frames]
        # Opaque, deterministic payloads resist TOAST compression like encoded images.
        payload_rng = random.Random(42)
        detections = [
            DetectionRecord(
                run_id=run_id,
                frame_id=frame_ids[index % len(frame_ids)],
                roi_index=index,
                bbox_x=index,
                bbox_y=index,
                bbox_w=128,
                bbox_h=128,
                area=8_192.0,
                perimeter=512.0,
                major_axis_length=128.0,
                minor_axis_length=64.0,
                min_gray_value=0,
                mean_gray_value=42.0,
                roi_payload=payload_rng.randbytes(16_384),
                mask_payload=payload_rng.randbytes(4_096),
                roi_encoding="zstd",
                roi_format="zstd_ndarray_c_order",
                roi_dtype="uint8",
                roi_shape=[128, 128],
                mask_encoding="zstd",
                mask_format="zstd_ndarray_c_order",
                mask_dtype="uint8",
                mask_shape=[128, 128],
                metadata={"benchmark": True, "index": index},
            )
            for index in range(27)
        ]

        def publish():
            kwargs = {"project_id": DEFAULT_PROJECT_ID}
            if args.variant == "batched_ids":
                kwargs["returning"] = "ids"
            return repository.replace_frame_detections(run_id, frame_ids, detections, **kwargs)

        for _ in range(args.warmups):
            assert len(publish()) == len(detections)

        samples_ms = []
        for _ in range(args.iterations):
            started = time.perf_counter()
            rows = publish()
            samples_ms.append((time.perf_counter() - started) * 1_000)
            assert len(rows) == len(detections)

        ordered = sorted(samples_ms)
        p95_index = max(0, int(len(ordered) * 0.95) - 1)
        median_ms = statistics.median(samples_ms)
        print(
            json.dumps(
                {
                    "variant": args.variant,
                    "detections": len(detections),
                    "frames": len(frame_ids),
                    "payload_bytes_per_detection": 20_480,
                    "payload_distribution": "seeded opaque random bytes",
                    "iterations": len(samples_ms),
                    "median_ms_per_publication": median_ms,
                    "p95_ms_per_publication": ordered[p95_index],
                    "median_ms_per_frame": median_ms / len(frame_ids),
                },
                sort_keys=True,
            )
        )
    finally:
        try:
            with repository.connect() as connection:
                with connection.cursor() as cursor:
                    cursor.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
                connection.commit()
        finally:
            repository.close()


def _run_variant(
    *,
    source: Path,
    variant: str,
    iterations: int,
    warmups: int,
) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child-source",
            str(source),
            "--variant",
            variant,
            "--iterations",
            str(iterations),
            "--warmups",
            str(warmups),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def main() -> None:
    child_parser = _child_parser()
    child_args, _ = child_parser.parse_known_args()
    if child_args.child_source is not None:
        _run_child(child_args)
        return

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-checkout", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmups", type=int, default=5)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        parser.error("iterations must be positive and warmups must be non-negative")
    if not args.baseline_checkout.is_dir():
        parser.error("baseline checkout does not exist")

    current_checkout = Path(__file__).resolve().parents[1]
    baseline = _run_variant(
        source=args.baseline_checkout,
        variant="baseline_full",
        iterations=args.iterations,
        warmups=args.warmups,
    )
    batched_full = _run_variant(
        source=current_checkout,
        variant="batched_full",
        iterations=args.iterations,
        warmups=args.warmups,
    )
    batched_ids = _run_variant(
        source=current_checkout,
        variant="batched_ids",
        iterations=args.iterations,
        warmups=args.warmups,
    )
    baseline_median = float(baseline["median_ms_per_publication"])
    print(
        json.dumps(
            {
                "benchmark": "detection_publication",
                "baseline": baseline,
                "batched_full": batched_full,
                "batched_ids": batched_ids,
                "worker_median_improvement_percent": (
                    (baseline_median - float(batched_ids["median_ms_per_publication"]))
                    / baseline_median
                    * 100
                ),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
