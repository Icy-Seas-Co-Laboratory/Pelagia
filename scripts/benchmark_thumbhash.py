#!/usr/bin/env python3
"""Measure the complete ThumbHash path and its Python-to-Rust encoder call.

Run with .venv/bin/python scripts/benchmark_thumbhash.py --input frame.png.
Input decoding and synthetic image construction are excluded from the timings.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from Pelagia.processing import thumbhash


def positive_int(value: str) -> int:
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Use a real image instead of seeded synthetic data.")
    parser.add_argument("--width", type=positive_int, default=4000)
    parser.add_argument("--height", type=positive_int, default=3000)
    parser.add_argument("--channels", type=int, choices=(1, 3, 4), default=1)
    parser.add_argument("--dtype", choices=("uint8", "uint16"), default="uint8")
    parser.add_argument("--max-dim", type=int, choices=range(1, 101), default=100)
    parser.add_argument("--iterations", type=positive_int, default=100)
    parser.add_argument("--threads", type=positive_int, default=1, help="OpenCV thread count.")
    args = parser.parse_args()
    cv2.setNumThreads(args.threads)
    if args.input:
        image = cv2.imread(str(args.input), cv2.IMREAD_UNCHANGED)
        if image is None:
            parser.error(f"cannot decode {args.input}")
    else:
        shape = (args.height, args.width)
        if args.channels != 1:
            shape += (args.channels,)
        dtype = np.dtype(args.dtype)
        image = np.random.default_rng(42).integers(
            0, np.iinfo(dtype).max + 1, size=shape, dtype=dtype,
        )
        if args.channels == 4:
            image[:, :, 3] = np.iinfo(dtype).max

    # Capture exactly the encoder input once, outside the timed path. The complete
    # measurement below uses the real encoder without instrumentation overhead.
    encoder = thumbhash.rgba_to_thumb_hash
    captured = None

    def capture(width: int, height: int, rgba: bytes) -> bytes:
        nonlocal captured
        captured = (width, height, rgba)
        return encoder(width, height, rgba)

    thumbhash.rgba_to_thumb_hash = capture
    try:
        payload = thumbhash.compute_thumbhash(image, max_dim=args.max_dim)
    finally:
        thumbhash.rgba_to_thumb_hash = encoder
    assert captured is not None

    def measure(function):
        for _ in range(5):
            function()
        samples = []
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            function()
            samples.append((time.perf_counter_ns() - start) / 1_000_000)
        return {
            "median_ms": statistics.median(samples),
            "p95_ms": float(np.percentile(samples, 95)),
            "frames_per_second": 1000 / statistics.mean(samples),
        }

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "encoder": "fast-thumbhash-rs",
        "encoder_version": importlib.metadata.version("fast-thumbhash-rs"),
        "opencv_version": cv2.__version__,
        "opencv_threads_requested": args.threads,
        "opencv_threads": cv2.getNumThreads(),
        "shape": image.shape,
        "dtype": str(image.dtype),
        "max_dim": args.max_dim,
        "iterations": args.iterations,
        "preview_shape": [captured[1], captured[0], 4],
        "hash_hex": payload.hex(),
        "encoder_call": measure(lambda: encoder(*captured)),
        "complete_thumbhash": measure(
            lambda: thumbhash.compute_thumbhash(image, max_dim=args.max_dim)
        ),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
