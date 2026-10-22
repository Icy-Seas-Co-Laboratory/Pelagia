#!/usr/bin/env python3
"""Deterministic v1/v2/v3 heuristic ROI refinement benchmark.

The benchmark uses synthetic crops with known object masks.  It reports median
and p95 runtime, IoU/recall/leakage, and the candidate-mask growth invariant.
Run from the repository root, for example::

    .venv/bin/python scripts/benchmark_heuristic_refinement.py --iterations 30
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from Pelagia.processing.heuristic_refinement import (
    HeuristicEdgeRefinementParameters,
    heuristic_edge_refine,
)
from Pelagia.processing.heuristic_refinement_v2 import HeuristicV2RefinementParameters, heuristic_v2_refine
from Pelagia.processing.heuristic_refinement_v3 import HeuristicV3RefinementParameters, heuristic_v3_refine


def _scenarios() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
    shape = (128, 128)
    yy, xx = np.indices(shape)
    scenarios: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]] = {}

    # Ground truth includes a dim halo beyond the bright candidate.
    image = np.zeros(shape, np.uint8)
    truth = ((xx - 64) ** 2 / 27**2 + (yy - 64) ** 2 / 20**2 <= 1)
    image[truth] = 90
    core = ((xx - 64) ** 2 / 16**2 + (yy - 64) ** 2 / 12**2 <= 1)
    image[core] = 150
    scenarios["dim_halo"] = (image, core, truth, None)

    # An internal intensity split should not terminate expansion.
    image = np.zeros(shape, np.uint8)
    truth = ((xx - 62) ** 2 / 24**2 + (yy - 65) ** 2 / 18**2 <= 1)
    image[truth] = 105
    image[truth & (xx < 62)] = 145
    core = truth & (xx < 55) & (xx > 42)
    scenarios["internal_gradient"] = (image, core, truth, None)

    # Diffuse low contrast background tests the stability penalty.
    image = np.full(shape, 16, np.uint8)
    diffuse = ((xx - 62) ** 2 / 28**2 + (yy - 65) ** 2 / 21**2 <= 1)
    truth = ((xx - 62) ** 2 / 21**2 + (yy - 65) ** 2 / 15**2 <= 1)
    image[diffuse] = 28
    image[truth] = 105
    core = ((xx - 62) ** 2 / 13**2 + (yy - 65) ** 2 / 10**2 <= 1)
    image[core] = 150
    scenarios["diffuse_background"] = (image, core, truth, None)

    # Two objects are separated by a dark valley; the second is leakage.
    image = np.zeros(shape, np.uint8)
    left = ((xx - 45) ** 2 / 22**2 + (yy - 64) ** 2 / 18**2 <= 1)
    right = ((xx - 88) ** 2 / 17**2 + (yy - 64) ** 2 / 15**2 <= 1)
    image[left] = 110
    image[right] = 130
    core = ((xx - 45) ** 2 / 12**2 + (yy - 64) ** 2 / 10**2 <= 1)
    image[core] = 160
    scenarios["neighboring_object"] = (image, core, left, right)

    # Uniform background has no defensible boundary, so only the seed is valid.
    image = np.full(shape, 25, np.uint8)
    truth = np.zeros(shape, bool)
    core = np.zeros(shape, bool)
    core[59:69, 59:69] = True
    truth[:] = core
    image[core] = 150
    scenarios["no_justified_growth"] = (image, core, truth, None)

    # A dim upper half and diffuse lower background require different local
    # boundary decisions; no single threshold recovers both cleanly.
    image = np.zeros(shape, np.uint8)
    truth = ((xx - 64) ** 2 / 25**2 + (yy - 64) ** 2 / 19**2 <= 1)
    diffuse = ((xx - 64) ** 2 / 33**2 + (yy - 64) ** 2 / 25**2 <= 1)
    image[diffuse & (yy >= 64)] = 30
    image[truth & (yy < 64)] = 35
    image[truth & (yy >= 64)] = 100
    core = ((xx - 64) ** 2 / 11**2 + (yy - 64) ** 2 / 8**2 <= 1)
    image[core] = 155
    scenarios["heterogeneous_boundary"] = (image, core, truth, None)
    return scenarios


def _metrics(mask: np.ndarray, seed: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    refined = np.asarray(mask) > 0
    seed = np.asarray(seed) > 0
    intersection = np.count_nonzero(refined & truth)
    union = np.count_nonzero(refined | truth)
    return {
        "iou": float(intersection / union) if union else 1.0,
        "recall": float(intersection / np.count_nonzero(truth)) if np.any(truth) else 1.0,
        "leakage_pixels": int(np.count_nonzero(refined & ~truth)),
        "growth_pixels": int(np.count_nonzero(refined & ~seed)),
        "growth_invariant": bool(np.all(refined[seed])),
        "foreground_pixels": int(np.count_nonzero(refined)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmups < 0:
        parser.error("iterations must be positive and warmups must be non-negative")
    cv2.setNumThreads(1)
    growth_radius = 32
    params = HeuristicEdgeRefinementParameters(max_growth_pixels=growth_radius)
    v2_params = HeuristicV2RefinementParameters(max_growth_pixels=growth_radius)
    v3_params = HeuristicV3RefinementParameters(max_growth_pixels=growth_radius)
    report: dict[str, Any] = {"opencv_threads": 1, "iterations": args.iterations, "warmups": args.warmups, "max_growth_pixels": growth_radius, "scenarios": {}}
    for name, (image, seed, truth, competitor) in _scenarios().items():
        inputs = (image, seed)
        functions = {"heuristic_v1": lambda: heuristic_edge_refine(*inputs, parameters=params)[0],
                     "heuristic_v2": lambda: heuristic_v2_refine(
                         *inputs, competing_seed_mask=competitor, parameters=v2_params)[0],
                     "heuristic_v3": lambda: heuristic_v3_refine(
                         *inputs, competing_seed_mask=competitor, parameters=v3_params)[0]}
        result: dict[str, Any] = {"shape": list(image.shape)}
        for label, function in functions.items():
            for _ in range(args.warmups):
                output = function()
            samples: list[float] = []
            for _ in range(args.iterations):
                start = time.perf_counter_ns()
                output = function()
                samples.append((time.perf_counter_ns() - start) / 1e6)
            result[label] = {"median_ms": statistics.median(samples),
                             "p95_ms": float(np.percentile(samples, 95)),
                             "metrics": _metrics(output, seed, truth)}
        report["scenarios"][name] = result
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
