#!/usr/bin/env python3
"""Benchmark deterministic candidate generation; decoding and DB I/O are excluded.

Use --checkout to compare an unchanged checkout with the working tree. Config loads
parse packaged defaults only, avoiding machine-specific configuration and secrets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import sys
import time
import tracemalloc
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkout', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--iterations', type=int, default=25)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.threads < 1:
        parser.error('iterations and threads must be positive')
    sys.path.insert(0, str(args.checkout.resolve()))
    from Pelagia.config import CoreConfig
    from Pelagia.processing.detection_candidate import segment_frame
    from Pelagia.processing.frame_model import FrameData
    from Pelagia.processing.timing import PhaseTimingCollector, _ACTIVE_COLLECTOR

    cv2.setNumThreads(args.threads)
    original_load = CoreConfig.load
    loads = 0

    def isolated_load(cls, *unused, **kwargs):
        nonlocal loads
        loads += 1
        return original_load(local_config_path=None, use_env=False)

    CoreConfig.load = classmethod(isolated_load)
    context = SimpleNamespace(config=CoreConfig())
    images = {}
    noisy = np.zeros((1024, 1024), np.uint8)
    for y in range(10, 810, 20):
        for x in range(10, 830, 20):
            noisy[y:y + 2, x:x + 2] = 200
    for i in range(7):
        cv2.ellipse(noisy, (60 + i * 140, 920), (30, 15), 25, 0, 360, 200, -1)
    images['many_rejected'] = noisy
    few = np.zeros_like(noisy)
    for i in range(7):
        cv2.ellipse(few, (80 + i * 140, 500), (35, 20), 25, 0, 360, 200, -1)
    images['few_components'] = few
    large = np.zeros_like(noisy)
    cv2.ellipse(large, (512, 512), (470, 310), 25, 0, 360, 200, -1)
    images['large_object'] = large
    dense = np.full_like(noisy, 200)
    dense[100:850:3, 100:850:3] = 0
    images['dense_mask'] = dense
    report = {'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                              'numpy': np.__version__, 'opencv': cv2.__version__,
                              'opencv_threads': args.threads},
              'iterations': args.iterations, 'scope': 'in-memory segmentation with raw ROI/mask encoding; no decoding, augmentation, preprocessing, or database',
              'scenarios': {}}
    for name, data in images.items():
        frame = FrameData(sourcePath='', filename='synthetic', frameNumber=1, data=data,
                          metadata={'run_id': '00000000-0000-0000-0000-000000000001',
                                    'frame_id': '00000000-0000-7000-8000-000000000042'})

        def run():
            return segment_frame(frame, context=context, threshold=1, apply_preprocessing=False,
                                 mask_augmentation_enabled=False, min_width=5, min_height=5,
                                 padding=0, roi_encoding='raw', mask_encoding='raw',
                                 store_roi_payload_min_width=0, store_roi_payload_min_height=0,
                                 store_roi_payload_min_width_plus_height=0)

        for _ in range(3):
            run()
        samples = []
        loads = 0
        for _ in range(args.iterations):
            start = time.perf_counter_ns()
            records = run()
            samples.append((time.perf_counter_ns() - start) / 1e6)
        loads_per_frame = loads / args.iterations
        collector = PhaseTimingCollector()
        token = _ACTIVE_COLLECTOR.set(collector)
        try:
            run()
        finally:
            _ACTIVE_COLLECTOR.reset(token)
        tracemalloc.start()
        run()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        digest = hashlib.sha256()
        for record in records:
            digest.update(str((record.roi_index, record.bbox_x, record.bbox_y, record.bbox_w,
                               record.bbox_h, record.area, record.perimeter)).encode())
            digest.update(record.roi_payload or b'')
            digest.update(record.mask_payload or b'')
        report['scenarios'][name] = {
            'shape': list(data.shape),
            'components': cv2.connectedComponents(data)[0] - 1,
            'detections': len(records), 'median_ms': statistics.median(samples),
            'mean_ms': statistics.mean(samples), 'p95_ms': float(np.percentile(samples, 95)),
            'frames_per_second': 1000 / statistics.mean(samples),
            'config_loads_per_frame': loads_per_frame, 'tracemalloc_peak_bytes': peak,
            'geometry_payload_sha256': digest.hexdigest(),
            'single_sample_phases_ms': dict(collector.durations_ms),
        }
    output = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(output)
    print(output)


if __name__ == '__main__':
    main()
