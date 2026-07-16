"""Collect low-overhead leaf-phase timings across nested processing calls."""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, Callable, Iterator, TypeVar, cast


@dataclass
class PhaseTimingCollector:
    durations_ms: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))

    def record(self, phase: str, duration_ms: float) -> None:
        self.durations_ms[phase] += duration_ms
        self.counts[phase] += 1

    def snapshot(self, *, total_ms: float, unit_count: int) -> dict[str, Any]:
        phases_ms = {
            phase: round(duration_ms, 3)
            for phase, duration_ms in sorted(self.durations_ms.items())
        }
        measured_ms = sum(self.durations_ms.values())
        return {
            "schema_version": 1,
            "total_ms": round(total_ms, 3),
            "measured_ms": round(measured_ms, 3),
            "unattributed_ms": round(max(0.0, total_ms - measured_ms), 3),
            "unit_count": unit_count,
            "average_unit_ms": round(total_ms / unit_count, 3) if unit_count else None,
            "phases_ms": phases_ms,
            "phase_counts": dict(sorted(self.counts.items())),
            "average_phase_ms": {
                phase: round(self.durations_ms[phase] / count, 3)
                for phase, count in sorted(self.counts.items())
                if count
            },
            "phase_percent": {
                phase: round(duration_ms / total_ms * 100.0, 2)
                for phase, duration_ms in sorted(self.durations_ms.items())
                if total_ms > 0
            },
        }


# Context-local collection lets deeply nested processing code report leaf timings
# without threading a metrics argument through every image-processing API.
_ACTIVE_COLLECTOR: ContextVar[PhaseTimingCollector | None] = ContextVar(
    "pelagia_phase_timing_collector",
    default=None,
)


@contextmanager
def measure_phase(phase: str) -> Iterator[None]:
    collector = _ACTIVE_COLLECTOR.get()
    if collector is None:
        yield
        return
    started = time.perf_counter()
    try:
        yield
    finally:
        collector.record(phase, (time.perf_counter() - started) * 1000)


_Result = TypeVar("_Result", bound=dict[str, Any])


def collect_result_timings(
    *,
    result_key: str = "timings",
    unit_count_key: str = "frame_count",
) -> Callable[[Callable[..., _Result]], Callable[..., _Result]]:
    def decorator(function: Callable[..., _Result]) -> Callable[..., _Result]:
        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> _Result:
            collector = PhaseTimingCollector()
            token = _ACTIVE_COLLECTOR.set(collector)
            started = time.perf_counter()
            try:
                result = function(*args, **kwargs)
            finally:
                _ACTIVE_COLLECTOR.reset(token)
            unit_count = int(result.get(unit_count_key) or 0)
            result[result_key] = collector.snapshot(
                total_ms=(time.perf_counter() - started) * 1000,
                unit_count=unit_count,
            )
            return result

        return cast(Callable[..., _Result], wrapper)

    return decorator
