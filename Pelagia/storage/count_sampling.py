"""Block sampling for informational queue previews; selection stays exact."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CountSample:
    fraction: float = 1.0

    @property
    def sql(self) -> str:
        if self.fraction == 1:
            return ""
        # Only an internally computed finite number is interpolated. A stable
        # seed avoids count jitter between polls of unchanged physical data.
        return f" TABLESAMPLE SYSTEM ({self.fraction * 100:.12g}) REPEATABLE (0)"

    def count(self, value: Any) -> int:
        return round(int(value or 0) / self.fraction)

    def metadata(self, matching_rows: int) -> dict[str, Any]:
        return {
            "exact": self.fraction == 1,
            "method": "exact" if self.fraction == 1 else "block_sample",
            "sample_fraction": self.fraction,
            "matching_sample_rows": matching_rows,
        }


def count_sample(cursor, schema: str, table: str, mode: str) -> CountSample:
    if mode not in {"exact", "auto"}:
        raise ValueError("count_mode must be exact or auto.")
    if mode == "exact":
        return CountSample()
    # Physical size remains useful when ANALYZE statistics are stale. Sampling
    # targets 256 heap pages (normally 2 MiB); it is an expected work budget,
    # not a hard limit or a statistical confidence guarantee.
    cursor.execute(
        "SELECT pg_relation_size(%s::regclass) / current_setting('block_size')::integer AS pages",
        (f"{schema}.{table}",),
    )
    pages = int((cursor.fetchone() or {}).get("pages") or 0)
    return CountSample(256 / pages if pages > 1024 else 1.0)
