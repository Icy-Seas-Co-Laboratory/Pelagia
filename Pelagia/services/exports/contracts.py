"""Stable contracts shared by export API, workers, and product writers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

EXPORT_BUNDLE_FORMAT_VERSION = "1.0"


class ExportProduct(str, Enum):
    RAW_ROI_STATISTICS = "raw_roi_statistics"
    BINNED_ROI_STATISTICS = "binned_roi_statistics"
    ROI_EVIDENCE = "roi_evidence"
    TELEMETRY = "telemetry"


class TabularFormat(str, Enum):
    XLSX = "xlsx"
    JSON = "json"
    SQLITE = "sqlite"


@dataclass(frozen=True, slots=True)
class ExportRequest:
    """Normalized, persistable request; IDs are always scoped by its project."""

    products: tuple[ExportProduct, ...]
    formats: dict[str, str] = field(default_factory=dict)
    asset_ids: tuple[str, ...] = ()
    run_ids: tuple[str, ...] = ()
    telemetry_source_ids: tuple[str, ...] = ()
    roi_stage: str = "refined"
    filters: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.products:
            raise ValueError("An export must select at least one product.")
        if len(set(self.products)) != len(self.products):
            raise ValueError("Export products must not be repeated.")
        if self.roi_stage != "refined":
            raise ValueError("The first export release supports refined ROIs only.")
        invalid = set(self.formats.values()) - {item.value for item in TabularFormat}
        if invalid:
            raise ValueError(f"Unsupported export format(s): {', '.join(sorted(invalid))}.")

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["products"] = [product.value for product in self.products]
        return value
