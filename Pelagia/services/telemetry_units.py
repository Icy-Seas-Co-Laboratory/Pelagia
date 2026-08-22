"""Explicit, dependency-free unit validation for telemetry ingestion.

The registry deliberately covers a small set of operational oceanographic
units.  It is not a general dimensional-analysis engine: a new unit or domain
must be added here with its conversion and reviewed provenance rather than
being accepted as an arbitrary string.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable


@dataclass(frozen=True, slots=True)
class TelemetryUnit:
    """A unit expressed as an affine transform to its domain reference unit."""

    canonical_unit: str
    dimension: str
    scale_to_reference: float = 1.0
    offset_to_reference: float = 0.0
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AffineUnitConversion:
    """``target_value = source_value * scale + offset``."""

    native_unit: str
    canonical_unit: str
    scale: float
    offset: float


class TelemetryUnitRegistry:
    """Resolve known units and calculate only dimensionally valid transforms."""

    version = "1"

    def __init__(self, units: Iterable[TelemetryUnit]):
        self._catalog = tuple(units)
        self._units: dict[str, TelemetryUnit] = {}
        for unit in self._catalog:
            for name in (unit.canonical_unit, *unit.aliases):
                key = self._key(name)
                if key in self._units and self._units[key] != unit:
                    raise ValueError(f"Duplicate telemetry unit alias {name!r}.")
                self._units[key] = unit

    def catalog(self) -> list[dict[str, object]]:
        """Return the stable public vocabulary used by import clients."""
        return [
            {
                "canonical_unit": unit.canonical_unit,
                "dimension": unit.dimension,
                "aliases": list(unit.aliases),
                "scale_to_reference": unit.scale_to_reference,
                "offset_to_reference": unit.offset_to_reference,
            }
            for unit in self._catalog
        ]

    @staticmethod
    def _key(value: object) -> str:
        return str(value).strip().casefold()

    def resolve(self, value: object, *, field_name: str = "unit") -> TelemetryUnit:
        text = str(value).strip()
        if not text:
            raise ValueError(f"Telemetry {field_name} must not be blank.")
        unit = self._units.get(self._key(text))
        if unit is None:
            raise ValueError(
                f"Unsupported telemetry {field_name} {text!r}. "
                "Add it to the explicit telemetry unit registry before importing it."
            )
        return unit

    def conversion(self, native_unit: object, canonical_unit: object) -> AffineUnitConversion:
        native = self.resolve(native_unit, field_name="native unit")
        canonical = self.resolve(canonical_unit, field_name="canonical unit")
        if native.dimension != canonical.dimension:
            raise ValueError(
                f"Cannot convert telemetry unit {native.canonical_unit!r} "
                f"({native.dimension}) to {canonical.canonical_unit!r} "
                f"({canonical.dimension})."
            )
        scale = native.scale_to_reference / canonical.scale_to_reference
        offset = (
            native.offset_to_reference - canonical.offset_to_reference
        ) / canonical.scale_to_reference
        return AffineUnitConversion(
            native_unit=native.canonical_unit,
            canonical_unit=canonical.canonical_unit,
            scale=scale,
            offset=offset,
        )

    def validate_affine_conversion(
        self,
        native_unit: object,
        canonical_unit: object,
        *,
        scale: object,
        offset: object,
    ) -> AffineUnitConversion:
        """Validate a declared mapping transform against the registered transform."""
        try:
            declared_scale = float(scale)
            declared_offset = float(offset)
        except (TypeError, ValueError) as exc:
            raise ValueError("Telemetry conversion scale and offset must be numeric.") from exc
        if not math.isfinite(declared_scale) or not math.isfinite(declared_offset):
            raise ValueError("Telemetry conversion scale and offset must be finite.")

        conversion = self.conversion(native_unit, canonical_unit)
        if not (
            math.isclose(declared_scale, conversion.scale, rel_tol=1e-12, abs_tol=1e-12)
            and math.isclose(declared_offset, conversion.offset, rel_tol=1e-12, abs_tol=1e-12)
        ):
            raise ValueError(
                f"Telemetry conversion from {conversion.native_unit!r} to "
                f"{conversion.canonical_unit!r} must use scale={conversion.scale!r} "
                f"and offset={conversion.offset!r}; got scale={declared_scale!r} "
                f"and offset={declared_offset!r}."
            )
        return conversion


# The reference unit for every domain has scale 1 and offset 0.  Temperature
# uses kelvin as its reference, which retains the affine part of conversions.
DEFAULT_TELEMETRY_UNIT_REGISTRY = TelemetryUnitRegistry(
    (
        TelemetryUnit("K", "temperature", aliases=("kelvin",)),
        TelemetryUnit("degC", "temperature", 1.0, 273.15, ("°C", "celsius")),
        TelemetryUnit("degF", "temperature", 5.0 / 9.0, 255.3722222222222, ("°F", "fahrenheit")),
        TelemetryUnit("Pa", "pressure", aliases=("pascal", "pascals")),
        TelemetryUnit("hPa", "pressure", 100.0),
        TelemetryUnit("kPa", "pressure", 1_000.0),
        TelemetryUnit("bar", "pressure", 100_000.0),
        TelemetryUnit("dbar", "pressure", 10_000.0),
        TelemetryUnit("m", "length", aliases=("metre", "meter", "metres", "meters")),
        TelemetryUnit("cm", "length", 0.01),
        TelemetryUnit("mm", "length", 0.001),
        TelemetryUnit("km", "length", 1_000.0),
        TelemetryUnit("ft", "length", 0.3048, ("foot", "feet")),
        TelemetryUnit("m/s", "speed", aliases=("m s-1", "m s^-1")),
        TelemetryUnit("km/h", "speed", 1_000.0 / 3_600.0),
        TelemetryUnit("kn", "speed", 0.5144444444444445, ("knot", "knots", "kt", "kts")),
        TelemetryUnit("S/m", "conductivity", aliases=("siemens/m",)),
        TelemetryUnit("mS/cm", "conductivity", 0.1),
        TelemetryUnit("uS/cm", "conductivity", 0.0001, ("µS/cm",)),
        TelemetryUnit("V", "electric_potential", aliases=("volt", "volts")),
        TelemetryUnit("mV", "electric_potential", 0.001),
        TelemetryUnit("1", "fraction", aliases=("dimensionless",)),
        TelemetryUnit("%", "fraction", 0.01, aliases=("percent", "percentage")),
        TelemetryUnit("PSU", "practical_salinity"),
        TelemetryUnit("NTU", "turbidity"),
        TelemetryUnit("count", "count", aliases=("counts",)),
        TelemetryUnit("mg/L", "mass_concentration", aliases=("mg l-1",)),
        TelemetryUnit("ug/L", "mass_concentration", 0.001, ("µg/L", "ug l-1")),
        TelemetryUnit("g/m3", "mass_concentration"),
        TelemetryUnit("umol/L", "amount_concentration", aliases=("µmol/L", "umol l-1")),
        TelemetryUnit("mmol/m3", "amount_concentration"),
    )
)
