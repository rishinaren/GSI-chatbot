"""Simple unit extraction and conversion helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass

MEASUREMENT_RE = re.compile(
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|in\.?|ft|kg|g|lb|kPa|MPa|psi|%|°C|C|°F|F)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Measurement:
    value: float
    unit: str
    text: str


def extract_measurements(text: str) -> list[Measurement]:
    measurements: list[Measurement] = []
    for match in MEASUREMENT_RE.finditer(text):
        unit = _normalize_unit(match.group("unit"))
        measurements.append(
            Measurement(value=float(match.group("value")), unit=unit, text=match.group(0))
        )
    return measurements


def convert_measurement(measurement: Measurement, preferred_system: str) -> Measurement | None:
    """Convert common engineering units for user-facing summaries."""

    preferred = preferred_system.lower()
    value = measurement.value
    unit = measurement.unit

    if preferred in {"imperial", "us", "english"}:
        if unit == "mm":
            return Measurement(value / 25.4, "in", measurement.text)
        if unit == "cm":
            return Measurement(value / 2.54, "in", measurement.text)
        if unit == "m":
            return Measurement(value * 3.28084, "ft", measurement.text)
        if unit == "kPa":
            return Measurement(value / 6.89476, "psi", measurement.text)
        if unit == "MPa":
            return Measurement(value * 145.038, "psi", measurement.text)
        if unit == "kg":
            return Measurement(value * 2.20462, "lb", measurement.text)
        if unit == "C":
            return Measurement((value * 9 / 5) + 32, "F", measurement.text)

    if preferred in {"metric", "si"}:
        if unit == "in":
            return Measurement(value * 25.4, "mm", measurement.text)
        if unit == "ft":
            return Measurement(value / 3.28084, "m", measurement.text)
        if unit == "psi":
            return Measurement(value * 6.89476, "kPa", measurement.text)
        if unit == "lb":
            return Measurement(value / 2.20462, "kg", measurement.text)
        if unit == "F":
            return Measurement((value - 32) * 5 / 9, "C", measurement.text)

    return None


def format_conversion(original: Measurement, converted: Measurement) -> str:
    return f"{original.text} is approximately {converted.value:.2f} {converted.unit}"


def _normalize_unit(unit: str) -> str:
    cleaned = unit.replace(".", "").replace("°", "")
    if cleaned.lower() in {"c"}:
        return "C"
    if cleaned.lower() in {"f"}:
        return "F"
    return cleaned
