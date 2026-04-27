"""F13 sensor-based plant health assessment.

Derives plant stress signals entirely from environmental sensor data — no
photo model is involved here.  The PlantVillage CNN is documented as an
optional future enhancement; its in-the-wild accuracy on ornamentals is poor
(see F13_research_plants.md, Section 2.1).

Rule sources
------------
* Heat stress thresholds: F10 humidex bands from ``src/biobot/risk/rules.py``
  (Wahid et al., *Env. Exp. Botany*, 2007).
* Cold stress limits: RHS hardiness zone conventions for tropical/sub-tropical
  ornamentals.
* Drought stress triggers: FAO Irrigation Paper 56 (Allen et al., 1998)
  soil-moisture-to-available-water fraction relationships, mapped to three
  water-need classes (xeric / mesic / hydric).
* Humidity stress: Pennisi & van Iersel, *HortScience* (2012) — RH < 30%
  induces desiccation stress in most tropical ornamentals.
* VPD stress: Buck equation SVP approximation; thresholds from Grossiord et al.,
  *New Phytologist* (2020) — VPD > 2.0 kPa causes measurable transpiration
  stress; > 3.0 kPa is critical.

Design constraint
-----------------
This module does NOT consume ``vivabilite_binary_mean`` or any F10-UC4
classifier output (CLAUDE.md leakage guardrail).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

VALID_WATER_NEED_CLASSES = ("xeric", "mesic", "hydric")

# Soil moisture thresholds (volumetric water content, dimensionless 0–1)
# indexed as {water_need_class: (stressed_threshold, critical_threshold)}
_DROUGHT_THRESHOLDS: dict[str, tuple[float, float]] = {
    "xeric": (0.15, 0.08),
    "mesic": (0.25, 0.15),
    "hydric": (0.35, 0.25),
}

# VPD thresholds in kPa (Grossiord et al., 2020)
_VPD_STRESSED_KPA = 2.0
_VPD_CRITICAL_KPA = 3.0

# Humidex thresholds — taken verbatim from F10 risk/rules.py
_HUMIDEX_HEAT_STRESSED = 40.0
_HUMIDEX_HEAT_CRITICAL = 45.0

# Temperature thresholds (°C) for cold stress
_TEMP_COLD_STRESSED = 10.0
_TEMP_COLD_CRITICAL = 5.0

# Relative humidity below which humidity stress is triggered (%)
_RH_LOW_THRESHOLD = 30.0

_SEVERITY_ORDER = {"healthy": 0, "stressed": 1, "critical": 2}


@dataclass
class HealthAssessment:
    """Plant health assessment derived from sensor data."""

    health_status: str
    stress_cause: str | None
    confidence: str
    notes: str


def _compute_vpd(temperature_c: float, relative_humidity: float) -> float:
    """Compute Vapor Pressure Deficit in kPa using the Buck equation SVP approximation.

    SVP (kPa) = 0.6108 * exp(17.27 * T / (T + 237.3))
    VPD (kPa) = SVP * (1 - RH / 100)
    """
    svp = 0.6108 * math.exp(17.27 * temperature_c / (temperature_c + 237.3))
    return svp * (1.0 - relative_humidity / 100.0)


def _merge_severities(
    assessments: list[tuple[str, str | None, str]],
) -> tuple[str, str | None, str]:
    """Return the highest-severity (status, cause, confidence) from a list.

    Each element is ``(health_status, stress_cause, confidence)``.  When
    multiple stressors reach the same severity the first one in the list wins
    on the primary cause; all causes are accumulated by the caller for notes.
    """
    best = ("healthy", None, "high")
    for status, cause, confidence in assessments:
        if _SEVERITY_ORDER[status] > _SEVERITY_ORDER[best[0]]:
            best = (status, cause, confidence)
    return best


def assess_health(
    humidex: float,
    temperature_c: float,
    relative_humidity: float,
    soil_moisture: float | None = None,
    species_water_need: str = "mesic",
) -> HealthAssessment:
    """Assess plant health from environmental sensor readings.

    Parameters
    ----------
    humidex:
        Humidex index (°C equivalent) as produced by the Meteo France pipeline.
    temperature_c:
        Air temperature in degrees Celsius.
    relative_humidity:
        Relative humidity as a percentage (0–100).
    soil_moisture:
        Volumetric water content (dimensionless, 0–1) from an Aquacheck
        sensor.  When ``None``, drought-stress assessment is skipped.
    species_water_need:
        Water-need class of the plant: ``xeric``, ``mesic``, or ``hydric``.
        Gates the soil moisture stress thresholds (FAO-56).

    Returns
    -------
    HealthAssessment
        Deterministic assessment.  ``health_status`` is the worst condition
        found; ``notes`` lists all active stressors.
    """
    if species_water_need not in VALID_WATER_NEED_CLASSES:
        raise ValueError(
            f"Unknown species_water_need {species_water_need!r}. "
            f"Expected one of {VALID_WATER_NEED_CLASSES}."
        )

    active_stressors: list[tuple[str, str | None, str]] = []
    cause_labels: list[str] = []

    # --- Heat stress (F10 humidex bands) ---
    if humidex >= _HUMIDEX_HEAT_CRITICAL:
        active_stressors.append(("critical", "heat", "high"))
        cause_labels.append("heat (humidex critical)")
    elif humidex >= _HUMIDEX_HEAT_STRESSED:
        active_stressors.append(("stressed", "heat", "high"))
        cause_labels.append("heat (humidex elevated)")

    # --- Cold stress ---
    if temperature_c < _TEMP_COLD_CRITICAL:
        active_stressors.append(("critical", "cold", "medium"))
        cause_labels.append("cold (temperature critical)")
    elif temperature_c < _TEMP_COLD_STRESSED:
        active_stressors.append(("stressed", "cold", "medium"))
        cause_labels.append("cold (temperature low)")

    # --- Drought stress ---
    if soil_moisture is not None:
        stressed_thr, critical_thr = _DROUGHT_THRESHOLDS[species_water_need]
        if soil_moisture < critical_thr:
            active_stressors.append(("critical", "drought", "high"))
            cause_labels.append(f"drought (soil moisture {soil_moisture:.2f} < {critical_thr})")
        elif soil_moisture < stressed_thr:
            active_stressors.append(("stressed", "drought", "high"))
            cause_labels.append(f"drought (soil moisture {soil_moisture:.2f} < {stressed_thr})")

    # --- Humidity stress ---
    if relative_humidity < _RH_LOW_THRESHOLD:
        active_stressors.append(("stressed", "humidity", "medium"))
        cause_labels.append(f"humidity (RH {relative_humidity:.1f}% < 30%)")

    # --- VPD stress ---
    vpd = _compute_vpd(temperature_c, relative_humidity)
    if vpd > _VPD_CRITICAL_KPA:
        active_stressors.append(("critical", "humidity", "medium"))
        cause_labels.append(f"VPD critical ({vpd:.2f} kPa > {_VPD_CRITICAL_KPA})")
    elif vpd > _VPD_STRESSED_KPA:
        active_stressors.append(("stressed", "humidity", "medium"))
        cause_labels.append(f"VPD elevated ({vpd:.2f} kPa > {_VPD_STRESSED_KPA})")

    if not active_stressors:
        return HealthAssessment(
            health_status="healthy",
            stress_cause=None,
            confidence="high",
            notes="No stress conditions detected.",
        )

    dominant_status, dominant_cause, dominant_confidence = _merge_severities(active_stressors)

    notes = "Active stressors: " + "; ".join(cause_labels) + "."
    return HealthAssessment(
        health_status=dominant_status,
        stress_cause=dominant_cause,
        confidence=dominant_confidence,
        notes=notes,
    )
