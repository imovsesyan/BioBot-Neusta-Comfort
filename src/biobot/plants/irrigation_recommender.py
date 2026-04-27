"""F13 rule-based irrigation adviser.

Maps the combination of F10 humidex risk level × plant drought stress × optional
time-of-day to an irrigation action.  The rule table is taken from Section 3.3
of F13_research_plants.md (FAO-56 / university extension simplification).

A note on deferred ET₀ modeling
--------------------------------
The agronomic gold standard (FAO Penman-Monteith ET₀, FAO-56 Paper 56) requires
net radiation and wind speed data not available for a typical indoor BioBot
deployment.  Rule-based scheduling is the recommended approach until precision
agronomy is explicitly required (F13_research_plants.md, Section 3.2).

Design constraint
-----------------
This module does NOT consume ``vivabilite_binary_mean`` or any F10-UC4
classifier output (CLAUDE.md leakage guardrail).
"""

from __future__ import annotations

from dataclasses import dataclass

from biobot.plants.health_assessor import HealthAssessment

VALID_RISK_LEVELS = ("livable", "discomfort", "high_risk", "dangerous", "critical")

# Risk levels at which watering must shift to the coolest available window.
_URGENT_RISK_LEVELS = ("dangerous", "critical")
# Risk levels at which midday watering should be avoided.
_HIGH_HEAT_RISK_LEVELS = ("high_risk", "dangerous", "critical")

# Hour range (inclusive) considered peak heat.
_PEAK_HEAT_HOUR_START = 10
_PEAK_HEAT_HOUR_END = 17

_ML_IMPROVEMENT_NOTE = (
    "ML improvement possible: train on real soil moisture time-series + species response data"
)


@dataclass
class IrrigationAdvice:
    """Irrigation recommendation for a plant under current conditions."""

    action: str
    suggested_time_slot: str
    reason: str
    ml_improvement_note: str


def _is_drought_stressed(health_assessment: HealthAssessment) -> bool:
    """Return True when the assessment indicates any drought stress."""
    return health_assessment.stress_cause == "drought" or (
        health_assessment.health_status in ("stressed", "critical")
        and health_assessment.stress_cause == "drought"
    )


def _drought_active(health_assessment: HealthAssessment) -> bool:
    """True when drought is the primary or a contributing stress cause.

    Drought may be the dominant cause or may appear alongside other stressors
    whose severity outranked drought in the health assessment.  We check the
    notes for the drought token so the irrigation logic still fires correctly
    when, e.g., heat stress dominates but the soil is also dry.
    """
    if health_assessment.stress_cause == "drought":
        return True
    return "drought" in health_assessment.notes.lower()


def get_irrigation_advice(
    humidex_risk_level: str,
    health_assessment: HealthAssessment,
    species_water_need: str = "mesic",
    soil_moisture: float | None = None,
    hour: int | None = None,
) -> IrrigationAdvice:
    """Determine the irrigation action for given conditions.

    Parameters
    ----------
    humidex_risk_level:
        F10 risk level string — one of ``livable``, ``discomfort``,
        ``high_risk``, ``dangerous``, ``critical``.
    health_assessment:
        Output of ``assess_health()``.  The ``stress_cause`` and ``notes``
        fields are inspected to detect active drought stress.
    species_water_need:
        Water-need class: ``xeric``, ``mesic``, or ``hydric``.  Currently
        used for context in the reason string; future versions can gate
        volume recommendations.
    soil_moisture:
        Raw soil moisture reading (0–1) for inclusion in the reason text.
        Optional — does not change the action logic (that lives in the health
        assessment rules).
    hour:
        Current hour of day (0–23).  When provided and the risk level is
        high_risk or above and the hour falls within the peak-heat window
        (10–17 inclusive), a peak-heat avoidance note is appended to the
        reason.

    Returns
    -------
    IrrigationAdvice
        Deterministic irrigation recommendation.

    Raises
    ------
    ValueError
        If ``humidex_risk_level`` is not one of the five canonical values.
    """
    if humidex_risk_level not in VALID_RISK_LEVELS:
        raise ValueError(
            f"Unknown humidex_risk_level {humidex_risk_level!r}. "
            f"Expected one of {VALID_RISK_LEVELS}."
        )

    drought = _drought_active(health_assessment)
    risk = humidex_risk_level

    # --- Rule table (Section 3.3, F13_research_plants.md) ---
    if risk in _URGENT_RISK_LEVELS:
        if risk == "critical":
            action = "urgent"
            slot = "coolest available window — emergency"
            reason = (
                "Humidex risk is critical. Emergency irrigation at the coolest "
                "window is required regardless of soil moisture to prevent severe "
                "transpiration stress."
            )
        else:  # dangerous
            action = "urgent"
            slot = "coolest available window"
            reason = (
                "Humidex risk is dangerous. Irrigate at the coolest available "
                "window to counter extreme evapotranspiration."
            )

    elif risk == "high_risk":
        if drought:
            action = "water_now"
            slot = "earliest morning (before 09:00)"
            reason = (
                f"High humidex risk combined with drought stress ({species_water_need} plant). "
                f"Water as early as possible to offset heat-driven transpiration losses."
            )
        else:
            action = "delay"
            slot = "morning or evening"
            reason = (
                f"High humidex risk but soil moisture appears adequate for a "
                f"{species_water_need} plant. Monitor soil moisture; mist foliage "
                f"if VPD remains elevated."
            )

    elif risk == "discomfort":
        if drought:
            action = "water_now"
            slot = "morning (06-10)"
            reason = (
                f"Moderate heat discomfort with drought stress detected "
                f"({species_water_need} plant). Irrigate in the morning window "
                f"to maximise uptake before midday heat."
            )
        else:
            action = "delay"
            slot = "morning or evening"
            reason = (
                f"Thermal discomfort but no drought stress for a {species_water_need} "
                f"plant. Consider light irrigation if the plant is mesic or hydric and "
                f"the period of discomfort is prolonged."
            )

    else:  # livable
        if drought:
            action = "water_now"
            slot = "morning (06-10)"
            reason = (
                f"Comfortable conditions but drought stress is active for a "
                f"{species_water_need} plant. Irrigate at the next morning window."
            )
        else:
            action = "none"
            slot = "morning or evening"
            reason = (
                f"Comfortable humidex conditions and no drought stress for a "
                f"{species_water_need} plant. No irrigation needed at this time."
            )

    # --- Time-of-day filter ---
    if (
        hour is not None
        and _PEAK_HEAT_HOUR_START <= hour <= _PEAK_HEAT_HOUR_END
        and risk in _HIGH_HEAT_RISK_LEVELS
    ):
        reason += " Avoid watering during peak heat; wait for evening."

    # Append soil moisture context when provided.
    if soil_moisture is not None:
        reason += f" (Soil moisture reading: {soil_moisture:.2f}.)"

    return IrrigationAdvice(
        action=action,
        suggested_time_slot=slot,
        reason=reason,
        ml_improvement_note=_ML_IMPROVEMENT_NOTE,
    )
