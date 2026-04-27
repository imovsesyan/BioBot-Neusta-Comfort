"""Tests for F13 plant species identification, health assessment, and irrigation.

Coverage
--------
1. ``test_species_identifier_mock_mode`` — mock=True returns a valid SpeciesResult
   without any HTTP call.
2. ``test_species_identifier_low_confidence_flag`` — SpeciesResult with confidence
   below 0.4 automatically sets is_low_confidence=True.
3. ``test_species_identifier_no_api_key_raises`` — live mode without PLANTNET_API_KEY
   raises ValueError.
4. ``test_health_assessor_heat_stress`` — humidex=45 → critical, stress_cause=heat.
5. ``test_health_assessor_healthy`` — normal benign conditions → healthy, cause=None.
6. ``test_health_assessor_drought_mesic`` — soil_moisture=0.10, mesic plant → stressed
   or critical drought.
7. ``test_irrigation_advice_urgent_at_dangerous`` — dangerous risk level → urgent action.
8. ``test_irrigation_advice_morning_filter`` — hour=12, high_risk → peak-heat avoidance
   note in reason.
9. ``test_all_risk_levels_have_advice`` — every canonical risk level returns a
   non-empty action string.
"""

from __future__ import annotations

import os

import pytest

from biobot.plants.species_identifier import SpeciesResult, identify_species
from biobot.plants.health_assessor import HealthAssessment, assess_health
from biobot.plants.irrigation_recommender import (
    VALID_RISK_LEVELS,
    IrrigationAdvice,
    get_irrigation_advice,
)


# ---------------------------------------------------------------------------
# Species identifier
# ---------------------------------------------------------------------------


def test_species_identifier_mock_mode():
    """mock=True must return a valid SpeciesResult without issuing any HTTP call."""
    result = identify_species(image_path="irrelevant.jpg", mock=True)

    assert isinstance(result, SpeciesResult)
    assert isinstance(result.species_name, str) and result.species_name != ""
    assert isinstance(result.common_name, str) and result.common_name != ""
    assert 0.0 <= result.confidence <= 1.0
    assert isinstance(result.family, str) and result.family != ""
    assert isinstance(result.is_low_confidence, bool)


def test_species_identifier_low_confidence_flag():
    """SpeciesResult.is_low_confidence must be True when confidence < 0.4."""
    low = SpeciesResult(
        species_name="Some plant",
        common_name="Plant",
        confidence=0.25,
        family="Araceae",
    )
    assert low.is_low_confidence is True

    high = SpeciesResult(
        species_name="Some plant",
        common_name="Plant",
        confidence=0.75,
        family="Araceae",
    )
    assert high.is_low_confidence is False

    # Edge: exactly 0.4 is NOT low confidence (threshold is strictly < 0.4).
    boundary = SpeciesResult(
        species_name="Some plant",
        common_name="Plant",
        confidence=0.4,
        family="Araceae",
    )
    assert boundary.is_low_confidence is False


def test_species_identifier_no_api_key_raises(monkeypatch):
    """Live mode without PLANTNET_API_KEY set must raise ValueError."""
    monkeypatch.delenv("PLANTNET_API_KEY", raising=False)
    assert os.environ.get("PLANTNET_API_KEY") is None

    with pytest.raises(ValueError, match="API key"):
        identify_species(image_path="any_image.jpg", api_key=None, mock=False)


# ---------------------------------------------------------------------------
# Health assessor
# ---------------------------------------------------------------------------


def test_health_assessor_heat_stress():
    """humidex >= 45 (F10 critical threshold) must produce critical + heat."""
    result = assess_health(
        humidex=45.0,
        temperature_c=35.0,
        relative_humidity=60.0,
    )

    assert isinstance(result, HealthAssessment)
    assert result.health_status == "critical"
    assert result.stress_cause == "heat"
    assert result.confidence == "high"
    assert result.notes != ""


def test_health_assessor_healthy():
    """Mild, comfortable conditions with adequate humidity must return healthy."""
    result = assess_health(
        humidex=22.0,
        temperature_c=20.0,
        relative_humidity=55.0,
    )

    assert isinstance(result, HealthAssessment)
    assert result.health_status == "healthy"
    assert result.stress_cause is None
    assert result.confidence == "high"


def test_health_assessor_drought_mesic():
    """soil_moisture=0.10 for a mesic plant falls below the stressed threshold (0.25)."""
    result = assess_health(
        humidex=25.0,
        temperature_c=22.0,
        relative_humidity=55.0,
        soil_moisture=0.10,
        species_water_need="mesic",
    )

    assert isinstance(result, HealthAssessment)
    # 0.10 < 0.15 (mesic critical threshold) → critical drought
    assert result.health_status in ("stressed", "critical")
    assert result.stress_cause == "drought"


def test_health_assessor_mesic_critical_drought():
    """soil_moisture=0.10 for mesic is below the critical threshold (0.15) → critical."""
    result = assess_health(
        humidex=25.0,
        temperature_c=22.0,
        relative_humidity=55.0,
        soil_moisture=0.10,
        species_water_need="mesic",
    )
    # 0.10 < 0.15 (mesic critical_thr) so this must be critical
    assert result.health_status == "critical"
    assert result.stress_cause == "drought"


def test_health_assessor_invalid_water_need():
    """An unknown species_water_need must raise ValueError."""
    with pytest.raises(ValueError, match="species_water_need"):
        assess_health(
            humidex=25.0,
            temperature_c=20.0,
            relative_humidity=50.0,
            species_water_need="desert",
        )


# ---------------------------------------------------------------------------
# Irrigation recommender
# ---------------------------------------------------------------------------


def _healthy_assessment() -> HealthAssessment:
    """Return a healthy assessment for use in irrigation tests."""
    return assess_health(
        humidex=22.0,
        temperature_c=20.0,
        relative_humidity=55.0,
    )


def _drought_assessment() -> HealthAssessment:
    """Return a critical drought assessment for use in irrigation tests."""
    return assess_health(
        humidex=22.0,
        temperature_c=20.0,
        relative_humidity=55.0,
        soil_moisture=0.05,
        species_water_need="mesic",
    )


def test_irrigation_advice_urgent_at_dangerous():
    """dangerous risk level must produce action=urgent regardless of drought state."""
    advice = get_irrigation_advice(
        humidex_risk_level="dangerous",
        health_assessment=_healthy_assessment(),
    )

    assert isinstance(advice, IrrigationAdvice)
    assert advice.action == "urgent"
    assert advice.ml_improvement_note != ""


def test_irrigation_advice_urgent_at_critical():
    """critical risk level must also produce action=urgent."""
    advice = get_irrigation_advice(
        humidex_risk_level="critical",
        health_assessment=_healthy_assessment(),
    )
    assert advice.action == "urgent"


def test_irrigation_advice_morning_filter():
    """hour=12 + high_risk must append the peak-heat avoidance note to the reason."""
    advice = get_irrigation_advice(
        humidex_risk_level="high_risk",
        health_assessment=_healthy_assessment(),
        hour=12,
    )

    assert isinstance(advice, IrrigationAdvice)
    assert "peak heat" in advice.reason.lower() or "avoid watering" in advice.reason.lower()


def test_irrigation_advice_morning_filter_no_append_outside_peak():
    """hour=8 (before peak) + high_risk must NOT append the avoidance note."""
    advice = get_irrigation_advice(
        humidex_risk_level="high_risk",
        health_assessment=_healthy_assessment(),
        hour=8,
    )
    assert "avoid watering" not in advice.reason.lower()


def test_all_risk_levels_have_advice():
    """Every canonical risk level must return a non-empty action string."""
    health = _healthy_assessment()

    for risk_level in VALID_RISK_LEVELS:
        advice = get_irrigation_advice(
            humidex_risk_level=risk_level,
            health_assessment=health,
        )
        assert isinstance(advice, IrrigationAdvice)
        assert advice.action.strip() != "", f"Empty action for risk_level={risk_level}"
        assert advice.suggested_time_slot.strip() != ""
        assert advice.reason.strip() != ""
        assert advice.ml_improvement_note.strip() != ""


def test_irrigation_advice_unknown_risk_raises():
    """An unknown risk level must raise ValueError."""
    with pytest.raises(ValueError, match="humidex_risk_level"):
        get_irrigation_advice(
            humidex_risk_level="extreme",
            health_assessment=_healthy_assessment(),
        )
