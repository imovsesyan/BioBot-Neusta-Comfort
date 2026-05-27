"""
Irrigation recommendation service for BioSense360.

Calculates water volume, timing, and frequency for plant irrigation based on
current Humidex conditions, plant type, and soil type.

Rules derived from:
  - OHCOW 2022 Humidex thresholds (heat stress on vegetation)
  - Agronomic soil-water retention coefficients
  - Standard irrigation horticultural practices
"""

from typing import Optional, Tuple
from backend.schemas.recommendations import IrrigationRequest, IrrigationResponse
from backend.services.humidex_service import humidex_to_risk_level


# ---------------------------------------------------------------------------
# Multiplier tables
# ---------------------------------------------------------------------------

# Plant water demand multiplier relative to baseline (Flowers = 1.0)
PLANT_MULTIPLIERS: dict[str, float] = {
    "Vegetables": 1.5,
    "Crops":      2.0,
    "Flowers":    1.0,
    "Grass":      0.8,
}

# Soil water retention: Sandy drains fast (needs more water), Clay retains well
SOIL_MULTIPLIERS: dict[str, float] = {
    "Sandy": 1.5,
    "Loamy": 1.0,
    "Clay":  0.7,
}

# Base water volume (litres per m²) per irrigation event at each risk level
BASE_WATER_LITRES: dict[str, float] = {
    "LOW":      1.0,
    "MODERATE": 1.5,
    "HIGH":     2.5,
    "CRITICAL": 4.0,
}

# Best watering time slots per risk level
BEST_SLOTS: dict[str, list[str]] = {
    "LOW":      ["07:00–09:00", "18:00–20:00"],
    "MODERATE": ["06:00–08:00", "19:00–21:00"],
    "HIGH":     ["05:00–07:00", "20:00–22:00"],
    "CRITICAL": ["05:00–06:30", "21:00–22:30"],
}

# Irrigation frequency description per risk level
FREQUENCY_LABELS: dict[str, str] = {
    "LOW":      "once daily",
    "MODERATE": "once daily (morning preferred)",
    "HIGH":     "twice daily (early morning and evening)",
    "CRITICAL": "three times daily (emergency schedule)",
}

# Human-readable reason per risk level
REASONS: dict[str, str] = {
    "LOW": (
        "Conditions are comfortable. Normal irrigation schedule is sufficient. "
        "Water in the morning to reduce evaporation."
    ),
    "MODERATE": (
        "Humidex in the Caution range (30–39). Evapotranspiration is elevated. "
        "Increase frequency to once daily and restrict to morning hours to minimise "
        "water loss and leaf scorch."
    ),
    "HIGH": (
        "Humidex in the Extreme Caution range (40–45). Severe plant heat stress expected. "
        "Irrigate only in the coolest parts of the day — very early morning and late evening. "
        "Consider temporary shading for sensitive crops."
    ),
    "CRITICAL": (
        "Humidex above 45 — emergency plant conditions. Without immediate irrigation, "
        "irreversible crop damage and wilting will occur. Activate emergency schedule. "
        "Emergency shading and misting are strongly recommended."
    ),
}


def _high_humidity_reduction(humidity: float, base_volume: float) -> Tuple[float, Optional[str]]:
    """
    Reduce irrigation volume when ambient humidity is very high (> 80 %).
    High humidity suppresses evapotranspiration and raises mold/rot risk.

    Returns: (adjusted_volume, alert_message_or_None)
    """
    if humidity > 80.0:
        adjusted = round(base_volume * 0.60, 2)  # −40 %
        alert = (
            f"Relative humidity is {humidity:.0f}% — above 80%. "
            "Irrigation volume reduced by 40% to prevent waterlogging, root rot, and fungal disease."
        )
        return adjusted, alert
    return base_volume, None


def generate_irrigation_recommendation(
    req: IrrigationRequest,
    avg_humidex: float,
    avg_humidity: float,
) -> IrrigationResponse:
    """
    Generate an irrigation plan for the given conditions.

    Args:
        req:          The parsed request (date, station_id, plant_type, soil_type).
        avg_humidex:  Average Humidex for the requested date at the station.
        avg_humidity: Average relative humidity for the same period.

    Returns:
        IrrigationResponse with volume, slots, frequency, and reasoning.
    """
    risk_level = humidex_to_risk_level(avg_humidex)

    plant_mult = PLANT_MULTIPLIERS.get(req.plant_type, 1.0)
    soil_mult = SOIL_MULTIPLIERS.get(req.soil_type, 1.0)
    base_volume = BASE_WATER_LITRES[risk_level]
    raw_volume = round(base_volume * plant_mult * soil_mult, 2)

    # Apply high-humidity reduction if applicable
    final_volume, alert = _high_humidity_reduction(avg_humidity, raw_volume)

    # Determine whether irrigation is actually needed
    should_irrigate = avg_humidex >= 25.0 or avg_humidity < 70.0

    return IrrigationResponse(
        should_irrigate=should_irrigate,
        best_slots=BEST_SLOTS[risk_level],
        water_liters=final_volume,
        frequency=FREQUENCY_LABELS[risk_level],
        reason=REASONS[risk_level],
        alert=alert,
    )
