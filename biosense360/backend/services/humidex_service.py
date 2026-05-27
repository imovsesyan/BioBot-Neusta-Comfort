"""
Humidex computation and classification service.

Implements the Masterton & Richardson (1979) formula:
    e = 6.112 × 10^(7.5 × T / (237.7 + T)) × (RH / 100)
    Humidex = T + 0.5555 × (e − 10)

Where:
    T   = dry-bulb temperature in °C
    RH  = relative humidity in %
    e   = vapour pressure in hPa

Classification thresholds from OHCOW 2022.
"""

import math


def compute_humidex(temp_c: float, rh: float) -> float:
    """
    Compute the Humidex index from temperature and relative humidity.

    Args:
        temp_c: Dry-bulb air temperature in degrees Celsius.
        rh:     Relative humidity in percent (0–100).

    Returns:
        Humidex value (dimensionless, roughly °C-equivalent perceived heat).

    Reference:
        Masterton, J.M. & Richardson, F.A. (1979). Humidex: A Method of
        Quantifying Human Discomfort Due to Excessive Heat and Humidity.
        CLI 1-79. Atmospheric Environment Service, Downsview, Ontario.
    """
    # Saturation vapour pressure adjusted by relative humidity (hPa)
    e = 6.112 * (10 ** (7.5 * temp_c / (237.7 + temp_c))) * (rh / 100.0)
    return temp_c + 0.5555 * (e - 10.0)


def classify_humidex(humidex: float) -> str:
    """
    Map a Humidex value to a comfort class label.

    Thresholds (OHCOW 2022):
        < 30           → Comfortable
        30.0 – 39.9   → Caution
        40.0 – 45.0   → Extreme Caution
        45.1 – 54.0   → Danger
        > 54.0        → Extreme Danger

    Args:
        humidex: Humidex value.

    Returns:
        One of: "Comfortable", "Caution", "Extreme Caution", "Danger",
        "Extreme Danger".
    """
    if humidex < 30.0:
        return "Comfortable"
    elif humidex < 40.0:
        return "Caution"
    elif humidex <= 45.0:
        return "Extreme Caution"
    elif humidex <= 54.0:
        return "Danger"
    else:
        return "Extreme Danger"


def humidex_to_risk_level(humidex: float) -> str:
    """
    Map a Humidex value to a categorical risk level used in alerts and maps.

    Risk levels align with OHCOW 2022 action thresholds:
        < 30   → LOW
        30–39  → MODERATE
        40–45  → HIGH
        > 45   → CRITICAL

    Args:
        humidex: Humidex value.

    Returns:
        "LOW" | "MODERATE" | "HIGH" | "CRITICAL"
    """
    if humidex < 30.0:
        return "LOW"
    elif humidex < 40.0:
        return "MODERATE"
    elif humidex <= 45.0:
        return "HIGH"
    else:
        return "CRITICAL"
