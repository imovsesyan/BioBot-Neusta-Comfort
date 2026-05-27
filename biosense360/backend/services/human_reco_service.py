"""
Human recommendation service for BioSense360.

Generates personalised thermal safety advice based on Humidex level,
population group, and activity type.

Protocol references:
  - HAS 2023  — heat stroke prevention for vulnerable groups
  - ACGIH TLV 2023 — occupational heat exposure limits
  - INRS R-447 — French occupational safety in heat
  - OHCOW 2022 — Humidex action thresholds
"""

from backend.constants.thresholds import VULNERABILITY_ADJUSTMENTS
from backend.schemas.recommendations import HumanRecoRequest, HumanRecoResponse
from backend.services.humidex_service import humidex_to_risk_level


# ---------------------------------------------------------------------------
# Advice lookup tables — keyed by risk level and profile
# ---------------------------------------------------------------------------

_BASE_ADVICE: dict[str, str] = {
    "LOW": (
        "Conditions are comfortable. Normal outdoor activities can proceed without restriction. "
        "Maintain regular hydration habits."
    ),
    "MODERATE": (
        "Conditions require caution. Limit intense physical effort during peak heat hours "
        "(11:00–17:00). Take regular breaks in shaded or cool areas."
    ),
    "HIGH": (
        "Conditions are hazardous. Avoid all non-essential outdoor exertion. Rest in shade or "
        "air-conditioned spaces. Monitor for signs of heat exhaustion."
    ),
    "CRITICAL": (
        "Life-threatening heat conditions. Stay indoors with cooling. "
        "Emergency cooling measures required. Call emergency services (15 / 18) "
        "immediately if anyone shows confusion, lack of sweating, or loss of consciousness."
    ),
}

_HYDRATION_ADVICE: dict[str, str] = {
    "LOW":      "Drink 1.5 L of water throughout the day. No special protocol required.",
    "MODERATE": "Drink at least 500 mL per hour during outdoor activity. Avoid alcohol and caffeine.",
    "HIGH":     "Drink 750 mL–1 L per hour. Electrolyte drinks recommended for prolonged exposure.",
    "CRITICAL": "Continuous hydration essential — minimum 1 L per hour. Oral rehydration salts if "
                "sweating is profuse. Seek medical support.",
}

_CLOTHING_ADVICE: dict[str, str] = {
    "LOW":      "Lightweight, breathable clothing. Sun protection (hat, SPF 30+) if outdoors.",
    "MODERATE": "Loose, light-coloured, moisture-wicking clothing. Wide-brim hat mandatory outdoors. "
                "SPF 50+ sunscreen.",
    "HIGH":     "Minimal clothing mass — loose linen or technical fabrics. Cooling vest if available. "
                "Avoid dark colours that absorb radiation.",
    "CRITICAL": "Lightest possible clothing. Cooling vest or wet towel on neck/wrists. "
                "Do not remain outdoors.",
}

_EMERGENCY_PROTOCOL: dict[str, str] = {
    "LOW":      "No emergency protocol required.",
    "MODERATE": "Know the signs of heat exhaustion: heavy sweating, weakness, cold/pale/moist skin, "
                "nausea, fainting.",
    "HIGH":     "HAS 2023 — Heat exhaustion protocol: move to cool area, remove excess clothing, "
                "apply cool water, monitor vitals. Call 15 if no improvement within 30 min.",
    "CRITICAL": "HAS 2023 — Heat stroke protocol: CALL 15 (SAMU) IMMEDIATELY. "
                "Move to coolest available location. Apply ice packs to neck, armpits, groin. "
                "Fan the person. Do not give fluids to an unconscious person.",
}

_SAFE_SLOTS_BY_RISK: dict[str, list[str]] = {
    "LOW":      ["All hours — conditions comfortable"],
    "MODERATE": ["06:00–10:00", "19:00–22:00"],
    "HIGH":     ["06:00–08:00", "20:00–22:00"],
    "CRITICAL": ["No safe outdoor slots — remain indoors"],
}


def _apply_vulnerability(humidex_val: float, age_group: str) -> float:
    """
    Adjust effective Humidex for more vulnerable population groups.
    HAS 2023 recommends applying standard thresholds 5 Humidex units
    earlier for the elderly and children.
    """
    adjustment = VULNERABILITY_ADJUSTMENTS.get(age_group, 0.0)
    return humidex_val - adjustment  # subtract negative → raises effective value


def _athlete_addendum(activity_type: str, risk_level: str) -> str:
    """Extra warm-up/cool-down guidance for athletes."""
    if risk_level in ("HIGH", "CRITICAL"):
        return (
            " ATHLETE PROTOCOL: Cancel or postpone intense training. "
            "If competition is unavoidable, execute a pre-cooling strategy "
            "(ice vest, cold water immersion of forearms) and extend cool-down phase "
            "to 20 min post-exercise."
        )
    return (
        " ATHLETE PROTOCOL: 10-min progressive warm-up mandatory. "
        "15-min cool-down with cold-water sponging. Acclimatisation period of 10–14 days "
        "recommended before high-intensity outdoor competition."
    )


def _worker_addendum(risk_level: str) -> str:
    """ACGIH TLV 2023 + INRS R-447 work-rest schedule guidance."""
    schedules = {
        "LOW":      "Standard work-rest schedule. ACGIH TLV 2023 — no additional restriction.",
        "MODERATE": (
            "ACGIH TLV 2023 / INRS R-447 — Moderate workload: 50 min work / 10 min rest per hour. "
            "Rotate outdoor tasks with indoor tasks."
        ),
        "HIGH": (
            "ACGIH TLV 2023 / INRS R-447 — Light workload only: 40 min work / 20 min rest per hour. "
            "Stop all heavy outdoor work. Décret 2025-482 — employer must provide shaded rest areas "
            "and free cold water."
        ),
        "CRITICAL": (
            "ACGIH TLV 2023 / INRS R-447 — STOP all outdoor work immediately. "
            "Décret 2025-482 — employer obligation to suspend outdoor activities. "
            "Workers must be moved to cool indoor environment. "
            "Activate company heat-wave emergency plan."
        ),
    }
    return " WORKER PROTOCOL: " + schedules.get(risk_level, "Follow standard protocols.")


def generate_human_recommendation(req: HumanRecoRequest) -> HumanRecoResponse:
    """
    Generate personalised human thermal safety advice.

    Vulnerability adjustments (HAS 2023) raise the effective Humidex for
    Elderly and Child groups, triggering higher risk classifications at lower
    actual Humidex values.
    """
    # Adjust Humidex for population vulnerability
    effective_humidex = _apply_vulnerability(req.humidex, req.age_group)
    risk_level = humidex_to_risk_level(effective_humidex)

    # Build base advice
    advice = _BASE_ADVICE[risk_level]

    # Append profile-specific addenda
    if req.age_group == "Athlete":
        advice += _athlete_addendum(req.activity_type, risk_level)
    elif req.age_group == "Worker":
        advice += _worker_addendum(risk_level)
    elif req.age_group == "Elderly":
        advice += (
            " ELDERLY ADVISORY (HAS 2023): Check on isolated elderly neighbours twice daily. "
            "Keep indoor temperature below 25 °C. Never leave alone in an unventilated room."
        )
    elif req.age_group == "Child":
        advice += (
            " CHILD ADVISORY (HAS 2023): Children under 4 are highly vulnerable. "
            "Never leave in a closed vehicle. Ensure regular wet cool-down. "
            "Monitor for irritability, reduced urination, and skin flushing."
        )

    # Emergency protocol string (always reference HAS 2023)
    emergency = _EMERGENCY_PROTOCOL[risk_level]

    return HumanRecoResponse(
        advice=advice,
        hydration=_HYDRATION_ADVICE[risk_level],
        clothing=_CLOTHING_ADVICE[risk_level],
        risk_level=risk_level,
        safe_slots=_SAFE_SLOTS_BY_RISK[risk_level],
        emergency_protocol=emergency,
    )
