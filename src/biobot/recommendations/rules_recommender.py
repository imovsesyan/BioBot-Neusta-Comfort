"""F11-UC1 rule-based comfort recommender.

A deterministic ``(humidex_risk_level x profile) -> recommendation`` lookup
that produces an ordered set of actions, clothing/activity guidance, and a
mandatory health alert when the user is vulnerable and the heat risk is
elevated.

Scientific framing
------------------
The five risk levels accepted on input are the F10 humidex bands defined in
``src/biobot/risk/rules.py`` plus the ``critical`` band emitted by
``create_rule_alerts`` for humidex above 54::

    risk_level     | humidex band   | meaning
    -------------- | -------------- | --------------------------------------
    livable        | < 30           | Little or no discomfort
    discomfort     | 30 <= h < 40   | Some discomfort
    high_risk      | 40 <= h <= 45  | Great discomfort, reduce exertion
    dangerous      | 45 < h <= 54   | Dangerous heat stress
    critical       | > 54           | Imminent heat-stroke risk

Per the F11 research recommendation (``docs/f11/F11_research_recommendations.md``)
the recommendation content is sourced from public-health guidance:

* WHO, *Public health advice on preventing health effects of heat*
  (Regional Office for Europe, 2011 / 2018 update).
* Santé publique France, *Plan national canicule*.
* ASHRAE 55-2023, *Thermal Environmental Conditions for Human Occupancy*
  (clothing/activity tables).
* ISO 7730:2005 / ISO 9920:2007 (background CLO/MET reference values).

Constraints
-----------
* Drives off humidex risk bands only — does NOT consume
  ``vivabilite_binary_mean`` or F10-UC4 classifier output (CLAUDE.md leakage
  guardrail; see ``docs/f11/F11_research_recommendations.md`` Section 4).
* Outputs are informational and non-medical. The integration script and any
  delivery layer must surface this disclaimer.
* No new thresholds are invented. Heat severity bands are taken verbatim from
  the F10 humidex rules; recommendations only differentiate WHAT to do at each
  band given the user profile.
"""

from __future__ import annotations

from typing import Dict

from biobot.recommendations.profile import UserProfile


# Canonical risk levels accepted by the recommender. The underlying F10
# ``assign_humidex_risk_level`` collapses humidex > 45 into ``dangerous`` and
# carries an ``is_critical_humidex`` flag for humidex > 54; ``create_rule_alerts``
# elevates the latter to ``critical``. The recommender accepts both surface
# forms so it can be plugged into either pipeline.
VALID_RISK_LEVELS = ("livable", "discomfort", "high_risk", "dangerous", "critical")
ELEVATED_RISK_LEVELS = ("high_risk", "dangerous", "critical")

# Required keys in every recommendation dict — exercised by the test suite.
REQUIRED_KEYS = ("action", "clothing_advice", "activity_advice", "alert")


# ---------------------------------------------------------------------------
# Action library
# ---------------------------------------------------------------------------
#
# The action strings below are paraphrased from WHO heat-health guidance, the
# Santé publique France Plan Canicule public advice pages, and ASHRAE 55-2023
# clothing/activity tables. They are intentionally short and operational so
# that the optional Claude rephrasing layer (F11-UC5, ai_recommender.py) can
# render them in a friendlier voice without changing meaning.

# Base "what to do now" by risk level (the spine of the rule table).
_BASE_ACTION_BY_RISK: Dict[str, str] = {
    "livable":     "No special action needed. Maintain normal hydration.",
    "discomfort":  "Drink water regularly. Ventilate the room and avoid direct sun exposure.",
    "high_risk":   "Reduce physical exertion, hydrate every 15-20 minutes, "
                   "move to a cooler or shaded space.",
    "dangerous":   "Stop physical activity, move indoors to the coolest available "
                   "space, drink water, and cool the skin (damp cloth, lukewarm shower).",
    "critical":    "Treat as a heat emergency: stop all activity, move to the "
                   "coolest space available, actively cool the body, and seek "
                   "medical assistance (in France: SAMU 15) if symptoms appear.",
}

# Activity-axis modifiers. Active users at moderate humidex already need to
# slow down before sedentary users do; this preserves the metabolic-heat
# distinction documented in ASHRAE 55 / ISO 7730.
_ACTIVITY_MODIFIERS: Dict[tuple, str] = {
    ("discomfort", 1): " If you are exerting yourself, slow down and take a break.",
    ("high_risk", 1):  " Stop intense activity now; resume only after temperatures drop.",
    ("dangerous", 1):  " All physical activity should stop until conditions improve.",
    ("critical", 1):   " Any continued exertion at this humidex level is unsafe.",
}


# ---------------------------------------------------------------------------
# Clothing advice
# ---------------------------------------------------------------------------
#
# Mapped to ASHRAE 55 CLO bands: light approx. 0.4 CLO (summer T-shirt + light
# trousers), heavy approx. 1.0+ CLO (long sleeves / layers / outerwear).
def _clothing_advice(risk_level: str, profile: UserProfile) -> str:
    if risk_level == "livable":
        if profile.clothing_level == 1:
            return (
                "Current clothing is suitable. Remove a layer if you start to "
                "feel warm."
            )
        return "Light clothing is appropriate for current conditions."

    if risk_level == "discomfort":
        if profile.clothing_level == 1:
            return (
                "Remove an outer layer. Switch to light, breathable fabrics "
                "(cotton, linen)."
            )
        return "Keep wearing light, loose, breathable clothing in light colors."

    if risk_level == "high_risk":
        if profile.clothing_level == 1:
            return (
                "Heavy clothing is unsafe at this humidex. Change into the "
                "lightest, loosest clothing available immediately."
            )
        return (
            "Wear the lightest, loosest, lightest-coloured clothing you have. "
            "Cover your head if exposed to the sun."
        )

    # dangerous and critical are treated identically on the clothing axis: the
    # priority is heat dissipation, not insulation.
    if profile.clothing_level == 1:
        return (
            "Remove all unnecessary layers. Wear only light, loose clothing; "
            "dampen fabric with cool water to aid evaporation."
        )
    return (
        "Stay in light, loose clothing. Dampen fabric with cool water to "
        "support evaporative cooling."
    )


# ---------------------------------------------------------------------------
# Activity advice
# ---------------------------------------------------------------------------
def _activity_advice(risk_level: str, profile: UserProfile) -> str:
    if risk_level == "livable":
        return "Normal activity is fine. Stay hydrated as usual."

    if risk_level == "discomfort":
        if profile.activity_level == 1:
            return (
                "Reduce sustained effort. Take more frequent breaks and avoid "
                "the hottest hours (typically 12:00-16:00)."
            )
        return (
            "No activity restriction; if you go outside, prefer the cooler "
            "parts of the day."
        )

    if risk_level == "high_risk":
        if profile.activity_level == 1:
            return (
                "Stop intense activity. Limit movement to short, low-effort "
                "tasks and rest in a cool space between them."
            )
        return (
            "Stay still in a cool space. Avoid going outdoors during the "
            "hottest hours."
        )

    if risk_level == "dangerous":
        return (
            "Avoid all non-essential physical activity. Remain in the coolest "
            "available indoor space and rest."
        )

    # critical
    return (
        "All physical activity should stop. Stay in the coolest available "
        "space and minimize movement until conditions improve or help arrives."
    )


# ---------------------------------------------------------------------------
# Vulnerability alert
# ---------------------------------------------------------------------------
#
# Per WHO / Santé publique France Plan Canicule, vulnerable populations
# (elderly, infants/young children, people with chronic cardiovascular,
# respiratory or metabolic conditions, pregnant women) experience heat-related
# morbidity and mortality at humidex levels well below those that affect
# healthy adults. A non-empty alert is therefore mandatory for vulnerable
# users at high_risk, dangerous, and critical bands.
_VULNERABILITY_ALERTS: Dict[str, str] = {
    "high_risk": (
        "Vulnerability alert: at high heat-stress risk, vulnerable individuals "
        "(elderly, children, chronic conditions, pregnancy) should already be "
        "in a cool space and be checked on regularly. Hydrate often."
    ),
    "dangerous": (
        "Vulnerability alert: humidex is in the dangerous band. Vulnerable "
        "individuals require active monitoring, continuous cooling, and "
        "frequent hydration. Contact a relative, neighbour, or caregiver if "
        "alone. Watch for dizziness, nausea, confusion, or rapid pulse."
    ),
    "critical": (
        "Critical vulnerability alert: heat-stroke risk is imminent for "
        "vulnerable individuals. Move to the coolest available space, cool "
        "the body actively, and call emergency services immediately if any "
        "warning sign appears (in France: SAMU 15, EU-wide: 112). Do not "
        "leave the person alone."
    ),
}


def _action_for(risk_level: str, profile: UserProfile) -> str:
    base = _BASE_ACTION_BY_RISK[risk_level]
    modifier = _ACTIVITY_MODIFIERS.get((risk_level, profile.activity_level), "")
    return base + modifier


def get_recommendation(risk_level: str, profile: UserProfile) -> dict:
    """Return the deterministic recommendation for a (risk_level, profile) pair.

    Parameters
    ----------
    risk_level
        One of ``livable``, ``discomfort``, ``high_risk``, ``dangerous``,
        ``critical``. These align with the F10 humidex bands defined in
        ``biobot.risk.rules`` (with ``critical`` reserved for humidex > 54
        as surfaced by ``create_rule_alerts``).
    profile
        The binary :class:`UserProfile`.

    Returns
    -------
    dict
        Keys: ``action``, ``clothing_advice``, ``activity_advice``, ``alert``.
        ``alert`` is a non-empty string when ``profile.vulnerability == 1`` and
        ``risk_level`` is in ``{high_risk, dangerous, critical}``; otherwise
        it is the empty string.
    """

    if risk_level not in VALID_RISK_LEVELS:
        raise ValueError(
            f"Unknown risk_level {risk_level!r}. Expected one of "
            f"{VALID_RISK_LEVELS}."
        )
    if not isinstance(profile, UserProfile):
        raise TypeError(
            f"profile must be a UserProfile instance, got {type(profile).__name__}"
        )

    alert = ""
    if profile.vulnerability == 1 and risk_level in ELEVATED_RISK_LEVELS:
        alert = _VULNERABILITY_ALERTS[risk_level]

    return {
        "action": _action_for(risk_level, profile),
        "clothing_advice": _clothing_advice(risk_level, profile),
        "activity_advice": _activity_advice(risk_level, profile),
        "alert": alert,
    }
