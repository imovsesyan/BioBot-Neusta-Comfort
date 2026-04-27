"""F12-UC1 temporal zone primitives.

Two pure functions plus their data tables:

* :data:`DIURNAL_WINDOWS` and :func:`assign_diurnal_window` — canonical
  time-of-day windows (HHEWS / extension-publication aligned, see
  ``docs/f13/F13_research_plants.md`` Section 4.3).
* :data:`HUMIDEX_TO_PLANT_ZONE` and :func:`assign_plant_zone_label` — translate
  the F10 humidex risk vocabulary (livable / discomfort / high_risk /
  dangerous / critical) into a plant-centric vigilance vocabulary
  (favorable / moderate / dangerous / critical).

Scientific notes
----------------
* Plants are more thermally sensitive than humans: the research report flags
  plant heat stress beginning at ``high_risk`` (humidex 40-45). We therefore
  map ``high_risk`` straight to plant-vocabulary ``dangerous``, while
  human-vocabulary ``dangerous`` (humidex > 45) also lands at plant
  ``dangerous`` and ``critical`` (humidex > 54) is reserved for the most
  acute band.
* Inputs to these functions must be the **raw humidex risk vocabulary** from
  ``biobot.risk.rules`` only. Per CLAUDE.md, ``vivabilite_binary_mean`` and
  F10-UC4 classifier outputs must never be used as zone inputs.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# ---------------------------------------------------------------------------
# Diurnal time windows
# ---------------------------------------------------------------------------

# Canonical windows. Each entry is (window_name, start_hour_inclusive,
# end_hour_exclusive). ``night`` is the only wrap-around window (22..24, 0..6).
DIURNAL_WINDOWS: List[Tuple[str, int, int]] = [
    ("night", 22, 6),
    ("morning", 6, 10),
    ("midday", 10, 15),
    ("afternoon", 15, 19),
    ("evening", 19, 22),
]


def assign_diurnal_window(hour: int) -> str:
    """Return the canonical diurnal window name for a clock hour 0..23.

    Boundaries are start-inclusive and end-exclusive::

        hour < 6  or hour >= 22  -> "night"
        6  <= hour <  10         -> "morning"
        10 <= hour <  15         -> "midday"
        15 <= hour <  19         -> "afternoon"
        19 <= hour <  22         -> "evening"

    Raises
    ------
    ValueError
        If ``hour`` is not an integer in [0, 23].
    """
    if not isinstance(hour, (int,)) or isinstance(hour, bool):
        raise ValueError(f"hour must be an int in [0, 23], got {hour!r}")
    if hour < 0 or hour > 23:
        raise ValueError(f"hour must be in [0, 23], got {hour}")

    if hour >= 22 or hour < 6:
        return "night"
    if hour < 10:
        return "morning"
    if hour < 15:
        return "midday"
    if hour < 19:
        return "afternoon"
    return "evening"


# ---------------------------------------------------------------------------
# Humidex risk level -> plant zone label
# ---------------------------------------------------------------------------

# Plants are more sensitive than humans (Wahid et al., Env. Exp. Bot. 2007;
# see docs/f13/F13_research_plants.md Section 2.2). Heat stress starts at
# humidex 40-45 ("high_risk" for humans), so for plants this band is already
# "dangerous". The acute "critical" band (humidex > 54) is preserved.
HUMIDEX_TO_PLANT_ZONE: Dict[str, str] = {
    "livable": "favorable",
    "discomfort": "moderate",
    "high_risk": "dangerous",
    "dangerous": "dangerous",
    "critical": "critical",
}

# Ordered for plotting / palettes (least to most severe).
PLANT_ZONE_LABELS: List[str] = ["favorable", "moderate", "dangerous", "critical"]


def assign_plant_zone_label(humidex_risk_level: str) -> str:
    """Map an F10 humidex risk level to a plant-vocabulary zone label.

    Parameters
    ----------
    humidex_risk_level
        One of ``livable``, ``discomfort``, ``high_risk``, ``dangerous``,
        ``critical``.

    Returns
    -------
    str
        One of ``favorable``, ``moderate``, ``dangerous``, ``critical``.

    Raises
    ------
    ValueError
        If the input is not a recognized humidex risk level.
    """
    try:
        return HUMIDEX_TO_PLANT_ZONE[humidex_risk_level]
    except KeyError as exc:
        raise ValueError(
            f"Unknown humidex_risk_level {humidex_risk_level!r}; "
            f"expected one of {sorted(HUMIDEX_TO_PLANT_ZONE)}."
        ) from exc
