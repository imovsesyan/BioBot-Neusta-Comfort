"""F12-UC2 thermal protection slots.

Two pure functions:

* :func:`generate_protection_slots` groups contiguous rows of an F12-UC1 zone
  table into ``safe_slot``, ``danger_slot``, ``transition_slot``, and
  ``neutral`` runs with a 2-hour minimum duration filter.
* :func:`generate_daily_schedule` maps each slot to a recommended-action
  string and a high/medium/low priority, applying a diurnal-window modifier
  for safe slots (morning vs evening).

Scientific notes
----------------
* The 2h minimum duration mirrors the rolling-window vigilance rule used by
  Heat-Health Early-Warning Systems (Météo-France / WHO Europe HHEWS) and
  the recommendation in ``docs/f13/F13_research_plants.md`` Section 4.1.
* The 30-60 min ``transition_slot`` is the actionable lead-time before a
  ``danger_slot`` opens — the "do something now" window for moving plants
  to shade or pre-irrigating.
* No ML model. All slot logic is deterministic over F10-derived plant zone
  labels.
"""

from __future__ import annotations

from typing import List

import pandas as pd

from biobot.zones.temporal import assign_diurnal_window

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DANGER_ZONES = {"dangerous", "critical"}
SAFE_ZONES = {"favorable"}

MIN_SLOT_HOURS = 2.0  # HHEWS-aligned minimum sustained duration

# Pre-danger transition slot: 30-60 min lead time. We use 60 min as a single
# canonical value here so the slot length is deterministic; downstream UIs
# can still apply finer-grained logic.
TRANSITION_LEAD_MINUTES = 60

# Action mapping. Lookup is by (slot_type, diurnal_window) when relevant,
# else by slot_type alone.
ACTIONS_BY_SLOT_AND_WINDOW = {
    ("safe_slot", "morning"): "water plants, ventilate, open windows",
    ("safe_slot", "evening"): "water plants, apply fertilizer if needed",
}
ACTIONS_BY_SLOT = {
    "safe_slot": "ventilate carefully, monitor soil moisture",
    "transition_slot": (
        "move sensitive plants to shade, close south-facing windows, "
        "prepare irrigation"
    ),
    "danger_slot": "avoid watering, close all windows, use shading",
    "neutral": "monitor conditions",
}
PRIORITY_BY_SLOT = {
    "danger_slot": "high",
    "transition_slot": "high",
    "safe_slot": "medium",
    "neutral": "low",
}

PROTECTION_SLOT_COLUMNS = [
    "slot_type",
    "start_ts",
    "end_ts",
    "dominant_zone",
    "duration_hours",
]

DAILY_SCHEDULE_COLUMNS = [
    "time_block",
    "slot_type",
    "diurnal_window",
    "recommended_actions",
    "priority",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _classify_zone(zone: str) -> str:
    """Coarse-grain a per-row plant zone into a slot category."""
    if zone in DANGER_ZONES:
        return "danger"
    if zone in SAFE_ZONES:
        return "safe"
    return "other"  # 'moderate' (and any unknown) -> neutral background


def _runs(df: pd.DataFrame) -> pd.DataFrame:
    """Group contiguous rows by their slot category, preserving timestamps.

    Returns a frame with columns:
        ``category``, ``start_ts``, ``end_ts``, ``dominant_zone``,
        ``duration_hours``, ``n_rows``.
    """
    if df.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "start_ts",
                "end_ts",
                "dominant_zone",
                "duration_hours",
                "n_rows",
            ]
        )

    work = df.copy().sort_values("timestamp").reset_index(drop=True)
    work["timestamp"] = pd.to_datetime(work["timestamp"], utc=True, errors="coerce")
    work = work.dropna(subset=["timestamp", "plant_zone_label"]).reset_index(drop=True)
    if work.empty:
        return pd.DataFrame(
            columns=[
                "category",
                "start_ts",
                "end_ts",
                "dominant_zone",
                "duration_hours",
                "n_rows",
            ]
        )

    work["category"] = work["plant_zone_label"].map(_classify_zone)
    # New run whenever the category changes vs the previous row.
    work["run_id"] = (work["category"] != work["category"].shift()).cumsum()

    runs: List[dict] = []
    for _, group in work.groupby("run_id", sort=True):
        start_ts = group["timestamp"].iloc[0]
        end_ts = group["timestamp"].iloc[-1]
        duration_hours = (end_ts - start_ts).total_seconds() / 3600.0
        # Mode of the per-row zone within the run as the "dominant" plant zone.
        modes = group["plant_zone_label"].mode()
        dominant_zone = (
            modes.iloc[0] if len(modes) else group["plant_zone_label"].iloc[0]
        )
        runs.append(
            {
                "category": group["category"].iloc[0],
                "start_ts": start_ts,
                "end_ts": end_ts,
                "dominant_zone": dominant_zone,
                "duration_hours": duration_hours,
                "n_rows": int(len(group)),
            }
        )
    return pd.DataFrame(runs)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_protection_slots(df: pd.DataFrame) -> pd.DataFrame:
    """Group F12-UC1 zone rows into safe / danger / transition / neutral slots.

    Parameters
    ----------
    df
        DataFrame with at minimum the columns ``timestamp`` and
        ``plant_zone_label`` as produced by F12-UC1.

    Returns
    -------
    pd.DataFrame
        Columns: ``slot_type``, ``start_ts``, ``end_ts``, ``dominant_zone``,
        ``duration_hours``. ``slot_type`` is one of ``safe_slot``,
        ``danger_slot``, ``transition_slot``, ``neutral``.

    Notes
    -----
    * A run of contiguous rows whose plant zone is in
      ``{dangerous, critical}`` becomes a ``danger_slot`` only if the run
      lasts at least :data:`MIN_SLOT_HOURS` (default 2h). Shorter runs are
      classified as ``neutral`` to avoid over-alerting on transient
      excursions.
    * Likewise for ``safe_slot`` (favorable, ≥2h).
    * A synthetic ``transition_slot`` of length
      :data:`TRANSITION_LEAD_MINUTES` is inserted immediately before each
      ``danger_slot``.
    """
    required = {"timestamp", "plant_zone_label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"generate_protection_slots: missing required columns {sorted(missing)}"
        )

    runs = _runs(df)
    if runs.empty:
        return pd.DataFrame(columns=PROTECTION_SLOT_COLUMNS)

    slots: List[dict] = []
    for _, run in runs.iterrows():
        category = run["category"]
        duration = float(run["duration_hours"])

        if category == "danger" and duration >= MIN_SLOT_HOURS:
            slot_type = "danger_slot"
        elif category == "safe" and duration >= MIN_SLOT_HOURS:
            slot_type = "safe_slot"
        else:
            slot_type = "neutral"

        slots.append(
            {
                "slot_type": slot_type,
                "start_ts": run["start_ts"],
                "end_ts": run["end_ts"],
                "dominant_zone": run["dominant_zone"],
                "duration_hours": duration,
            }
        )

    slots_df = pd.DataFrame(slots, columns=PROTECTION_SLOT_COLUMNS)

    # Insert a transition_slot immediately before each danger_slot.
    transitions: List[dict] = []
    for _, slot in slots_df[slots_df["slot_type"] == "danger_slot"].iterrows():
        start = slot["start_ts"] - pd.Timedelta(minutes=TRANSITION_LEAD_MINUTES)
        end = slot["start_ts"]
        transitions.append(
            {
                "slot_type": "transition_slot",
                "start_ts": start,
                "end_ts": end,
                "dominant_zone": slot["dominant_zone"],
                "duration_hours": TRANSITION_LEAD_MINUTES / 60.0,
            }
        )

    if transitions:
        slots_df = pd.concat(
            [slots_df, pd.DataFrame(transitions, columns=PROTECTION_SLOT_COLUMNS)],
            ignore_index=True,
        )

    slots_df = slots_df.sort_values("start_ts").reset_index(drop=True)
    return slots_df[PROTECTION_SLOT_COLUMNS]


def _slot_diurnal_window(slot_row: pd.Series) -> str:
    """Return the diurnal window for a slot, anchored on its midpoint."""
    start = pd.to_datetime(slot_row["start_ts"], utc=True, errors="coerce")
    end = pd.to_datetime(slot_row["end_ts"], utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return "unknown"
    midpoint = start + (end - start) / 2
    return assign_diurnal_window(int(midpoint.hour))


def _format_time_block(slot_row: pd.Series) -> str:
    start = pd.to_datetime(slot_row["start_ts"], utc=True, errors="coerce")
    end = pd.to_datetime(slot_row["end_ts"], utc=True, errors="coerce")
    if pd.isna(start) or pd.isna(end):
        return ""
    return f"{start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%H:%M')} UTC"


def generate_daily_schedule(slots_df: pd.DataFrame) -> pd.DataFrame:
    """Map each protection slot to a recommended action + priority.

    Parameters
    ----------
    slots_df
        Output of :func:`generate_protection_slots`.

    Returns
    -------
    pd.DataFrame
        Columns: ``time_block``, ``slot_type``, ``diurnal_window``,
        ``recommended_actions``, ``priority``.
    """
    if slots_df.empty:
        return pd.DataFrame(columns=DAILY_SCHEDULE_COLUMNS)

    required = {"slot_type", "start_ts", "end_ts"}
    missing = required - set(slots_df.columns)
    if missing:
        raise ValueError(
            f"generate_daily_schedule: missing required columns {sorted(missing)}"
        )

    rows: List[dict] = []
    for _, slot in slots_df.iterrows():
        slot_type = str(slot["slot_type"])
        diurnal_window = _slot_diurnal_window(slot)

        action = ACTIONS_BY_SLOT_AND_WINDOW.get(
            (slot_type, diurnal_window),
            ACTIONS_BY_SLOT.get(slot_type, "monitor conditions"),
        )
        priority = PRIORITY_BY_SLOT.get(slot_type, "low")

        rows.append(
            {
                "time_block": _format_time_block(slot),
                "slot_type": slot_type,
                "diurnal_window": diurnal_window,
                "recommended_actions": action,
                "priority": priority,
            }
        )

    return pd.DataFrame(rows, columns=DAILY_SCHEDULE_COLUMNS)
