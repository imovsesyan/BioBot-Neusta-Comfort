"""Tests for the F12-UC1 / F12-UC2 zone and protection-slot modules.

Coverage
--------
1. ``test_plant_zone_label_mapping`` — all 5 humidex risk levels map to the
   correct plant zone label.
2. ``test_diurnal_window_boundaries`` — boundary hours (06, 10, 15, 19, 22)
   land in the correct window.
3. ``test_zone_csv_columns`` — the F12-UC1 builder yields a frame with all
   required columns.
4. ``test_protection_slots_columns`` — slot output has all required columns.
5. ``test_danger_slot_minimum_duration`` — no danger slot shorter than 2h.
6. ``test_daily_schedule_actions`` — every slot_type yields a non-empty
   recommended_actions string.
"""

from __future__ import annotations

import pandas as pd
import pytest

from biobot.zones.protection_slots import (
    DAILY_SCHEDULE_COLUMNS,
    MIN_SLOT_HOURS,
    PROTECTION_SLOT_COLUMNS,
    generate_daily_schedule,
    generate_protection_slots,
)
from biobot.zones.temporal import (
    HUMIDEX_TO_PLANT_ZONE,
    PLANT_ZONE_LABELS,
    assign_diurnal_window,
    assign_plant_zone_label,
)

# Import the script's table builder so the integration test exercises the
# user-facing CSV pathway.
import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_UC1_SCRIPT = _ROOT / "scripts" / "f12_uc1_zone_analysis.py"
_spec = importlib.util.spec_from_file_location("_f12_uc1", _UC1_SCRIPT)
_uc1 = importlib.util.module_from_spec(_spec)
sys.modules["_f12_uc1"] = _uc1
assert _spec.loader is not None
_spec.loader.exec_module(_uc1)


# ---------------------------------------------------------------------------
# 1. Plant zone label mapping
# ---------------------------------------------------------------------------


def test_plant_zone_label_mapping():
    expected = {
        "livable": "favorable",
        "discomfort": "moderate",
        "high_risk": "dangerous",
        "dangerous": "dangerous",
        "critical": "critical",
    }

    # The function must agree with the documented mapping for every band.
    for risk_level, plant_zone in expected.items():
        assert assign_plant_zone_label(risk_level) == plant_zone

    # And the table itself must be exactly that mapping (no drift).
    assert HUMIDEX_TO_PLANT_ZONE == expected

    # Plant vocab is exactly four ordered labels.
    assert PLANT_ZONE_LABELS == ["favorable", "moderate", "dangerous", "critical"]

    # Unknown input must raise.
    with pytest.raises(ValueError):
        assign_plant_zone_label("unknown_band")


# ---------------------------------------------------------------------------
# 2. Diurnal window boundaries
# ---------------------------------------------------------------------------


def test_diurnal_window_boundaries():
    # Boundary hours called out by the task spec.
    assert assign_diurnal_window(6) == "morning"
    assert assign_diurnal_window(10) == "midday"
    assert assign_diurnal_window(15) == "afternoon"
    assert assign_diurnal_window(19) == "evening"
    assert assign_diurnal_window(22) == "night"

    # Just-before each boundary belongs to the previous window.
    assert assign_diurnal_window(5) == "night"
    assert assign_diurnal_window(9) == "morning"
    assert assign_diurnal_window(14) == "midday"
    assert assign_diurnal_window(18) == "afternoon"
    assert assign_diurnal_window(21) == "evening"

    # Wrap-around night.
    assert assign_diurnal_window(0) == "night"
    assert assign_diurnal_window(23) == "night"

    # Out-of-range raises.
    with pytest.raises(ValueError):
        assign_diurnal_window(-1)
    with pytest.raises(ValueError):
        assign_diurnal_window(24)


# ---------------------------------------------------------------------------
# 3. Zone CSV columns (UC1 integration)
# ---------------------------------------------------------------------------


def test_zone_csv_columns():
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-07-01 04:00",
                    "2025-07-01 09:00",
                    "2025-07-01 14:00",
                    "2025-07-01 17:00",
                    "2025-07-01 20:00",
                ],
                utc=True,
            ),
            "humidex_c": [22.0, 35.0, 42.0, 48.0, 56.0],
            "risk_level": ["livable", "discomfort", "high_risk", "dangerous", "critical"],
        }
    )

    out = _uc1.build_zone_table(df)

    expected_cols = {
        "timestamp",
        "humidex_c",
        "humidex_risk_level",
        "plant_zone_label",
        "diurnal_window",
        "hour",
        "day_of_week",
    }
    assert expected_cols.issubset(set(out.columns))

    # Sanity: each plant_zone_label is in the documented vocabulary.
    assert set(out["plant_zone_label"]).issubset(set(PLANT_ZONE_LABELS))

    # Hour and day_of_week are integers in valid ranges.
    assert out["hour"].between(0, 23).all()
    assert out["day_of_week"].between(0, 6).all()


# ---------------------------------------------------------------------------
# 4. Protection slots: columns
# ---------------------------------------------------------------------------


def _hourly_zone_frame(zones: list[str]) -> pd.DataFrame:
    """Helper: build a per-hour zone frame starting at 2025-07-01 00:00 UTC."""
    timestamps = pd.date_range("2025-07-01 00:00", periods=len(zones), freq="1h", tz="UTC")
    return pd.DataFrame({"timestamp": timestamps, "plant_zone_label": zones})


def test_protection_slots_columns():
    df = _hourly_zone_frame(
        ["favorable"] * 4 + ["moderate"] * 2 + ["dangerous"] * 4 + ["favorable"] * 3
    )

    slots = generate_protection_slots(df)

    assert list(slots.columns) == PROTECTION_SLOT_COLUMNS
    # Slot types must be drawn from the documented vocabulary.
    assert set(slots["slot_type"]).issubset(
        {"safe_slot", "danger_slot", "transition_slot", "neutral"}
    )


# ---------------------------------------------------------------------------
# 5. Danger slot minimum duration
# ---------------------------------------------------------------------------


def test_danger_slot_minimum_duration():
    # Three different cases:
    #   - 4h dangerous run -> qualifies as danger_slot
    #   - 1h dangerous excursion -> filtered out (becomes neutral)
    #   - 5h favorable run -> safe_slot (not relevant here, but asserts that
    #     the filter is type-specific).
    df = _hourly_zone_frame(
        ["favorable"] * 5
        + ["dangerous"] * 4  # qualifies
        + ["moderate"] * 2
        + ["dangerous"]  # 1h, sub-threshold
        + ["moderate"] * 2
    )

    slots = generate_protection_slots(df)

    danger_slots = slots[slots["slot_type"] == "danger_slot"]
    assert len(danger_slots) >= 1
    # Every danger slot must meet or exceed the 2-hour minimum.
    assert (danger_slots["duration_hours"] >= MIN_SLOT_HOURS).all()


# ---------------------------------------------------------------------------
# 6. Daily schedule actions
# ---------------------------------------------------------------------------


def test_daily_schedule_actions():
    # Build a frame that exercises every slot type:
    #   - 4h favorable in the morning window (safe_slot in 'morning')
    #   - 4h dangerous starting after the favorable run (danger_slot,
    #     plus a synthetic transition_slot inserted before it)
    #   - 1h dangerous excursion (becomes neutral)
    timestamps = pd.date_range(
        "2025-07-01 06:00", periods=15, freq="1h", tz="UTC"
    )
    zones = (
        ["favorable"] * 4   # 06,07,08,09  -> safe_slot in morning
        + ["moderate"] * 2  # 10,11        -> neutral
        + ["dangerous"] * 4 # 12,13,14,15  -> danger_slot (+ transition)
        + ["moderate"] * 1  # 16           -> neutral
        + ["dangerous"] * 1 # 17           -> neutral (sub-threshold)
        + ["favorable"] * 3 # 18,19,20     -> safe_slot (evening / afternoon mix)
    )
    df = pd.DataFrame({"timestamp": timestamps, "plant_zone_label": zones})

    slots = generate_protection_slots(df)
    schedule = generate_daily_schedule(slots)

    assert list(schedule.columns) == DAILY_SCHEDULE_COLUMNS

    # Every represented slot_type produces a non-empty action string.
    for slot_type in schedule["slot_type"].unique():
        rows = schedule[schedule["slot_type"] == slot_type]
        assert len(rows) > 0
        assert (rows["recommended_actions"].astype(str).str.len() > 0).all(), (
            f"Empty recommended_actions for slot_type={slot_type}"
        )

    # All four canonical slot types should be present given the constructed
    # input above (safe + danger + transition + neutral).
    assert {"safe_slot", "danger_slot", "transition_slot", "neutral"}.issubset(
        set(schedule["slot_type"])
    )

    # Priorities follow the documented mapping.
    priorities = dict(zip(schedule["slot_type"], schedule["priority"]))
    assert priorities.get("danger_slot") == "high"
    assert priorities.get("transition_slot") == "high"
    assert priorities.get("safe_slot") == "medium"
    assert priorities.get("neutral") == "low"
