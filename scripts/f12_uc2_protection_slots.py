"""F12-UC2: Build thermal protection slots and a daily action schedule.

Pipeline
--------
1. Load ``data/outputs/f12_zones_periods.csv`` (output of F12-UC1).
2. Run :func:`biobot.zones.generate_protection_slots` to group contiguous
   plant-zone runs into safe / danger / transition / neutral slots with a
   2-hour minimum duration.
3. Run :func:`biobot.zones.generate_daily_schedule` to map each slot to a
   recommended action and priority.
4. Save:
     * ``data/outputs/f12_protection_slots.csv``
     * ``data/outputs/f12_daily_schedule.csv``
5. Print a summary count of safe / danger / transition slots.

Scientific notes
----------------
* Pure deterministic windowing over the F12-UC1 plant zone labels (which
  are themselves deterministic over F10 humidex bands). No new model.
* No use of ``vivabilite_binary_mean`` or F10-UC4 outputs.

Run from the repository root::

    python scripts/f12_uc2_protection_slots.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.zones.protection_slots import (  # noqa: E402
    generate_daily_schedule,
    generate_protection_slots,
)


DEFAULT_INPUT = ROOT / "data" / "outputs" / "f12_zones_periods.csv"
DEFAULT_SLOTS_OUT = ROOT / "data" / "outputs" / "f12_protection_slots.csv"
DEFAULT_SCHEDULE_OUT = ROOT / "data" / "outputs" / "f12_daily_schedule.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate F12-UC2 protection slots and daily schedule."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--slots-output", type=Path, default=DEFAULT_SLOTS_OUT)
    parser.add_argument("--schedule-output", type=Path, default=DEFAULT_SCHEDULE_OUT)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"F12-UC1 zone table not found at {args.input}. "
            "Run scripts/f12_uc1_zone_analysis.py first."
        )

    print(f"Reading F12-UC1 zone table from: {args.input}")
    zone_df = pd.read_csv(args.input, low_memory=False)
    zone_df["timestamp"] = pd.to_datetime(
        zone_df["timestamp"], errors="coerce", utc=True
    )
    zone_df = zone_df.dropna(subset=["timestamp", "plant_zone_label"]).sort_values(
        "timestamp"
    )
    print(f"Loaded {len(zone_df)} zone row(s).")

    slots_df = generate_protection_slots(zone_df)
    schedule_df = generate_daily_schedule(slots_df)

    args.slots_output.parent.mkdir(parents=True, exist_ok=True)
    slots_df.to_csv(args.slots_output, index=False)
    schedule_df.to_csv(args.schedule_output, index=False)

    print(f"Wrote {args.slots_output} ({len(slots_df)} slot(s)).")
    print(f"Wrote {args.schedule_output} ({len(schedule_df)} schedule row(s)).")

    counts = slots_df["slot_type"].value_counts().to_dict() if len(slots_df) else {}
    summary = {
        "safe_slot": int(counts.get("safe_slot", 0)),
        "danger_slot": int(counts.get("danger_slot", 0)),
        "transition_slot": int(counts.get("transition_slot", 0)),
        "neutral": int(counts.get("neutral", 0)),
    }
    print("Slot summary:", summary)


if __name__ == "__main__":
    main()
