"""F13-UC1 integration script: per-row plant health and irrigation assessment.

Loads the processed Meteo France 1h dataset, samples 200 rows, applies the
F10 humidex risk labelling, then runs the F13 sensor-based health assessor and
irrigation recommender on each row.

Output
------
``data/outputs/f13_irrigation_plan.csv`` — one row per sampled timestamp with:
    timestamp, humidex, humidex_risk_level, health_status, stress_cause,
    irrigation_action, suggested_time_slot, irrigation_reason

A summary table (count per health_status and per irrigation_action) is printed
to stdout.

Design notes
------------
* Soil moisture is not available in the Meteo France dataset (it lives in the
  Aquacheck source which is not spatially merged per CLAUDE.md constraints).
  The health assessor is therefore called without ``soil_moisture``; drought
  stress assessment is skipped gracefully.
* Species water need defaults to ``mesic`` — the most common case for
  temperate/suburban outdoor plants in the Neusta deployment area.
* The 200-row sample uses a fixed random seed (42) for reproducibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from biobot.plants.health_assessor import assess_health
from biobot.plants.irrigation_recommender import get_irrigation_advice
from biobot.risk.rules import assign_humidex_risk_level

SAMPLE_SIZE = 200
RANDOM_SEED = 42
INPUT_PATH = Path("data/processed/meteo_france_1h_clean.csv.gz")
OUTPUT_PATH = Path("data/outputs/f13_irrigation_plan.csv")
SPECIES_WATER_NEED = "mesic"


def _safe_risk_level(risk_label) -> str:
    """Map the raw risk_level value to a string safe for the recommender.

    The ``assign_humidex_risk_level`` function returns ``pd.NA`` for NaN rows.
    We fall back to ``livable`` for missing data to keep the pipeline running.
    Also collapses ``is_critical_humidex`` into ``critical`` — the recommender
    accepts all five canonical levels.
    """
    if pd.isna(risk_label):
        return "livable"
    return str(risk_label)


def main() -> None:
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)

    # Drop rows with missing humidex, temperature, or humidity — the assessors
    # require real values for all three core inputs.
    required = ["humidex_c", "temperature_c", "relative_humidity_pct"]
    df = df.dropna(subset=required).reset_index(drop=True)
    print(f"  Rows after dropping NaN in core fields: {len(df):,}")

    sample = df.sample(n=min(SAMPLE_SIZE, len(df)), random_state=RANDOM_SEED).copy()
    print(f"  Sampled {len(sample)} rows for assessment.")

    # Apply F10 risk labels.
    risk_df = assign_humidex_risk_level(sample["humidex_c"])
    sample["risk_level"] = risk_df["risk_level"].values
    sample["is_critical_humidex"] = risk_df["is_critical_humidex"].values

    # Elevate is_critical_humidex rows to the ``critical`` label so the
    # recommender gets the full five-level vocabulary.
    sample.loc[sample["is_critical_humidex"], "risk_level"] = "critical"

    # Parse timestamp hour for the time-of-day filter.
    sample["timestamp_dt"] = pd.to_datetime(sample["timestamp_utc"], utc=True, errors="coerce")
    sample["hour"] = sample["timestamp_dt"].dt.hour

    records = []
    for _, row in sample.iterrows():
        risk_level = _safe_risk_level(row["risk_level"])
        humidex = float(row["humidex_c"])
        temperature_c = float(row["temperature_c"])
        relative_humidity = float(row["relative_humidity_pct"])
        hour = int(row["hour"]) if pd.notna(row["hour"]) else None

        health = assess_health(
            humidex=humidex,
            temperature_c=temperature_c,
            relative_humidity=relative_humidity,
            soil_moisture=None,
            species_water_need=SPECIES_WATER_NEED,
        )

        advice = get_irrigation_advice(
            humidex_risk_level=risk_level,
            health_assessment=health,
            species_water_need=SPECIES_WATER_NEED,
            soil_moisture=None,
            hour=hour,
        )

        records.append(
            {
                "timestamp": row["timestamp_utc"],
                "humidex": humidex,
                "humidex_risk_level": risk_level,
                "health_status": health.health_status,
                "stress_cause": health.stress_cause if health.stress_cause else "none",
                "irrigation_action": advice.action,
                "suggested_time_slot": advice.suggested_time_slot,
                "irrigation_reason": advice.reason,
            }
        )

    out_df = pd.DataFrame(records)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out_df)} rows to {OUTPUT_PATH}")

    print("\n--- Health status summary ---")
    print(out_df["health_status"].value_counts().to_string())

    print("\n--- Irrigation action summary ---")
    print(out_df["irrigation_action"].value_counts().to_string())


if __name__ == "__main__":
    main()
