"""F12-UC1: Plant zone labeling and diurnal pattern analysis.

Pipeline
--------
1. Load a single source of F10 humidex-labelled rows. Preference order:
     a. ``reports/tables/f10_uc1_risk_period_examples.csv``
     b. ``reports/tables/f10_uc3_alert_examples.csv``
     c. Fallback: ``data/processed/meteo_france_1h_clean.csv.gz`` labelled
        on the fly via ``biobot.risk.rules.add_risk_labels``.
2. Promote rows where ``humidex_c > 54`` (or ``alert_severity == 'critical'``)
   from ``risk_level == 'dangerous'`` to ``risk_level == 'critical'`` so the
   plant-zone mapping sees the full 5-band space (matches F11 convention).
3. Apply :func:`biobot.zones.assign_plant_zone_label` per row.
4. Compute ``diurnal_window``, ``hour``, and ``day_of_week``.
5. Save the per-row table to ``data/outputs/f12_zones_periods.csv``.
6. Render the hour x day-of-week heatmap of the modal plant zone label to
   ``reports/figures/f12_uc1_zone_heatmap.png``.

Scientific notes
----------------
* Plant zone labels are derived **exclusively from raw humidex thresholds**
  (``biobot.risk.rules``). No use of ``vivabilite_binary_mean`` or F10-UC4
  classifier outputs, in line with CLAUDE.md target-leakage guardrails.
* The four upstream sources are not spatially/temporally aligned. This
  script reads ONE source per run.

Run from the repository root::

    MPLCONFIGDIR=.cache/matplotlib python scripts/f12_uc1_zone_analysis.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.risk.rules import add_risk_labels  # noqa: E402
from biobot.zones.temporal import (  # noqa: E402
    PLANT_ZONE_LABELS,
    assign_diurnal_window,
    assign_plant_zone_label,
)


# Candidate F10 inputs in preference order (UC1 first per task spec).
CANDIDATE_INPUTS = [
    ROOT / "reports" / "tables" / "f10_uc1_risk_period_examples.csv",
    ROOT / "reports" / "tables" / "f10_uc3_alert_examples.csv",
]
FALLBACK_RAW = ROOT / "data" / "processed" / "meteo_france_1h_clean.csv.gz"

OUTPUT_DIR = ROOT / "data" / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "f12_zones_periods.csv"
FIGURE_PATH = ROOT / "reports" / "figures" / "f12_uc1_zone_heatmap.png"

# Discrete palette: favorable=green, moderate=yellow, dangerous=orange, critical=red.
PLANT_ZONE_COLORS = {
    "favorable": "#2e8b57",
    "moderate": "#f4d03f",
    "dangerous": "#e67e22",
    "critical": "#c0392b",
}

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ---------------------------------------------------------------------------
# Input selection / loading
# ---------------------------------------------------------------------------


def _select_input(explicit: Optional[Path]) -> Path:
    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit input not found: {explicit}")
        return explicit

    for candidate in CANDIDATE_INPUTS:
        if candidate.exists():
            return candidate
    if FALLBACK_RAW.exists():
        return FALLBACK_RAW
    raise FileNotFoundError(
        "No F10 humidex-labelled CSV found. Looked at: "
        + ", ".join(str(p) for p in CANDIDATE_INPUTS + [FALLBACK_RAW])
    )


def _load_humidex_rows(path: Path) -> pd.DataFrame:
    """Load the chosen input and ensure (timestamp, humidex_c, risk_level)."""
    df = pd.read_csv(path, low_memory=False)

    # Synthesize risk labels from raw humidex when absent (fallback path).
    if "risk_level" not in df.columns:
        if "humidex_c" not in df.columns:
            raise ValueError(
                f"Input {path} has neither 'risk_level' nor 'humidex_c'."
            )
        df = add_risk_labels(df)

    if "humidex_c" not in df.columns:
        raise ValueError(f"Input {path} is missing 'humidex_c'.")

    # Promote dangerous -> critical when humidex > 54. Two equivalent signals
    # may carry this: alert_severity == 'critical' (alerts CSV) or the
    # is_critical_humidex flag emitted by add_risk_labels (raw fallback).
    if "alert_severity" in df.columns:
        crit_mask = df["alert_severity"].astype("string") == "critical"
        df.loc[crit_mask, "risk_level"] = "critical"
    if "is_critical_humidex" in df.columns:
        crit_mask = (
            df["is_critical_humidex"].astype("boolean").fillna(False)
        )
        df.loc[crit_mask, "risk_level"] = "critical"

    # Direct humidex check as a final safety net (handles inputs that lack
    # both flag columns yet still contain humidex_c).
    humidex_numeric = pd.to_numeric(df["humidex_c"], errors="coerce")
    df.loc[humidex_numeric > 54, "risk_level"] = "critical"

    # Locate the timestamp column. Both example CSVs use timestamp_utc; the
    # raw cleaned Meteo CSV also uses timestamp_utc.
    ts_col = "timestamp_utc" if "timestamp_utc" in df.columns else "timestamp"
    if ts_col not in df.columns:
        raise ValueError(
            f"Input {path} has no timestamp column (expected 'timestamp_utc')."
        )

    df = df.rename(columns={ts_col: "timestamp"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    df["humidex_c"] = humidex_numeric

    keep = ["timestamp", "humidex_c", "risk_level"]
    df = df.dropna(subset=keep)
    df = df[df["risk_level"].isin(set(PLANT_ZONE_LABELS_INPUTS()))].copy()
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def PLANT_ZONE_LABELS_INPUTS() -> list[str]:
    """Allowed input vocabulary on the humidex side (5-band)."""
    return ["livable", "discomfort", "high_risk", "dangerous", "critical"]


# ---------------------------------------------------------------------------
# Zone analysis
# ---------------------------------------------------------------------------


def build_zone_table(df: pd.DataFrame) -> pd.DataFrame:
    """Append plant_zone_label, diurnal_window, hour, day_of_week."""
    out = df.copy()
    out["plant_zone_label"] = out["risk_level"].map(assign_plant_zone_label)
    out["hour"] = out["timestamp"].dt.hour.astype(int)
    out["day_of_week"] = out["timestamp"].dt.dayofweek.astype(int)
    out["diurnal_window"] = out["hour"].apply(assign_diurnal_window)
    out = out.rename(columns={"risk_level": "humidex_risk_level"})
    return out[
        [
            "timestamp",
            "humidex_c",
            "humidex_risk_level",
            "plant_zone_label",
            "diurnal_window",
            "hour",
            "day_of_week",
        ]
    ]


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def _modal_zone_grid(zone_df: pd.DataFrame) -> np.ndarray:
    """Return a (7 days x 24 hours) grid of integer-coded modal plant zones.

    Cells with no data are coded -1 and rendered as white in the heatmap.
    """
    label_to_idx = {label: i for i, label in enumerate(PLANT_ZONE_LABELS)}
    grid = np.full((7, 24), -1, dtype=int)

    for (dow, hour), group in zone_df.groupby(["day_of_week", "hour"]):
        modes = group["plant_zone_label"].mode()
        if len(modes):
            grid[int(dow), int(hour)] = label_to_idx.get(modes.iloc[0], -1)
    return grid


def save_zone_heatmap(zone_df: pd.DataFrame, path: Path) -> None:
    grid = _modal_zone_grid(zone_df)

    # Build a colormap with an extra "no data" colour at index -1.
    palette = ["#ffffff"] + [PLANT_ZONE_COLORS[lbl] for lbl in PLANT_ZONE_LABELS]
    cmap = ListedColormap(palette)
    # Bounds: -1.5, -0.5, 0.5, ..., len(PLANT_ZONE_LABELS)-0.5
    bounds = [-1.5] + [i + 0.5 for i in range(-1, len(PLANT_ZONE_LABELS))]
    norm = BoundaryNorm(bounds, cmap.N)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xticks(range(24))
    ax.set_xticklabels([str(h) for h in range(24)])
    ax.set_yticks(range(7))
    ax.set_yticklabels(DAY_NAMES)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Day of week")
    ax.set_title("Plant thermal zone by hour x day of week")

    # Legend for the four plant zone colours.
    handles = [
        Patch(facecolor=PLANT_ZONE_COLORS[lbl], edgecolor="black", label=lbl)
        for lbl in PLANT_ZONE_LABELS
    ]
    handles.append(Patch(facecolor="#ffffff", edgecolor="black", label="no data"))
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=len(handles),
        frameon=False,
    )

    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate F12-UC1 plant zone labels and pattern heatmap."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit path to an F10 humidex-labelled CSV.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--figure", type=Path, default=FIGURE_PATH)
    args = parser.parse_args()

    input_path = _select_input(args.input)
    print(f"Reading F10 humidex-labelled rows from: {input_path}")

    raw = _load_humidex_rows(input_path)
    print(f"Loaded {len(raw)} row(s) with humidex_c and risk_level.")

    zone_df = build_zone_table(raw)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    zone_df.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(zone_df)} row(s)).")

    save_zone_heatmap(zone_df, args.figure)
    print(f"Wrote {args.figure}")

    print("Plant zone counts:", zone_df["plant_zone_label"].value_counts().to_dict())


if __name__ == "__main__":
    main()
