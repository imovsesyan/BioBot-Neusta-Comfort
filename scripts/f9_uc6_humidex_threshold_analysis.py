"""F9-UC6: Analyze critical humidex thresholds in processed datasets.

Run from the repository root:
    python scripts/f9_uc6_humidex_threshold_analysis.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_NEUSTA = ROOT / "data" / "processed" / "neusta_15min_clean.csv"
DEFAULT_METEO = ROOT / "data" / "processed" / "meteo_france_1h_clean.csv"
DEFAULT_RESULTS = ROOT / "reports" / "tables" / "f9_uc6_humidex_threshold_summary.json"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "f9_uc6_humidex_distribution.png"


HUMIDEX_LABELS = [
    "below_30_little_or_no_discomfort",
    "30_to_39_some_discomfort",
    "40_to_45_great_discomfort",
    "above_45_to_54_dangerous",
    "above_54_imminent_heat_stroke_risk",
]


def categorize_humidex(humidex: pd.Series) -> pd.Categorical:
    """Assign humidex values to ordered comfort and danger bands."""

    categories = pd.Series(index=humidex.index, dtype="object")
    categories.loc[humidex < 30] = HUMIDEX_LABELS[0]
    categories.loc[(humidex >= 30) & (humidex < 40)] = HUMIDEX_LABELS[1]
    categories.loc[(humidex >= 40) & (humidex <= 45)] = HUMIDEX_LABELS[2]
    categories.loc[(humidex > 45) & (humidex <= 54)] = HUMIDEX_LABELS[3]
    categories.loc[humidex > 54] = HUMIDEX_LABELS[4]
    return pd.Categorical(categories, categories=HUMIDEX_LABELS, ordered=True)


def summarize_humidex(path: Path, source: str) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path, low_memory=False)
    humidex = pd.to_numeric(df["humidex_c"], errors="coerce").dropna()
    categories = categorize_humidex(humidex)
    counts = categories.value_counts().reindex(HUMIDEX_LABELS, fill_value=0)
    summary = {
        "source": source,
        "input_file": str(path),
        "non_null_humidex_rows": int(len(humidex)),
        "min": float(humidex.min()) if len(humidex) else None,
        "max": float(humidex.max()) if len(humidex) else None,
        "mean": float(humidex.mean()) if len(humidex) else None,
        "threshold_counts": {label: int(counts[label]) for label in HUMIDEX_LABELS},
        "threshold_percentages": {
            label: float(100 * counts[label] / len(humidex)) if len(humidex) else 0.0
            for label in HUMIDEX_LABELS
        },
    }
    return humidex.to_frame(name=source), summary


def save_distribution_figure(neusta: pd.Series, meteo: pd.Series, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.hist(neusta, bins=30, alpha=0.75, label="Neusta", color="#2f6f7e")
    plt.hist(meteo, bins=50, alpha=0.45, label="Meteo France", color="#d59635")
    for threshold, label in [(30, "30"), (40, "40"), (45, "45"), (54, "54")]:
        plt.axvline(threshold, color="#333333", linestyle="--", linewidth=1)
        plt.text(threshold + 0.3, plt.ylim()[1] * 0.88, label, rotation=90)
    plt.title("F9-UC6 humidex distribution and critical thresholds")
    plt.xlabel("Humidex")
    plt.ylabel("Rows")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze humidex thresholds.")
    parser.add_argument("--neusta", type=Path, default=DEFAULT_NEUSTA)
    parser.add_argument("--meteo", type=Path, default=DEFAULT_METEO)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    neusta_humidex, neusta_summary = summarize_humidex(args.neusta, "neusta")
    meteo_humidex, meteo_summary = summarize_humidex(args.meteo, "meteo_france")

    summary = {
        "task": "F9-UC6 critical humidex thresholds",
        "threshold_reference": {
            "below_30": "Little or no discomfort",
            "30_to_39": "Some discomfort",
            "40_to_45": "Great discomfort; reduce exertion",
            "above_45": "Dangerous",
            "above_54": "Imminent heat-stroke risk in occupational guidance",
        },
        "sources": {
            "neusta": neusta_summary,
            "meteo_france": meteo_summary,
        },
        "scope_note": "Humidex thresholds are used for safety context and future risk labels; they are not treated as the only livability target.",
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    save_distribution_figure(neusta_humidex["neusta"], meteo_humidex["meteo_france"], args.figure)

    print(f"Wrote {args.results}")
    print(f"Wrote {args.figure}")
    print("Neusta threshold counts:", neusta_summary["threshold_counts"])
    print("Meteo threshold counts:", meteo_summary["threshold_counts"])


if __name__ == "__main__":
    main()
