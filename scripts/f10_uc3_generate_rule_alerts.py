"""F10-UC3: Generate rule-based alerts from F10 risk labels.

Run from the repository root:
    python scripts/f10_uc3_generate_rule_alerts.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.risk.rules import add_risk_labels, create_rule_alerts


DEFAULT_INPUT = ROOT / "data" / "processed" / "f10_meteo_risk_labels.csv"
DEFAULT_FALLBACK_METEO = ROOT / "data" / "processed" / "meteo_france_1h_clean.csv"
DEFAULT_ALERTS = ROOT / "data" / "processed" / "f10_meteo_rule_alerts.csv"
DEFAULT_SUMMARY = ROOT / "reports" / "tables" / "f10_uc3_rule_alert_summary.json"
DEFAULT_EXAMPLES = ROOT / "reports" / "tables" / "f10_uc3_alert_examples.csv"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "f10_uc3_alert_severity_distribution.png"


def load_risk_labels(path: Path, fallback_meteo: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_csv(fallback_meteo, low_memory=False)
        df = add_risk_labels(df)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df.dropna(subset=["timestamp_utc", "risk_level"]).sort_values("timestamp_utc")


def save_alert_figure(alerts: pd.DataFrame, path: Path) -> None:
    counts = alerts["alert_severity"].value_counts().reindex(
        ["info", "warning", "danger", "critical"],
        fill_value=0,
    )
    ax = counts.plot.bar(figsize=(8, 4), color=["#7aa6a1", "#d59635", "#b94040", "#6b1f1f"])
    ax.set_title("F10-UC3 rule-based alert severity distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Alerts")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate F10 rule-based alerts.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fallback-meteo", type=Path, default=DEFAULT_FALLBACK_METEO)
    parser.add_argument("--alerts", type=Path, default=DEFAULT_ALERTS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    risk_df = load_risk_labels(args.input, args.fallback_meteo)
    alerts = create_rule_alerts(risk_df)

    args.alerts.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(args.alerts, index=False)

    alert_counts = alerts["alert_severity"].value_counts().to_dict()
    alert_type_counts = alerts["alert_type"].value_counts().to_dict()
    summary = {
        "task": "F10-UC3 rule-based alerts",
        "input_file": str(args.input if args.input.exists() else args.fallback_meteo),
        "output_file": str(args.alerts),
        "total_input_rows": int(len(risk_df)),
        "total_alerts": int(len(alerts)),
        "alert_severity_counts": {key: int(value) for key, value in alert_counts.items()},
        "alert_type_counts": {key: int(value) for key, value in alert_type_counts.items()},
        "critical_alerts": int((alerts["alert_severity"] == "critical").sum()),
        "scope_note": "Alerts are simulated rule-based events. They are not connected to a production notification system yet.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    columns = [
        "timestamp_utc",
        "station_id",
        "station_name",
        "temperature_c",
        "relative_humidity_pct",
        "humidex_c",
        "risk_level",
        "alert_severity",
        "alert_type",
        "alert_message",
    ]
    examples = alerts.sort_values(["risk_score", "humidex_c"], ascending=False).head(100)
    args.examples.parent.mkdir(parents=True, exist_ok=True)
    examples[columns].to_csv(args.examples, index=False)
    save_alert_figure(alerts, args.figure)

    print(f"Wrote {args.alerts}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.examples}")
    print(f"Wrote {args.figure}")
    print("Alert severity counts:", summary["alert_severity_counts"])


if __name__ == "__main__":
    main()

