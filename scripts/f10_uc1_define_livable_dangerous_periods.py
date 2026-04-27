"""F10-UC1: Determine livable and dangerous periods from humidex thresholds.

Run from the repository root:
    python scripts/f10_uc1_define_livable_dangerous_periods.py
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

from biobot.risk.rules import (  # noqa: E402
    DEFAULT_LIVABILITY_THRESHOLD,
    LIVABILITY_STATUS_ORDER,
    RISK_LEVEL_DETAILS,
    RISK_LEVEL_ORDER,
    add_livability_score_status,
    add_risk_labels,
    livability_status_counts,
    risk_counts,
)


DEFAULT_METEO = ROOT / "data" / "processed" / "meteo_france_1h_clean.csv"
DEFAULT_NEUSTA = ROOT / "data" / "processed" / "neusta_15min_clean.csv"
DEFAULT_F9_PREDICTIONS = ROOT / "reports" / "tables" / "f9_uc7_livability_test_predictions.csv"
DEFAULT_METEO_LABELS = ROOT / "data" / "processed" / "f10_meteo_risk_labels.csv"
DEFAULT_NEUSTA_LABELS = ROOT / "data" / "processed" / "f10_neusta_risk_labels.csv"
DEFAULT_SUMMARY = ROOT / "reports" / "tables" / "f10_uc1_livable_dangerous_summary.json"
DEFAULT_EXAMPLES = ROOT / "reports" / "tables" / "f10_uc1_risk_period_examples.csv"
DEFAULT_SCORE_SUMMARY = ROOT / "reports" / "tables" / "f10_uc1_livability_score_status_summary.json"
DEFAULT_SCORE_EXAMPLES = ROOT / "reports" / "tables" / "f10_uc1_livability_score_examples.csv"
DEFAULT_DISTRIBUTION_FIGURE = ROOT / "reports" / "figures" / "f10_uc1_risk_level_distribution.png"
DEFAULT_TIMELINE_FIGURE = ROOT / "reports" / "figures" / "f10_uc1_high_risk_timeline.png"
DEFAULT_SCORE_FIGURE = ROOT / "reports" / "figures" / "f10_uc1_livability_score_status.png"


def load_and_label(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = add_risk_labels(df)
    return df.dropna(subset=["timestamp_utc", "risk_level"]).sort_values("timestamp_utc")


def add_neusta_livability_status(df: pd.DataFrame) -> pd.DataFrame:
    if "vivabilite_binary_mean" not in df.columns:
        return df
    return add_livability_score_status(df, score_column="vivabilite_binary_mean")


def summarize_source(df: pd.DataFrame, source: str, input_file: Path, output_file: Path) -> dict:
    humidex = pd.to_numeric(df["humidex_c"], errors="coerce")
    counts = risk_counts(df)
    return {
        "source": source,
        "input_file": str(input_file),
        "output_file": str(output_file),
        "rows_with_risk_label": int(len(df)),
        "date_min": df["timestamp_utc"].min().isoformat() if len(df) else None,
        "date_max": df["timestamp_utc"].max().isoformat() if len(df) else None,
        "humidex_min": float(humidex.min()) if len(humidex) else None,
        "humidex_max": float(humidex.max()) if len(humidex) else None,
        "risk_counts": counts,
        "risk_percentages": {
            level: float(100 * count / len(df)) if len(df) else 0.0
            for level, count in counts.items()
        },
        "critical_humidex_rows": int(df["is_critical_humidex"].sum()),
    }


def load_prediction_status(path: Path) -> pd.DataFrame:
    predictions = pd.read_csv(path)
    predictions["timestamp_utc"] = pd.to_datetime(
        predictions["timestamp_utc"],
        errors="coerce",
        utc=True,
    )
    predictions = predictions.dropna(subset=["timestamp_utc", "predicted"])
    predictions = add_livability_score_status(predictions, score_column="predicted")
    predictions = predictions.rename(
        columns={
            "actual": "actual_livability_score",
            "predicted": "predicted_livability_score",
        }
    )
    if "actual_livability_score" in predictions.columns:
        actual_status = add_livability_score_status(
            predictions[["actual_livability_score"]].rename(
                columns={"actual_livability_score": "actual"}
            ),
            score_column="actual",
        )
        predictions["actual_livability_status"] = actual_status["livability_status"].values
    return predictions.sort_values("timestamp_utc")


def summarize_livability_score_status(
    df: pd.DataFrame,
    source: str,
    score_column: str,
    input_file: Path,
) -> dict:
    score = pd.to_numeric(df[score_column], errors="coerce")
    counts = livability_status_counts(df)
    return {
        "source": source,
        "input_file": str(input_file),
        "score_column": score_column,
        "threshold": DEFAULT_LIVABILITY_THRESHOLD,
        "score_interpretation": "Current Neusta binary target: score >= 0.5 means not livable.",
        "rows_with_status": int(len(df)),
        "score_min": float(score.min()) if len(score) else None,
        "score_max": float(score.max()) if len(score) else None,
        "score_mean": float(score.mean()) if len(score) else None,
        "status_counts": counts,
        "status_percentages": {
            status: float(100 * count / len(df)) if len(df) else 0.0
            for status, count in counts.items()
        },
    }


def save_distribution_figure(summaries: dict[str, dict], path: Path) -> None:
    plot_rows = []
    for source, summary in summaries.items():
        for level in RISK_LEVEL_ORDER:
            plot_rows.append(
                {
                    "source": source,
                    "risk_level": level,
                    "rows": summary["risk_counts"][level],
                }
            )
    plot_df = pd.DataFrame(plot_rows)
    pivot = plot_df.pivot(index="risk_level", columns="source", values="rows").reindex(
        RISK_LEVEL_ORDER
    )
    ax = pivot.plot.bar(figsize=(9, 4.5), color=["#2f6f7e", "#d59635"])
    ax.set_title("F10-UC1 risk-level distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def save_high_risk_timeline(meteo_df: pd.DataFrame, path: Path) -> None:
    work = meteo_df[meteo_df["risk_level"].isin(["high_risk", "dangerous"])].copy()
    work["month"] = work["timestamp_utc"].dt.tz_convert(None).dt.to_period("M").dt.to_timestamp()
    timeline = (
        work.groupby(["month", "risk_level"], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["high_risk", "dangerous"], fill_value=0)
    )
    ax = timeline.plot(figsize=(10, 4.5), color=["#d59635", "#b94040"], linewidth=2)
    ax.set_title("F10-UC1 Meteo France high-risk and dangerous periods")
    ax.set_xlabel("Month")
    ax.set_ylabel("Rows")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def save_livability_status_figure(summaries: dict[str, dict], path: Path) -> None:
    rows = []
    for source, summary in summaries.items():
        for status in LIVABILITY_STATUS_ORDER:
            rows.append(
                {
                    "source": source,
                    "livability_status": status,
                    "rows": summary["status_counts"][status],
                }
            )
    plot_df = pd.DataFrame(rows)
    pivot = plot_df.pivot(
        index="livability_status",
        columns="source",
        values="rows",
    ).reindex(LIVABILITY_STATUS_ORDER)
    ax = pivot.plot.bar(figsize=(8, 4), color=["#2f6f7e", "#d59635"])
    ax.set_title("F10-UC1 livability score status")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def save_examples(meteo_df: pd.DataFrame, neusta_df: pd.DataFrame, path: Path) -> None:
    columns = [
        "source",
        "timestamp_utc",
        "station_id",
        "station_name",
        "temperature_c",
        "relative_humidity_pct",
        "humidex_c",
        "risk_level",
        "risk_score",
        "risk_meaning",
        "is_critical_humidex",
    ]
    meteo_examples = meteo_df[meteo_df["risk_level"].isin(["high_risk", "dangerous"])].copy()
    meteo_examples["source"] = "meteo_france"
    neusta_examples = neusta_df[neusta_df["risk_level"] != "livable"].copy()
    neusta_examples["source"] = "neusta"
    examples = pd.concat([meteo_examples, neusta_examples], ignore_index=True)
    if examples.empty:
        examples = pd.concat([meteo_df.head(20), neusta_df.head(20)], ignore_index=True)
        examples["source"] = examples.get("source", "unknown")
    examples = examples.sort_values("humidex_c", ascending=False).head(100)
    for column in columns:
        if column not in examples.columns:
            examples[column] = None
    path.parent.mkdir(parents=True, exist_ok=True)
    examples[columns].to_csv(path, index=False)


def save_livability_score_examples(
    neusta_df: pd.DataFrame,
    predictions_df: pd.DataFrame,
    path: Path,
) -> None:
    rows = []
    if "livability_status" in neusta_df.columns:
        neusta_examples = neusta_df[
            [
                "timestamp_utc",
                "temperature_c",
                "relative_humidity_pct",
                "humidex_c",
                "vivabilite_binary_mean",
                "livability_status",
            ]
        ].copy()
        neusta_examples["source"] = "neusta_actual_score"
        neusta_examples = neusta_examples.rename(
            columns={"vivabilite_binary_mean": "score_used"}
        )
        rows.append(neusta_examples)
    if len(predictions_df):
        prediction_examples = predictions_df[
            [
                "timestamp_utc",
                "predicted_livability_score",
                "livability_status",
                "actual_livability_score",
                "actual_livability_status",
                "model",
            ]
        ].copy()
        prediction_examples["source"] = "f9_predicted_score"
        prediction_examples = prediction_examples.rename(
            columns={"predicted_livability_score": "score_used"}
        )
        rows.append(prediction_examples)
    examples = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if len(examples):
        examples = examples.sort_values(["livability_status", "timestamp_utc"]).head(100)
    path.parent.mkdir(parents=True, exist_ok=True)
    examples.to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Define F10 livable and dangerous periods.")
    parser.add_argument("--meteo", type=Path, default=DEFAULT_METEO)
    parser.add_argument("--neusta", type=Path, default=DEFAULT_NEUSTA)
    parser.add_argument("--f9-predictions", type=Path, default=DEFAULT_F9_PREDICTIONS)
    parser.add_argument("--meteo-labels", type=Path, default=DEFAULT_METEO_LABELS)
    parser.add_argument("--neusta-labels", type=Path, default=DEFAULT_NEUSTA_LABELS)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--examples", type=Path, default=DEFAULT_EXAMPLES)
    parser.add_argument("--score-summary", type=Path, default=DEFAULT_SCORE_SUMMARY)
    parser.add_argument("--score-examples", type=Path, default=DEFAULT_SCORE_EXAMPLES)
    parser.add_argument("--distribution-figure", type=Path, default=DEFAULT_DISTRIBUTION_FIGURE)
    parser.add_argument("--timeline-figure", type=Path, default=DEFAULT_TIMELINE_FIGURE)
    parser.add_argument("--score-figure", type=Path, default=DEFAULT_SCORE_FIGURE)
    args = parser.parse_args()

    meteo_df = load_and_label(args.meteo)
    neusta_df = add_neusta_livability_status(load_and_label(args.neusta))
    prediction_df = load_prediction_status(args.f9_predictions) if args.f9_predictions.exists() else pd.DataFrame()

    args.meteo_labels.parent.mkdir(parents=True, exist_ok=True)
    args.neusta_labels.parent.mkdir(parents=True, exist_ok=True)
    meteo_df.to_csv(args.meteo_labels, index=False)
    neusta_df.to_csv(args.neusta_labels, index=False)

    summaries = {
        "meteo_france": summarize_source(meteo_df, "meteo_france", args.meteo, args.meteo_labels),
        "neusta": summarize_source(neusta_df, "neusta", args.neusta, args.neusta_labels),
    }
    summary = {
        "task": "F10-UC1 livable and dangerous periods",
        "risk_rules": RISK_LEVEL_DETAILS,
        "sources": summaries,
        "scope_note": "Current risk labels are humidex-rule labels, not independent medical or human-comfort annotations.",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    score_summaries = {
        "neusta_actual_score": summarize_livability_score_status(
            neusta_df.dropna(subset=["livability_status"]),
            "neusta_actual_score",
            "vivabilite_binary_mean",
            args.neusta,
        )
    }
    if len(prediction_df):
        score_summaries["f9_predicted_score"] = summarize_livability_score_status(
            prediction_df.dropna(subset=["livability_status"]),
            "f9_predicted_score",
            "predicted_livability_score",
            args.f9_predictions,
        )
    score_summary = {
        "task": "F10-UC1 livability score status",
        "status_rule": {
            "score_column": "vivabilite_binary_mean or F9 predicted score",
            "threshold": DEFAULT_LIVABILITY_THRESHOLD,
            "current_interpretation": "score < 0.5 is livable; score >= 0.5 is not livable",
            "reason": "In the current Neusta binary target, higher values occur during warmer, less comfortable periods.",
            "validation_needed": "Confirm the meaning of Neusta vivabilite_binary with the project owner.",
        },
        "sources": score_summaries,
    }
    args.score_summary.parent.mkdir(parents=True, exist_ok=True)
    args.score_summary.write_text(json.dumps(score_summary, indent=2), encoding="utf-8")
    save_examples(meteo_df, neusta_df, args.examples)
    save_livability_score_examples(neusta_df, prediction_df, args.score_examples)
    save_distribution_figure(summaries, args.distribution_figure)
    save_high_risk_timeline(meteo_df, args.timeline_figure)
    save_livability_status_figure(score_summaries, args.score_figure)

    print(f"Wrote {args.meteo_labels}")
    print(f"Wrote {args.neusta_labels}")
    print(f"Wrote {args.summary}")
    print(f"Wrote {args.examples}")
    print(f"Wrote {args.score_summary}")
    print(f"Wrote {args.score_examples}")
    print(f"Wrote {args.distribution_figure}")
    print(f"Wrote {args.timeline_figure}")
    print(f"Wrote {args.score_figure}")
    print("Meteo risk counts:", summaries["meteo_france"]["risk_counts"])
    print("Neusta risk counts:", summaries["neusta"]["risk_counts"])
    print(
        "Neusta score status counts:",
        score_summaries["neusta_actual_score"]["status_counts"],
    )


if __name__ == "__main__":
    main()
