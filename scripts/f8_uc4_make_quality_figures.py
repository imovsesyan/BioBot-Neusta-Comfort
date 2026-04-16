"""Create F8-UC4 quality figures from the cleaning summary.

Run from the repository root after F8-UC4:
    .venv/bin/python scripts/f8_uc4_make_quality_figures.py
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
DEFAULT_SUMMARY = ROOT / "reports" / "tables" / "f8_uc4_cleaning_summary.json"
DEFAULT_OUTPUT_DIR = ROOT / "reports" / "figures"


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def save_row_count_figure(summary: dict, output_dir: Path) -> Path:
    rows = []
    for source, profile in summary.items():
        rows.append({"source": source, "stage": "input", "rows": profile["input_rows"]})
        rows.append({"source": source, "stage": "aggregated", "rows": profile["output_rows"]})
    df = pd.DataFrame(rows)
    pivot = df.pivot(index="source", columns="stage", values="rows").loc[
        ["iot", "aquacheck", "neusta", "meteo_france"]
    ]

    ax = pivot.plot(kind="bar", figsize=(9, 5), color=["#2f6f7e", "#d59635"])
    ax.set_title("F8-UC4 row counts before and after aggregation")
    ax.set_xlabel("")
    ax.set_ylabel("Rows")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()

    path = output_dir / "f8_uc4_row_counts.png"
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def save_quality_bar(
    rows: list[dict],
    value_column: str,
    title: str,
    output_path: Path,
) -> Path:
    df = pd.DataFrame(rows)
    if df.empty:
        return output_path

    df = df[df[value_column] > 0].copy()
    if df.empty:
        return output_path

    df["label"] = df["source"] + " / " + df["variable"]
    df = df.sort_values(value_column).tail(25)

    fig_height = max(5, 0.28 * len(df))
    ax = df.plot.barh(x="label", y=value_column, figsize=(10, fig_height), legend=False, color="#6b8f71")
    ax.set_title(title)
    ax.set_xlabel("Count")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def save_out_of_range_figure(summary: dict, output_dir: Path) -> Path:
    rows = []
    for source, profile in summary.items():
        for variable, details in profile["range_rules"].items():
            rows.append(
                {
                    "source": source,
                    "variable": variable,
                    "out_of_range": details["out_of_range_replaced_with_null"],
                }
            )
    return save_quality_bar(
        rows,
        "out_of_range",
        "F8-UC4 values replaced with null by range rules",
        output_dir / "f8_uc4_out_of_range_counts.png",
    )


def save_imputation_figure(summary: dict, output_dir: Path) -> Path:
    rows = []
    for source, profile in summary.items():
        for variable, count in profile["imputed_values"].items():
            rows.append({"source": source, "variable": variable, "imputed": count})
    return save_quality_bar(
        rows,
        "imputed",
        "F8-UC4 values imputed after aggregation",
        output_dir / "f8_uc4_imputation_counts.png",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Create F8-UC4 quality figures.")
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)
    paths = [
        save_row_count_figure(summary, args.output_dir),
        save_out_of_range_figure(summary, args.output_dir),
        save_imputation_figure(summary, args.output_dir),
    ]

    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
