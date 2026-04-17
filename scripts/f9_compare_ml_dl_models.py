"""Compare F9 classical machine learning models with the CNN-LSTM experiment.

This script does not train models. It combines the already generated F9-UC7
tabular model results and the F9-UC8 CNN-LSTM result into one comparison table.

Run from the repository root:
    python scripts/f9_compare_ml_dl_models.py
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
DEFAULT_ML_RESULTS = ROOT / "reports" / "tables" / "f9_uc7_livability_model_results.json"
DEFAULT_DL_RESULTS = ROOT / "reports" / "tables" / "f9_uc8_sequence_model_results.json"
DEFAULT_OUTPUT_JSON = ROOT / "reports" / "tables" / "f9_ml_vs_dl_comparison.json"
DEFAULT_OUTPUT_CSV = ROOT / "reports" / "tables" / "f9_ml_vs_dl_comparison.csv"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "f9_ml_vs_dl_comparison.png"


MODEL_DESCRIPTIONS = {
    "mean_baseline": ("classical_ml", "Naive mean baseline"),
    "ridge_regression": ("classical_ml", "Linear regression with regularization"),
    "random_forest": ("classical_ml", "Tree ensemble"),
    "hist_gradient_boosting": ("classical_ml", "Gradient boosting tree ensemble"),
    "xgboost": ("classical_ml", "XGBoost tree ensemble"),
    "cnn_lstm": ("deep_learning", "Sequence neural network"),
    "lstm": ("deep_learning", "Sequence neural network"),
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def model_family(model_name: str) -> str:
    return MODEL_DESCRIPTIONS.get(model_name, ("unknown", ""))[0]


def model_description(model_name: str) -> str:
    return MODEL_DESCRIPTIONS.get(model_name, ("unknown", ""))[1]


def build_comparison_rows(ml_results: dict, dl_results: dict) -> list[dict]:
    rows = []
    test_rows = ml_results["split"]["test_rows"]
    for result in ml_results["results"]:
        model_name = result["model"]
        rows.append(
            {
                "model": model_name,
                "family": model_family(model_name),
                "description": model_description(model_name),
                "test_mae": result["test_mae"],
                "test_rmse": result["test_rmse"],
                "test_r2": result["test_r2"],
                "validation_mae": result["validation_mae"],
                "evaluation_rows": test_rows,
                "uses_sequence_window": False,
            }
        )

    dl_model = dl_results["model"]
    dl_metrics = dl_results["test_metrics"]
    rows.append(
        {
            "model": dl_model,
            "family": model_family(dl_model),
            "description": model_description(dl_model),
            "test_mae": dl_metrics["mae"],
            "test_rmse": dl_metrics["rmse"],
            "test_r2": dl_metrics["r2"],
            "validation_mae": None,
            "evaluation_rows": dl_results["test_sequences"],
            "uses_sequence_window": True,
        }
    )
    return rows


def save_comparison_figure(comparison: pd.DataFrame, path: Path) -> None:
    plot_df = comparison.sort_values("test_mae", ascending=True)
    colors = [
        "#2f6f7e" if family == "classical_ml" else "#d59635"
        for family in plot_df["family"].tolist()
    ]
    ax = plot_df.plot.barh(
        x="model",
        y="test_mae",
        legend=False,
        figsize=(9, 5),
        color=colors,
    )
    ax.set_title("F9 livability score prediction: ML vs deep learning")
    ax.set_xlabel("Test MAE, lower is better")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare F9 ML and DL model results.")
    parser.add_argument("--ml-results", type=Path, default=DEFAULT_ML_RESULTS)
    parser.add_argument("--dl-results", type=Path, default=DEFAULT_DL_RESULTS)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    args = parser.parse_args()

    ml_results = load_json(args.ml_results)
    dl_results = load_json(args.dl_results)
    rows = build_comparison_rows(ml_results, dl_results)
    comparison = pd.DataFrame(rows).sort_values("test_mae", ascending=True)
    comparison_records = (
        comparison.astype(object).where(pd.notna(comparison), None).to_dict(orient="records")
    )
    best_row = comparison.iloc[0]

    summary = {
        "task": "F9 ML vs deep learning comparison",
        "target": ml_results["target"],
        "input_file": ml_results["input_file"],
        "comparison_rows": comparison_records,
        "best_model_by_test_mae": best_row["model"],
        "best_family_by_test_mae": best_row["family"],
        "conclusion": (
            "On the current Neusta livability target, classical tabular machine learning "
            "models outperform the CNN-LSTM experiment. This does not prove deep learning "
            "is generally worse; it means the current dataset is small and tabular, and "
            "the target may be formula-derived from environmental variables."
        ),
        "fairness_notes": [
            "The classical ML models and XGBoost use the same chronological test split.",
            "The CNN-LSTM uses the same chronological split but evaluates fewer rows because sequence windows need previous observations.",
            "A stricter future comparison can align all models on exactly the same test timestamps.",
        ],
        "scope_note": "Recommendation systems and model interpretation remain out of scope.",
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)
    save_comparison_figure(comparison, args.figure)

    print(f"Best model: {best_row['model']} ({best_row['family']})")
    for row in comparison.to_dict(orient="records"):
        print(
            f"{row['model']}: family={row['family']}, "
            f"test MAE={row['test_mae']:.4f}, "
            f"RMSE={row['test_rmse']:.4f}, "
            f"R2={row['test_r2']:.4f}"
        )
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.figure}")


if __name__ == "__main__":
    main()
