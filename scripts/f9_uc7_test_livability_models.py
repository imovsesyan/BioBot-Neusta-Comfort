"""F9-UC7: Test baseline livability score prediction models.

This script predicts Neusta `vivabilite_binary_mean` as the first livability
score target. It uses a chronological split to avoid future leakage.

Run from the repository root:
    python scripts/f9_uc7_test_livability_models.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.modeling.livability_features import (  # noqa: E402
    chronological_split,
    prepare_neusta_livability_table,
)
from biobot.modeling.metrics import regression_metrics  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "processed" / "neusta_15min_clean.csv"
DEFAULT_RESULTS = ROOT / "reports" / "tables" / "f9_uc7_livability_model_results.json"
DEFAULT_PREDICTIONS = ROOT / "reports" / "tables" / "f9_uc7_livability_test_predictions.csv"
DEFAULT_FIGURE = ROOT / "reports" / "figures" / "f9_uc7_model_comparison.png"
DEFAULT_TIMESERIES_FIGURE = ROOT / "reports" / "figures" / "f9_uc7_test_predictions_timeseries.png"


def build_xgboost_model(random_state: int) -> Pipeline | None:
    """Create an XGBoost model when the optional dependency is installed."""

    try:
        from xgboost import XGBRegressor
    except Exception as exc:
        message = next((line.strip() for line in str(exc).splitlines() if line.strip()), repr(exc))
        print(f"Skipping XGBoost because it could not be imported: {message}")
        return None

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    eval_metric="rmse",
                    n_estimators=400,
                    learning_rate=0.03,
                    max_depth=3,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=random_state,
                    n_jobs=1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_models(random_state: int) -> tuple[dict[str, Pipeline], list[str]]:
    """Create baseline tabular models."""

    models = {
        "mean_baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyRegressor(strategy="mean")),
            ]
        ),
        "ridge_regression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    TransformedTargetRegressor(
                        regressor=Ridge(alpha=1.0),
                        transformer=StandardScaler(),
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        min_samples_leaf=4,
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingRegressor(
                        max_iter=300,
                        learning_rate=0.04,
                        l2_regularization=0.05,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    skipped_models = []
    xgboost_model = build_xgboost_model(random_state)
    if xgboost_model is None:
        skipped_models.append("xgboost_missing_optional_dependency")
    else:
        models["xgboost"] = xgboost_model

    return models, skipped_models


def clip_score(predictions):
    """Keep livability score predictions inside the known 0..1 target range."""

    return predictions.clip(0.0, 1.0)


def save_model_comparison_figure(results: list[dict], path: Path) -> None:
    df = pd.DataFrame(results).sort_values("test_mae", ascending=True)
    ax = df.plot.barh(x="model", y="test_mae", legend=False, figsize=(8, 4), color="#2f6f7e")
    ax.set_title("F9-UC7 livability score models: test MAE")
    ax.set_xlabel("MAE, lower is better")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def save_test_prediction_figure(predictions: pd.DataFrame, path: Path) -> None:
    plt.figure(figsize=(11, 4.5))
    plt.plot(
        pd.to_datetime(predictions["timestamp_utc"]),
        predictions["actual"],
        label="Actual",
        linewidth=1.8,
        color="#2f6f7e",
    )
    plt.plot(
        pd.to_datetime(predictions["timestamp_utc"]),
        predictions["predicted"],
        label="Predicted",
        linewidth=1.4,
        color="#d59635",
        alpha=0.9,
    )
    plt.title("F9-UC7 best model test predictions")
    plt.xlabel("Time")
    plt.ylabel("Livability score")
    plt.ylim(-0.05, 1.05)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Test F9 livability prediction models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--figure", type=Path, default=DEFAULT_FIGURE)
    parser.add_argument("--timeseries-figure", type=Path, default=DEFAULT_TIMESERIES_FIGURE)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    df, feature_columns, target_column = prepare_neusta_livability_table(str(args.input))
    train_df, validation_df, test_df = chronological_split(df)

    original_feature_columns = list(feature_columns)
    feature_columns = [column for column in feature_columns if train_df[column].notna().any()]
    dropped_features = sorted(set(original_feature_columns) - set(feature_columns))

    x_train = train_df[feature_columns]
    y_train = train_df[target_column]
    x_validation = validation_df[feature_columns]
    y_validation = validation_df[target_column]
    x_test = test_df[feature_columns]
    y_test = test_df[target_column]

    results = []
    fitted_models = {}
    models, skipped_models = build_models(args.random_state)
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        validation_pred = clip_score(pd.Series(model.predict(x_validation)))
        test_pred = clip_score(pd.Series(model.predict(x_test)))
        validation_metrics = regression_metrics(y_validation.to_numpy(), validation_pred.to_numpy())
        test_metrics = regression_metrics(y_test.to_numpy(), test_pred.to_numpy())
        results.append(
            {
                "model": model_name,
                "validation_mae": validation_metrics["mae"],
                "validation_rmse": validation_metrics["rmse"],
                "validation_r2": validation_metrics["r2"],
                "test_mae": test_metrics["mae"],
                "test_rmse": test_metrics["rmse"],
                "test_r2": test_metrics["r2"],
            }
        )
        fitted_models[model_name] = (model, test_pred)

    best_result = min(results, key=lambda item: item["test_mae"])
    best_model_name = best_result["model"]
    _, best_test_predictions = fitted_models[best_model_name]

    prediction_table = pd.DataFrame(
        {
            "timestamp_utc": test_df["timestamp_utc"].astype(str).to_numpy(),
            "actual": y_test.to_numpy(),
            "predicted": best_test_predictions.to_numpy(),
            "model": best_model_name,
        }
    )

    summary = {
        "task": "F9-UC7 livability score prediction",
        "target": target_column,
        "target_scale": "0..1 interval livability score from Neusta aggregated labels",
        "input_file": str(args.input),
        "n_rows_with_target": int(len(df)),
        "n_features": int(len(feature_columns)),
        "feature_columns": feature_columns,
        "dropped_features": {
            "reason": "No non-missing values in the training split.",
            "columns": dropped_features,
        },
        "split": {
            "method": "chronological",
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
        },
        "best_model_by_test_mae": best_model_name,
        "results": results,
        "skipped_models": skipped_models,
        "scope_note": "Recommendation systems and model interpretation are intentionally out of scope for this F9 run.",
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(args.predictions, index=False)
    save_model_comparison_figure(results, args.figure)
    save_test_prediction_figure(prediction_table, args.timeseries_figure)

    print(f"Rows with target: {len(df):,}")
    print(f"Features: {len(feature_columns):,}")
    print(f"Best model by test MAE: {best_model_name}")
    for result in sorted(results, key=lambda item: item["test_mae"]):
        print(
            f"{result['model']}: "
            f"test MAE={result['test_mae']:.4f}, "
            f"RMSE={result['test_rmse']:.4f}, "
            f"R2={result['test_r2']:.4f}"
        )
    if skipped_models:
        print("Skipped models:", ", ".join(skipped_models))
    print(f"Wrote {args.results}")
    print(f"Wrote {args.predictions}")
    print(f"Wrote {args.figure}")
    print(f"Wrote {args.timeseries_figure}")


if __name__ == "__main__":
    main()
