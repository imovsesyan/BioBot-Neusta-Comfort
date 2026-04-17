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
import numpy as np
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
TREE_ENSEMBLE_MEMBERS = ["random_forest", "hist_gradient_boosting", "xgboost"]


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


def clip_score(predictions) -> pd.Series:
    """Keep livability score predictions inside the known 0..1 target range."""

    return pd.Series(predictions).clip(0.0, 1.0)


def score_predictions(
    model_name: str,
    y_validation: pd.Series,
    validation_pred: pd.Series,
    y_test: pd.Series,
    test_pred: pd.Series,
    extra: dict | None = None,
) -> dict:
    """Create a result row from validation and test predictions."""

    validation_metrics = regression_metrics(y_validation.to_numpy(), validation_pred.to_numpy())
    test_metrics = regression_metrics(y_test.to_numpy(), test_pred.to_numpy())
    row = {
        "model": model_name,
        "validation_mae": validation_metrics["mae"],
        "validation_rmse": validation_metrics["rmse"],
        "validation_r2": validation_metrics["r2"],
        "test_mae": test_metrics["mae"],
        "test_rmse": test_metrics["rmse"],
        "test_r2": test_metrics["r2"],
    }
    if extra:
        row.update(extra)
    return row


def weighted_average_predictions(
    predictions_by_model: dict[str, dict[str, pd.Series]],
    model_names: list[str],
    weights: dict[str, float],
    prediction_key: str,
) -> pd.Series:
    """Blend predictions from several already fitted models."""

    blended = np.zeros(len(predictions_by_model[model_names[0]][prediction_key]), dtype=float)
    for model_name in model_names:
        blended += weights[model_name] * predictions_by_model[model_name][prediction_key].to_numpy()
    return clip_score(blended)


def inverse_validation_mae_weights(results: list[dict], model_names: list[str]) -> dict[str, float]:
    """Weight stronger validation models more heavily."""

    mae_by_model = {
        result["model"]: result["validation_mae"]
        for result in results
        if result["model"] in model_names
    }
    inverse_scores = {
        model_name: 1.0 / max(mae_by_model[model_name], 1e-12)
        for model_name in model_names
    }
    total = sum(inverse_scores.values())
    return {model_name: inverse_scores[model_name] / total for model_name in model_names}


def add_tree_blend_ensembles(
    results: list[dict],
    predictions_by_model: dict[str, dict[str, pd.Series]],
    y_validation: pd.Series,
    y_test: pd.Series,
) -> None:
    """Add prediction-level ensembles made from the strongest tree models."""

    available_members = [
        model_name
        for model_name in TREE_ENSEMBLE_MEMBERS
        if model_name in predictions_by_model
    ]
    if len(available_members) < 2:
        return

    ensemble_specs = {
        "tree_blend_equal_weight": {
            model_name: 1.0 / len(available_members)
            for model_name in available_members
        },
        "tree_blend_validation_weighted": inverse_validation_mae_weights(
            results,
            available_members,
        ),
    }

    for ensemble_name, weights in ensemble_specs.items():
        validation_pred = weighted_average_predictions(
            predictions_by_model,
            available_members,
            weights,
            "validation_pred",
        )
        test_pred = weighted_average_predictions(
            predictions_by_model,
            available_members,
            weights,
            "test_pred",
        )
        results.append(
            score_predictions(
                ensemble_name,
                y_validation,
                validation_pred,
                y_test,
                test_pred,
                extra={
                    "model_family": "prediction_blend_ensemble",
                    "ensemble_members": available_members,
                    "ensemble_weights": weights,
                },
            )
        )
        predictions_by_model[ensemble_name] = {
            "validation_pred": validation_pred,
            "test_pred": test_pred,
        }


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
    predictions_by_model = {}
    models, skipped_models = build_models(args.random_state)
    for model_name, model in models.items():
        model.fit(x_train, y_train)
        validation_pred = clip_score(model.predict(x_validation))
        test_pred = clip_score(model.predict(x_test))
        results.append(score_predictions(model_name, y_validation, validation_pred, y_test, test_pred))
        predictions_by_model[model_name] = {
            "validation_pred": validation_pred,
            "test_pred": test_pred,
        }

    add_tree_blend_ensembles(results, predictions_by_model, y_validation, y_test)

    best_result = min(results, key=lambda item: item["test_mae"])
    best_model_name = best_result["model"]
    best_test_predictions = predictions_by_model[best_model_name]["test_pred"]

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
