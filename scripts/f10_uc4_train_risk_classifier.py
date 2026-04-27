"""F10-UC4: Train ML classifiers for rule-based risk levels.

Run from the repository root:
    python scripts/f10_uc4_train_risk_classifier.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.risk.rules import RISK_LEVEL_ORDER, add_risk_labels  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "processed" / "f10_meteo_risk_labels.csv"
DEFAULT_FALLBACK_METEO = ROOT / "data" / "processed" / "meteo_france_1h_clean.csv"
DEFAULT_RESULTS = ROOT / "reports" / "tables" / "f10_uc4_risk_classifier_results.json"
DEFAULT_PREDICTIONS = ROOT / "data" / "processed" / "f10_risk_classifier_test_predictions.csv"
DEFAULT_PREDICTION_EXAMPLES = (
    ROOT / "reports" / "tables" / "f10_uc4_risk_classifier_prediction_examples.csv"
)
DEFAULT_COMPARISON_FIGURE = ROOT / "reports" / "figures" / "f10_uc4_classifier_comparison.png"
DEFAULT_CONFUSION_FIGURE = ROOT / "reports" / "figures" / "f10_uc4_best_classifier_confusion_matrix.png"

BASE_FEATURES = [
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "pressure_pa",
    "rain_1h_mm",
    "humidex_c",
]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    timestamps = pd.to_datetime(out["timestamp_utc"], errors="coerce", utc=True)
    out["hour_sin"] = np.sin(2 * np.pi * timestamps.dt.hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * timestamps.dt.hour / 24)
    out["month_sin"] = np.sin(2 * np.pi * timestamps.dt.month / 12)
    out["month_cos"] = np.cos(2 * np.pi * timestamps.dt.month / 12)
    return out


def load_risk_data(path: Path, fallback_meteo: Path) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path, low_memory=False)
    else:
        df = pd.read_csv(fallback_meteo, low_memory=False)
        df = add_risk_labels(df)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["timestamp_utc", "risk_level"]).sort_values("timestamp_utc")
    return add_calendar_features(df)


def chronological_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n_rows = len(df)
    train_end = int(n_rows * train_size)
    validation_end = int(n_rows * (train_size + validation_size))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:validation_end].copy(),
        df.iloc[validation_end:].copy(),
    )


def cap_training_rows(
    train_df: pd.DataFrame,
    max_train_rows: int,
    random_state: int,
) -> pd.DataFrame:
    """Limit training size while keeping all rare high-risk rows."""

    if len(train_df) <= max_train_rows:
        return train_df

    protected = train_df[train_df["risk_level"].isin(["high_risk", "dangerous"])]
    remaining_quota = max_train_rows - len(protected)
    if remaining_quota <= 0:
        return protected.sample(n=max_train_rows, random_state=random_state)

    common = train_df[~train_df.index.isin(protected.index)]
    common_sample = common.sample(n=min(remaining_quota, len(common)), random_state=random_state)
    return pd.concat([protected, common_sample]).sort_values("timestamp_utc")


def build_xgboost_classifier(random_state: int, n_classes: int) -> Pipeline | None:
    try:
        from xgboost import XGBClassifier
    except Exception as exc:
        message = next((line.strip() for line in str(exc).splitlines() if line.strip()), repr(exc))
        print(f"Skipping XGBoost classifier because it could not be imported: {message}")
        return None

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=n_classes,
                    eval_metric="mlogloss",
                    n_estimators=180,
                    learning_rate=0.05,
                    max_depth=4,
                    min_child_weight=2,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=random_state,
                    n_jobs=1,
                    tree_method="hist",
                ),
            ),
        ]
    )


def build_classifiers(random_state: int, n_classes: int) -> tuple[dict[str, Pipeline], list[str]]:
    models = {
        "most_frequent_baseline": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", DummyClassifier(strategy="most_frequent")),
            ]
        ),
        "logistic_regression_balanced": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=600,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
        "random_forest_classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        min_samples_leaf=4,
                        max_depth=18,
                        class_weight="balanced_subsample",
                        random_state=random_state,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting_classifier": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_iter=180,
                        learning_rate=0.06,
                        l2_regularization=0.01,
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }
    skipped_models = []
    xgboost_model = build_xgboost_classifier(random_state, n_classes)
    if xgboost_model is None:
        skipped_models.append("xgboost_classifier_missing_optional_dependency")
    else:
        models["xgboost_classifier"] = xgboost_model
    return models, skipped_models


def evaluate_classifier(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
) -> dict:
    labels = list(range(len(class_names)))
    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }


def encode_risk_levels(values: pd.Series) -> np.ndarray:
    mapping = {level: idx for idx, level in enumerate(RISK_LEVEL_ORDER)}
    return values.map(mapping).to_numpy(dtype=int)


def decode_risk_levels(values: np.ndarray) -> list[str]:
    return [RISK_LEVEL_ORDER[int(value)] for value in values]


def save_comparison_figure(results: list[dict], path: Path) -> None:
    df = pd.DataFrame(results).sort_values("macro_f1", ascending=True)
    ax = df.plot.barh(x="model", y="macro_f1", legend=False, figsize=(9, 4.5), color="#2f6f7e")
    ax.set_title("F10-UC4 risk classifiers: macro F1")
    ax.set_xlabel("Macro F1, higher is better")
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def save_confusion_matrix_figure(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    path: Path,
) -> None:
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    ax.set_title("F10-UC4 best classifier confusion matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_names)), class_names, rotation=25, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                str(matrix[row_idx, col_idx]),
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train F10 risk classification models.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--fallback-meteo", type=Path, default=DEFAULT_FALLBACK_METEO)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--prediction-examples", type=Path, default=DEFAULT_PREDICTION_EXAMPLES)
    parser.add_argument("--comparison-figure", type=Path, default=DEFAULT_COMPARISON_FIGURE)
    parser.add_argument("--confusion-figure", type=Path, default=DEFAULT_CONFUSION_FIGURE)
    parser.add_argument("--max-train-rows", type=int, default=160_000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--no-humidex",
        action="store_true",
        help="Ablation: remove humidex_c from features to test non-circular classification.",
    )
    args = parser.parse_args()

    df = load_risk_data(args.input, args.fallback_meteo)
    train_df, validation_df, test_df = chronological_split(df)
    train_df_used = cap_training_rows(train_df, args.max_train_rows, args.random_state)

    candidate_features = BASE_FEATURES + ["hour_sin", "hour_cos", "month_sin", "month_cos"]
    if args.no_humidex:
        candidate_features = [c for c in candidate_features if c != "humidex_c"]
    feature_columns = [
        column
        for column in candidate_features
        if column in df.columns and train_df_used[column].notna().any()
    ]

    x_train = train_df_used[feature_columns]
    y_train = encode_risk_levels(train_df_used["risk_level"])
    x_validation = validation_df[feature_columns]
    y_validation = encode_risk_levels(validation_df["risk_level"])
    x_test = test_df[feature_columns]
    y_test = encode_risk_levels(test_df["risk_level"])

    models, skipped_models = build_classifiers(args.random_state, len(RISK_LEVEL_ORDER))
    results = []
    predictions_by_model = {}
    for model_name, model in models.items():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            model.fit(x_train, y_train)
            validation_pred = model.predict(x_validation)
            test_pred = model.predict(x_test)
        validation_metrics = evaluate_classifier(
            f"{model_name}_validation",
            y_validation,
            validation_pred,
            RISK_LEVEL_ORDER,
        )
        test_metrics = evaluate_classifier(model_name, y_test, test_pred, RISK_LEVEL_ORDER)
        test_metrics["validation_macro_f1"] = validation_metrics["macro_f1"]
        test_metrics["validation_balanced_accuracy"] = validation_metrics["balanced_accuracy"]
        results.append(test_metrics)
        predictions_by_model[model_name] = test_pred

    best_result = max(results, key=lambda item: item["macro_f1"])
    best_model_name = best_result["model"]
    best_predictions = predictions_by_model[best_model_name]

    prediction_table = pd.DataFrame(
        {
            "timestamp_utc": test_df["timestamp_utc"].astype(str).to_numpy(),
            "station_id": test_df.get("station_id", pd.Series([None] * len(test_df))).to_numpy(),
            "actual_risk_level": decode_risk_levels(y_test),
            "predicted_risk_level": decode_risk_levels(best_predictions),
            "model": best_model_name,
        }
    )

    summary = {
        "task": "F10-UC4 ML-based risk classification",
        "humidex_ablation": args.no_humidex,
        "input_file": str(args.input if args.input.exists() else args.fallback_meteo),
        "prediction_file": str(args.predictions),
        "prediction_examples_file": str(args.prediction_examples),
        "target": "risk_level",
        "target_source": "Rule-derived humidex risk labels from F10-UC1.",
        "feature_columns": feature_columns,
        "class_order": RISK_LEVEL_ORDER,
        "class_counts": {
            key: int(value)
            for key, value in df["risk_level"].value_counts().reindex(RISK_LEVEL_ORDER, fill_value=0).items()
        },
        "split": {
            "method": "chronological",
            "train_rows_original": int(len(train_df)),
            "train_rows_used": int(len(train_df_used)),
            "validation_rows": int(len(validation_df)),
            "test_rows": int(len(test_df)),
        },
        "best_model_by_macro_f1": best_model_name,
        "results": results,
        "skipped_models": skipped_models,
        "scope_note": "This is classification of rule-derived risk labels, not independent real-world incident labels.",
    }

    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.predictions.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.to_csv(args.predictions, index=False)
    args.prediction_examples.parent.mkdir(parents=True, exist_ok=True)
    prediction_table.head(200).to_csv(args.prediction_examples, index=False)
    save_comparison_figure(results, args.comparison_figure)
    save_confusion_matrix_figure(
        y_test,
        best_predictions,
        RISK_LEVEL_ORDER,
        args.confusion_figure,
    )

    print(f"Rows with risk labels: {len(df):,}")
    print(f"Training rows used: {len(train_df_used):,}")
    print(f"Best model by macro F1: {best_model_name}")
    for result in sorted(results, key=lambda item: item["macro_f1"], reverse=True):
        print(
            f"{result['model']}: "
            f"macro F1={result['macro_f1']:.4f}, "
            f"balanced accuracy={result['balanced_accuracy']:.4f}, "
            f"accuracy={result['accuracy']:.4f}"
        )
    print(f"Wrote {args.results}")
    print(f"Wrote {args.predictions}")
    print(f"Wrote {args.prediction_examples}")
    print(f"Wrote {args.comparison_figure}")
    print(f"Wrote {args.confusion_figure}")


if __name__ == "__main__":
    main()
