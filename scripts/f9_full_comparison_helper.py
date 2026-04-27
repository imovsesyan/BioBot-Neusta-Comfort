"""Helper script for the F9 full model comparison study.

Computes train/val/test MAE/RMSE/R2 for:
  - All F9-UC7 classical models (mean, ridge, RF, HGB, XGBoost) WITH humidex_c
  - The CNN-LSTM (F9-UC8 architecture) WITH and WITHOUT humidex_c

Outputs a single JSON to reports/tables/f9_full_model_comparison_metrics.json.

This is read-only with respect to the existing reports — it writes to a new file.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.modeling.livability_features import (  # noqa: E402
    chronological_split,
    make_sequence_arrays,
    prepare_neusta_livability_table,
)
from biobot.modeling.metrics import regression_metrics  # noqa: E402


INPUT = ROOT / "data" / "processed" / "neusta_15min_clean.csv.gz"
OUTPUT = ROOT / "reports" / "tables" / "f9_full_model_comparison_metrics.json"


def _clip(arr):
    return np.clip(np.asarray(arr).reshape(-1), 0.0, 1.0)


def _metrics(y_true, y_pred):
    m = regression_metrics(np.asarray(y_true), np.asarray(y_pred))
    return {"mae": float(m["mae"]), "rmse": float(m["rmse"]), "r2": float(m["r2"])}


def build_classical_models(random_state: int):
    models = {
        "mean_baseline": Pipeline(
            [("imputer", SimpleImputer(strategy="median")),
             ("model", DummyRegressor(strategy="mean"))]
        ),
        "ridge_regression": Pipeline(
            [("imputer", SimpleImputer(strategy="median")),
             ("scaler", StandardScaler()),
             ("model", TransformedTargetRegressor(
                 regressor=Ridge(alpha=1.0), transformer=StandardScaler()))]
        ),
        "random_forest": Pipeline(
            [("imputer", SimpleImputer(strategy="median")),
             ("model", RandomForestRegressor(
                 n_estimators=300, min_samples_leaf=4,
                 random_state=random_state, n_jobs=1))]
        ),
        "hist_gradient_boosting": Pipeline(
            [("imputer", SimpleImputer(strategy="median")),
             ("model", HistGradientBoostingRegressor(
                 max_iter=300, learning_rate=0.04, l2_regularization=0.05,
                 random_state=random_state))]
        ),
    }
    try:
        from xgboost import XGBRegressor
        models["xgboost"] = Pipeline(
            [("imputer", SimpleImputer(strategy="median")),
             ("model", XGBRegressor(
                 objective="reg:squarederror", eval_metric="rmse",
                 n_estimators=400, learning_rate=0.03, max_depth=3,
                 min_child_weight=2, subsample=0.9, colsample_bytree=0.9,
                 reg_lambda=1.0, random_state=random_state, n_jobs=1,
                 tree_method="hist"))]
        )
    except Exception as e:
        print(f"XGBoost not available: {e}")
    return models


def run_classical(df, feature_columns, target_column, random_state: int = 42):
    train_df, val_df, test_df = chronological_split(df)
    feature_columns = [c for c in feature_columns if train_df[c].notna().any()]
    x_train = train_df[feature_columns]; y_train = train_df[target_column]
    x_val = val_df[feature_columns]; y_val = val_df[target_column]
    x_test = test_df[feature_columns]; y_test = test_df[target_column]

    out = {}
    for name, model in build_classical_models(random_state).items():
        model.fit(x_train, y_train)
        train_pred = _clip(model.predict(x_train))
        val_pred = _clip(model.predict(x_val))
        test_pred = _clip(model.predict(x_test))
        out[name] = {
            "train": _metrics(y_train, train_pred),
            "validation": _metrics(y_val, val_pred),
            "test": _metrics(y_test, test_pred),
            "n_train": int(len(y_train)),
            "n_validation": int(len(y_val)),
            "n_test": int(len(y_test)),
            "n_features": int(len(feature_columns)),
        }
        print(f"[CLASSICAL] {name}: train_mae={out[name]['train']['mae']:.4f} "
              f"val_mae={out[name]['validation']['mae']:.4f} "
              f"test_mae={out[name]['test']['mae']:.4f} "
              f"train_r2={out[name]['train']['r2']:.4f} "
              f"test_r2={out[name]['test']['r2']:.4f}")
    return out


def build_cnn_lstm_model(tf, window_size: int, n_features: int):
    return tf.keras.Sequential([
        tf.keras.layers.Input(shape=(window_size, n_features)),
        tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"),
        tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="causal", activation="relu"),
        tf.keras.layers.LSTM(48, return_sequences=False),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid"),
    ])


def run_cnn_lstm(df, feature_columns, target_column, no_humidex: bool,
                 epochs: int = 8, window_size: int = 16, random_state: int = 42):
    import tensorflow as tf
    tf.keras.utils.set_random_seed(random_state)

    if no_humidex:
        feature_columns = [
            c for c in feature_columns
            if c != "humidex_c" and c != "humidex_c_norm"
            and not c.startswith("humidex_c_lag_")
            and not c.startswith("humidex_c_rolling_")
        ]

    train_df, val_df, test_df = chronological_split(df)
    feature_columns = [c for c in feature_columns if train_df[c].notna().any()]

    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()

    train_features = scaler.fit_transform(imputer.fit_transform(train_df[feature_columns]))
    val_features = scaler.transform(imputer.transform(val_df[feature_columns]))
    test_features = scaler.transform(imputer.transform(test_df[feature_columns]))

    def rebuild(original, transformed):
        out = original[["timestamp_utc", target_column]].copy()
        for idx, col in enumerate(feature_columns):
            out[col] = transformed[:, idx]
        return out

    train_df_p = rebuild(train_df, train_features)
    val_df_p = rebuild(val_df, val_features)
    test_df_p = rebuild(test_df, test_features)

    x_train, y_train, _ = make_sequence_arrays(train_df_p, feature_columns, target_column, window_size)
    x_val, y_val, _ = make_sequence_arrays(val_df_p, feature_columns, target_column, window_size)
    x_test, y_test, _ = make_sequence_arrays(test_df_p, feature_columns, target_column, window_size)

    model = build_cnn_lstm_model(tf, window_size, len(feature_columns))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=4, restore_best_weights=True)]
    history = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs, batch_size=32,
        callbacks=callbacks, verbose=0, shuffle=False,
    )

    train_pred = _clip(model.predict(x_train, verbose=0))
    val_pred = _clip(model.predict(x_val, verbose=0))
    test_pred = _clip(model.predict(x_test, verbose=0))

    return {
        "no_humidex": bool(no_humidex),
        "epochs_completed": int(len(history.history["loss"])),
        "epochs_requested": int(epochs),
        "n_features": int(len(feature_columns)),
        "n_train_seq": int(len(x_train)),
        "n_val_seq": int(len(x_val)),
        "n_test_seq": int(len(x_test)),
        "train": _metrics(y_train, train_pred),
        "validation": _metrics(y_val, val_pred),
        "test": _metrics(y_test, test_pred),
    }


def main():
    df, feature_columns, target_column = prepare_neusta_livability_table(str(INPUT))
    train_df, val_df, test_df = chronological_split(df)

    classical = run_classical(df, feature_columns, target_column)

    print("\n[CNN-LSTM] WITH humidex_c — training...")
    cnn_with = run_cnn_lstm(df, feature_columns, target_column, no_humidex=False, epochs=8)
    print(f"  train_mae={cnn_with['train']['mae']:.4f} "
          f"val_mae={cnn_with['validation']['mae']:.4f} "
          f"test_mae={cnn_with['test']['mae']:.4f} "
          f"train_r2={cnn_with['train']['r2']:.4f} "
          f"test_r2={cnn_with['test']['r2']:.4f} "
          f"epochs={cnn_with['epochs_completed']}")

    print("\n[CNN-LSTM] WITHOUT humidex_c — training...")
    cnn_no = run_cnn_lstm(df, feature_columns, target_column, no_humidex=True, epochs=8)
    print(f"  train_mae={cnn_no['train']['mae']:.4f} "
          f"val_mae={cnn_no['validation']['mae']:.4f} "
          f"test_mae={cnn_no['test']['mae']:.4f} "
          f"train_r2={cnn_no['train']['r2']:.4f} "
          f"test_r2={cnn_no['test']['r2']:.4f} "
          f"epochs={cnn_no['epochs_completed']}")

    output = {
        "task": "F9 full model comparison — train/val/test for classical and CNN-LSTM",
        "input_file": str(INPUT),
        "target": target_column,
        "split": {
            "method": "chronological",
            "train_rows": int(len(train_df)),
            "validation_rows": int(len(val_df)),
            "test_rows": int(len(test_df)),
            "n_total_rows": int(len(df)),
        },
        "classical_with_humidex": classical,
        "cnn_lstm_with_humidex": cnn_with,
        "cnn_lstm_no_humidex": cnn_no,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(f"\nWrote {OUTPUT}")


if __name__ == "__main__":
    main()
