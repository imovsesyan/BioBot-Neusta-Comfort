"""F9-UC8: Optional advanced sequence model for livability score prediction.

This script trains a small LSTM or CNN-LSTM on Neusta `vivabilite_binary_mean`.
It is intentionally optional because advanced models should be compared only
after F9-UC7 baselines are working.

Run from the repository root:
    python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.modeling.livability_features import (  # noqa: E402
    chronological_split,
    make_sequence_arrays,
    prepare_neusta_livability_table,
)
from biobot.modeling.metrics import regression_metrics  # noqa: E402


DEFAULT_INPUT = ROOT / "data" / "processed" / "neusta_15min_clean.csv"
DEFAULT_RESULTS = ROOT / "reports" / "tables" / "f9_uc8_sequence_model_results.json"


def import_tensorflow():
    """Import TensorFlow with a helpful error when it is unavailable."""

    try:
        import tensorflow as tf
    except ImportError as exc:
        raise SystemExit(
            "TensorFlow is required for F9-UC8 advanced sequence models. "
            "Install optional dependencies with: pip install -r requirements-advanced.txt"
        ) from exc
    return tf


def build_lstm_model(tf, window_size: int, n_features: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, n_features)),
            tf.keras.layers.LSTM(64, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def build_cnn_lstm_model(tf, window_size: int, n_features: int):
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(window_size, n_features)),
            tf.keras.layers.Conv1D(filters=32, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.Conv1D(filters=16, kernel_size=3, padding="causal", activation="relu"),
            tf.keras.layers.LSTM(48, return_sequences=False),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return model


def preprocess_features(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Fit imputation/scaling on train only, then transform all splits."""

    feature_columns = [column for column in feature_columns if train_df[column].notna().any()]
    imputer = SimpleImputer(strategy="median")
    scaler = MinMaxScaler()

    train_features = imputer.fit_transform(train_df[feature_columns])
    train_features = scaler.fit_transform(train_features)
    validation_features = scaler.transform(imputer.transform(validation_df[feature_columns]))
    test_features = scaler.transform(imputer.transform(test_df[feature_columns]))

    def rebuild(original: pd.DataFrame, transformed: np.ndarray) -> pd.DataFrame:
        out = original[["timestamp_utc", "vivabilite_binary_mean"]].copy()
        for idx, column in enumerate(feature_columns):
            out[column] = transformed[:, idx]
        return out

    return (
        rebuild(train_df, train_features),
        rebuild(validation_df, validation_features),
        rebuild(test_df, test_features),
        feature_columns,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train optional F9 sequence model.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--model", choices=["lstm", "cnn_lstm"], default="cnn_lstm")
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(args.random_state)

    df, feature_columns, target_column = prepare_neusta_livability_table(str(args.input))
    train_df, validation_df, test_df = chronological_split(df)
    train_df, validation_df, test_df, feature_columns = preprocess_features(
        train_df,
        validation_df,
        test_df,
        feature_columns,
    )

    x_train, y_train, _ = make_sequence_arrays(
        train_df,
        feature_columns,
        target_column,
        args.window_size,
    )
    x_validation, y_validation, _ = make_sequence_arrays(
        validation_df,
        feature_columns,
        target_column,
        args.window_size,
    )
    x_test, y_test, test_timestamps = make_sequence_arrays(
        test_df,
        feature_columns,
        target_column,
        args.window_size,
    )

    builder = build_cnn_lstm_model if args.model == "cnn_lstm" else build_lstm_model
    model = builder(tf, args.window_size, len(feature_columns))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=4,
            restore_best_weights=True,
        )
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_validation, y_validation),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=0,
        shuffle=False,
    )

    test_predictions = model.predict(x_test, verbose=0).reshape(-1)
    test_predictions = np.clip(test_predictions, 0.0, 1.0)
    metrics = regression_metrics(y_test, test_predictions)

    summary = {
        "task": "F9-UC8 advanced sequence livability prediction",
        "model": args.model,
        "target": target_column,
        "window_size": args.window_size,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history.history["loss"]),
        "n_features": len(feature_columns),
        "train_sequences": int(len(x_train)),
        "validation_sequences": int(len(x_validation)),
        "test_sequences": int(len(x_test)),
        "test_metrics": metrics,
        "scope_note": "Recommendation systems and model interpretation are intentionally out of scope.",
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    predictions_path = args.results.with_name("f9_uc8_sequence_test_predictions.csv")
    pd.DataFrame(
        {
            "timestamp_utc": test_timestamps.astype(str),
            "actual": y_test,
            "predicted": test_predictions,
            "model": args.model,
        }
    ).to_csv(predictions_path, index=False)

    print(f"Model: {args.model}")
    print(
        f"Test MAE={metrics['mae']:.4f}, "
        f"RMSE={metrics['rmse']:.4f}, "
        f"R2={metrics['r2']:.4f}"
    )
    print(f"Wrote {args.results}")
    print(f"Wrote {predictions_path}")


if __name__ == "__main__":
    main()

