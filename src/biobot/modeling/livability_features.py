"""Feature preparation for F9 livability score prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd


CORE_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "humidex_c",
    "record_count",
]


def add_time_features(df: pd.DataFrame, timestamp_column: str = "timestamp_utc") -> pd.DataFrame:
    """Add cyclic calendar features from the UTC timestamp."""

    out = df.copy()
    timestamps = pd.to_datetime(out[timestamp_column], errors="coerce", utc=True)
    out["hour"] = timestamps.dt.hour
    out["dayofweek"] = timestamps.dt.dayofweek
    out["month"] = timestamps.dt.month

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dayofweek_sin"] = np.sin(2 * np.pi * out["dayofweek"] / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * out["dayofweek"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * out["month"] / 12)
    out["month_cos"] = np.cos(2 * np.pi * out["month"] / 12)
    return out


def add_lag_and_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    lags: tuple[int, ...] = (1, 4, 16),
    windows: tuple[int, ...] = (4, 16),
) -> pd.DataFrame:
    """Add past-only lag and rolling features for time-aware tabular models."""

    out = df.copy()
    for column in columns:
        if column not in out.columns:
            continue
        for lag in lags:
            out[f"{column}_lag_{lag}"] = out[column].shift(lag)
        for window in windows:
            out[f"{column}_rolling_mean_{window}"] = (
                out[column].shift(1).rolling(window=window, min_periods=2).mean()
            )
            out[f"{column}_rolling_std_{window}"] = (
                out[column].shift(1).rolling(window=window, min_periods=2).std()
            )
    return out


def prepare_neusta_livability_table(
    path: str,
    target_column: str = "vivabilite_binary_mean",
) -> tuple[pd.DataFrame, list[str], str]:
    """Load Neusta processed data and create model-ready tabular features."""

    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    df = df.sort_values("timestamp_utc").reset_index(drop=True)

    df = add_time_features(df)
    df = add_lag_and_rolling_features(
        df,
        columns=["temperature_c", "relative_humidity_pct", "humidex_c"],
    )

    feature_columns = [
        column
        for column in df.columns
        if column
        not in {
            "timestamp_utc",
            target_column,
            "vivabilite_binary_mode",
        }
        and not column.endswith("_was_imputed")
        and (
            column in CORE_FEATURES
            or column.endswith("_norm")
            or "_lag_" in column
            or "_rolling_" in column
            or column
            in {
                "hour_sin",
                "hour_cos",
                "dayofweek_sin",
                "dayofweek_cos",
                "month_sin",
                "month_cos",
            }
        )
    ]

    df = df.dropna(subset=["timestamp_utc", target_column]).copy()
    return df, feature_columns, target_column


def chronological_split(
    df: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data chronologically into train, validation, and test tables."""

    n_rows = len(df)
    train_end = int(n_rows * train_size)
    validation_end = int(n_rows * (train_size + validation_size))
    return (
        df.iloc[:train_end].copy(),
        df.iloc[train_end:validation_end].copy(),
        df.iloc[validation_end:].copy(),
    )


def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    min_train_fraction: float = 0.40,
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Expanding-window train/test pairs for time-series cross-validation.

    Each pair uses all data up to a cutpoint as train and the next
    step-sized slice as test. More reliable than a single split on small datasets.
    """
    n = len(df)
    min_train = int(n * min_train_fraction)
    step = max(1, (n - min_train) // n_splits)
    splits = []
    for i in range(n_splits):
        train_end = min_train + i * step
        test_end = min(train_end + step, n)
        if train_end >= n or test_end > n:
            break
        splits.append((df.iloc[:train_end].copy(), df.iloc[train_end:test_end].copy()))
    return splits


def make_sequence_arrays(
    df: pd.DataFrame,
    feature_columns: list[str],
    target_column: str,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray, pd.Series]:
    """Create sliding-window arrays for LSTM/CNN-LSTM experiments."""

    work = df.sort_values("timestamp_utc").reset_index(drop=True)
    feature_values = work[feature_columns].to_numpy(dtype="float32")
    target_values = work[target_column].to_numpy(dtype="float32")

    x_values = []
    y_values = []
    timestamps = []
    for end_idx in range(window_size, len(work)):
        x_values.append(feature_values[end_idx - window_size : end_idx])
        y_values.append(target_values[end_idx])
        timestamps.append(work.loc[end_idx, "timestamp_utc"])

    return np.asarray(x_values), np.asarray(y_values), pd.Series(timestamps)

