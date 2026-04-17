import pandas as pd
import pytest

from biobot.modeling.livability_features import (
    add_lag_and_rolling_features,
    chronological_split,
    make_sequence_arrays,
)


def test_chronological_split_keeps_order():
    df = pd.DataFrame({"value": range(10)})

    train, validation, test = chronological_split(df, train_size=0.6, validation_size=0.2)

    assert train["value"].tolist() == [0, 1, 2, 3, 4, 5]
    assert validation["value"].tolist() == [6, 7]
    assert test["value"].tolist() == [8, 9]


def test_lag_and_rolling_features_use_past_values_only():
    df = pd.DataFrame({"temperature_c": [10.0, 20.0, 30.0, 40.0]})

    result = add_lag_and_rolling_features(
        df,
        columns=["temperature_c"],
        lags=(1,),
        windows=(3,),
    )

    assert pd.isna(result.loc[0, "temperature_c_lag_1"])
    assert result.loc[1, "temperature_c_lag_1"] == 10.0
    assert result.loc[3, "temperature_c_rolling_mean_3"] == 20.0


def test_make_sequence_arrays_uses_previous_window_to_predict_current_target():
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=4, freq="15min", tz="UTC"),
            "feature": [1.0, 2.0, 3.0, 4.0],
            "target": [0.1, 0.2, 0.3, 0.4],
        }
    )

    x_values, y_values, timestamps = make_sequence_arrays(
        df,
        feature_columns=["feature"],
        target_column="target",
        window_size=2,
    )

    assert x_values.shape == (2, 2, 1)
    assert x_values[0, :, 0].tolist() == [1.0, 2.0]
    assert y_values.tolist() == pytest.approx([0.3, 0.4])
    assert timestamps.astype(str).tolist()[0].startswith("2026-01-01 00:30:00")
