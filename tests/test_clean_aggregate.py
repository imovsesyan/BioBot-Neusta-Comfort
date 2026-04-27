import numpy as np
import pandas as pd
import pytest

from biobot.data.clean_aggregate import (
    add_minmax_normalized_columns,
    apply_range_rules,
    impute_short_gaps,
    normalize_iot_sensor_id,
)


def test_normalize_iot_sensor_id_keeps_14_digits():
    assert normalize_iot_sensor_id("20240313101500") == "20240313101500"


def test_normalize_iot_sensor_id_removes_date_separators():
    assert normalize_iot_sensor_id("2024-09-20 08:30:00") == "20240920083000"


def test_normalize_iot_sensor_id_rejects_incomplete_ids():
    assert normalize_iot_sensor_id("2024-09-20") == "unknown_sensor"
    assert normalize_iot_sensor_id("") == "unknown_sensor"


# ── FIX 2: apply_range_rules ─────────────────────────────────────────────────


def test_apply_range_rules_nulls_out_of_range_values():
    df = pd.DataFrame({"temperature_c": [-100.0, 20.0, 80.0]})
    result, summary = apply_range_rules(df, ["temperature_c"])
    assert result["temperature_c"].isna().sum() == 2
    assert result["temperature_c"].iloc[1] == pytest.approx(20.0)
    assert summary["temperature_c"]["out_of_range_replaced_with_null"] == 2


def test_apply_range_rules_keeps_all_in_range_values():
    df = pd.DataFrame({"relative_humidity_pct": [0.0, 50.0, 100.0]})
    result, summary = apply_range_rules(df, ["relative_humidity_pct"])
    assert result["relative_humidity_pct"].isna().sum() == 0
    assert summary["relative_humidity_pct"]["out_of_range_replaced_with_null"] == 0


# ── FIX 2: impute_short_gaps ─────────────────────────────────────────────────


def test_impute_short_gaps_fills_single_interior_gap():
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
            "temperature_c": [10.0, np.nan, 30.0],
        }
    )
    result, counts = impute_short_gaps(df, [], ["temperature_c"], max_gap_steps=4)
    assert result["temperature_c"].isna().sum() == 0
    assert result["temperature_c_was_imputed"].iloc[1]


def test_impute_short_gaps_large_gap_has_unfilled_interior():
    df = pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2024-01-01", periods=12, freq="15min", tz="UTC"),
            "temperature_c": [10.0] + [np.nan] * 10 + [50.0],
        }
    )
    result, counts = impute_short_gaps(df, [], ["temperature_c"], max_gap_steps=1)
    assert result["temperature_c"].isna().any()


# ── FIX 2: add_minmax_normalized_columns ─────────────────────────────────────


def test_add_minmax_normalized_columns_min_to_zero_max_to_one():
    df = pd.DataFrame({"temperature_c": [0.0, 25.0, 50.0]})
    result, stats = add_minmax_normalized_columns(df, ["temperature_c"])
    assert "temperature_c_norm" in result.columns
    assert result["temperature_c_norm"].min() == pytest.approx(0.0)
    assert result["temperature_c_norm"].max() == pytest.approx(1.0)
    assert stats["temperature_c"]["min"] == pytest.approx(0.0)
    assert stats["temperature_c"]["max"] == pytest.approx(50.0)


def test_add_minmax_normalized_columns_constant_column_maps_to_zero():
    df = pd.DataFrame({"temperature_c": [22.0, 22.0, 22.0]})
    result, stats = add_minmax_normalized_columns(df, ["temperature_c"])
    assert result["temperature_c_norm"].tolist() == pytest.approx([0.0, 0.0, 0.0])

