"""Clean, impute, normalize, and aggregate BioBot standardized CSV files.

F8-UC4 keeps each data source separate. This avoids pretending that sensors,
weather stations, and soil probes are automatically colocated before that
relationship has been justified.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_FREQ = "15min"


RANGES: dict[str, tuple[float, float]] = {
    "temperature_c": (-20.0, 60.0),
    "temperature_secondary_c": (-20.0, 60.0),
    "relative_humidity_pct": (0.0, 100.0),
    "humidity_secondary_pct": (0.0, 100.0),
    "co2_ppm": (250.0, 5000.0),
    "tvoc_ppb": (0.0, 10000.0),
    "pm1_ugm3": (0.0, 1000.0),
    "pm25_ugm3": (0.0, 1000.0),
    "pm10_ugm3": (0.0, 1000.0),
    "sound_level_db": (20.0, 120.0),
    "soil_moisture_pct": (0.0, 100.0),
    "humidex_c": (-20.0, 80.0),
    "battery_level_pct": (0.0, 100.0),
    "wind_speed_mps": (0.0, 75.0),
    "wind_direction_deg": (0.0, 360.0),
    "pressure_pa": (80000.0, 110000.0),
    "sea_level_pressure_pa": (80000.0, 110000.0),
    "rain_1h_mm": (0.0, 300.0),
    "rain_3h_mm": (0.0, 500.0),
    "rain_6h_mm": (0.0, 700.0),
    "dew_point_c": (-40.0, 40.0),
    "vivabilite_binary": (0.0, 1.0),
    "vivabilite_score_meteo": (0.0, 7.0),
}


IOT_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "co2_ppm",
    "tvoc_ppb",
    "pm1_ugm3",
    "pm25_ugm3",
    "pm10_ugm3",
    "sound_level_db",
]

AQUACHECK_FEATURES = [
    "soil_moisture_pct",
    "temperature_c",
    "relative_humidity_pct",
    "humidex_c",
    "battery_level_pct",
]

NEUSTA_FEATURES = [
    "temperature_c",
    "relative_humidity_pct",
    "temperature_secondary_c",
    "humidity_secondary_pct",
    "pm1_ugm3",
    "pm25_ugm3",
    "pm10_ugm3",
    "humidex_c",
]

METEO_FEATURES = [
    "temperature_c",
    "dew_point_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "wind_direction_deg",
    "pressure_pa",
    "sea_level_pressure_pa",
    "rain_1h_mm",
    "rain_3h_mm",
    "rain_6h_mm",
    "humidex_c",
]


def read_standardized_csv(path: Path) -> pd.DataFrame:
    """Read a standardized CSV and parse the canonical UTC timestamp."""

    df = pd.read_csv(path, low_memory=False)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], errors="coerce", utc=True)
    return df


def numeric_columns_present(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return numeric candidate columns that are present in ``df``."""

    return [column for column in candidates if column in df.columns]


def coerce_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Coerce selected columns to numeric values."""

    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def normalize_iot_sensor_id(value: Any) -> str:
    """Normalize IoT sensor IDs while keeping missing/malformed IDs explicit."""

    if value is None or pd.isna(value):
        return "unknown_sensor"
    text = str(value).strip()
    if not text:
        return "unknown_sensor"

    digits = re.sub(r"\D", "", text)
    if len(digits) == 14:
        return digits

    return "unknown_sensor"


def apply_range_rules(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Replace physically impossible values with nulls and count them."""

    out = df.copy()
    summary: dict[str, Any] = {}
    for column in columns:
        if column not in RANGES:
            continue
        low, high = RANGES[column]
        non_null_before = int(out[column].notna().sum())
        bad_mask = out[column].notna() & ~out[column].between(low, high)
        bad_count = int(bad_mask.sum())
        out.loc[bad_mask, column] = pd.NA
        summary[column] = {
            "allowed_min": low,
            "allowed_max": high,
            "non_null_before": non_null_before,
            "out_of_range_replaced_with_null": bad_count,
        }
    return out, summary


def aggregate_existing_intervals(
    df: pd.DataFrame,
    group_columns: list[str],
    numeric_columns: list[str],
    freq: str,
) -> pd.DataFrame:
    """Aggregate observations into time intervals without creating new rows."""

    work = df.dropna(subset=["timestamp_utc"]).copy()
    work["timestamp_utc"] = work["timestamp_utc"].dt.floor(freq)
    work["record_count"] = 1
    aggregations = {column: "mean" for column in numeric_columns}
    aggregations["record_count"] = "sum"

    keys = group_columns + ["timestamp_utc"]
    grouped = work.groupby(keys, dropna=False).agg(aggregations).reset_index()
    return grouped.sort_values(keys).reset_index(drop=True)


def impute_short_gaps(
    df: pd.DataFrame,
    group_columns: list[str],
    numeric_columns: list[str],
    max_gap_steps: int,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Interpolate short gaps inside each group and report imputation counts."""

    if not numeric_columns:
        return df, {}

    out = df.sort_values(group_columns + ["timestamp_utc"] if group_columns else ["timestamp_utc"])
    impute_counts = {column: 0 for column in numeric_columns}

    def impute_group(group: pd.DataFrame) -> pd.DataFrame:
        group = group.sort_values("timestamp_utc").copy()
        original_missing = group[numeric_columns].isna()
        indexed = group.set_index("timestamp_utc")
        indexed[numeric_columns] = indexed[numeric_columns].interpolate(
            method="time",
            limit=max_gap_steps,
            limit_direction="both",
        )
        indexed[numeric_columns] = indexed[numeric_columns].ffill(limit=max_gap_steps)
        indexed[numeric_columns] = indexed[numeric_columns].bfill(limit=max_gap_steps)
        group = indexed.reset_index()
        for column in numeric_columns:
            flag = f"{column}_was_imputed"
            group[flag] = original_missing[column].to_numpy() & group[column].notna().to_numpy()
        return group

    if group_columns:
        parts = [
            impute_group(group)
            for _, group in out.groupby(group_columns, dropna=False, sort=False)
        ]
        out = pd.concat(parts, ignore_index=True) if parts else out
    else:
        out = impute_group(out)

    for column in numeric_columns:
        flag = f"{column}_was_imputed"
        if flag in out.columns:
            impute_counts[column] = int(out[flag].sum())

    sort_columns = group_columns + ["timestamp_utc"] if group_columns else ["timestamp_utc"]
    return out.sort_values(sort_columns).reset_index(drop=True), impute_counts


def add_minmax_normalized_columns(
    df: pd.DataFrame,
    numeric_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Add min-max normalized columns for later baseline ML experiments."""

    out = df.copy()
    stats: dict[str, Any] = {}
    for column in numeric_columns:
        non_null = out[column].dropna()
        norm_column = f"{column}_norm"
        if non_null.empty:
            out[norm_column] = pd.NA
            stats[column] = {"min": None, "max": None, "normalized_column": norm_column}
            continue

        minimum = float(non_null.min())
        maximum = float(non_null.max())
        if minimum == maximum:
            out[norm_column] = 0.0
        else:
            out[norm_column] = (out[column] - minimum) / (maximum - minimum)
        stats[column] = {
            "min": minimum,
            "max": maximum,
            "normalized_column": norm_column,
        }
    return out, stats


def compact_profile(
    source: str,
    input_rows: int,
    output: pd.DataFrame,
    range_summary: dict[str, Any],
    impute_counts: dict[str, int],
    normalization_stats: dict[str, Any],
) -> dict[str, Any]:
    """Create a reproducible summary for one UC4 output."""

    timestamps = pd.to_datetime(output["timestamp_utc"], errors="coerce", utc=True)
    return {
        "source": source,
        "input_rows": int(input_rows),
        "output_rows": int(len(output)),
        "output_columns": int(len(output.columns)),
        "date_min_utc": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "date_max_utc": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "range_rules": range_summary,
        "imputed_values": impute_counts,
        "normalization": normalization_stats,
    }


def process_iot(interim_dir: Path, output_dir: Path, freq: str) -> dict[str, Any]:
    path = interim_dir / "iot_observations_standardized.csv"
    df = read_standardized_csv(path)
    input_rows = len(df)
    original_sensor_count = int(df["sensor_id"].nunique(dropna=True))
    df["sensor_id"] = df["sensor_id"].map(normalize_iot_sensor_id)
    normalized_sensor_count = int(df["sensor_id"].nunique(dropna=True))
    features = numeric_columns_present(df, IOT_FEATURES)
    df = coerce_numeric(df, features)
    df, range_summary = apply_range_rules(df, features)
    output = aggregate_existing_intervals(df, ["sensor_id"], features, freq)
    output, impute_counts = impute_short_gaps(output, ["sensor_id"], features, max_gap_steps=4)
    output, norm_stats = add_minmax_normalized_columns(output, features)
    output_path = output_dir / "iot_15min_clean.csv"
    output.to_csv(output_path, index=False)
    profile = compact_profile("iot", input_rows, output, range_summary, impute_counts, norm_stats)
    profile["output_file"] = str(output_path)
    profile["aggregation_frequency"] = freq
    profile["sensor_id_normalization"] = {
        "raw_non_null_sensor_ids": original_sensor_count,
        "normalized_sensor_ids": normalized_sensor_count,
        "rule": "Keep only IoT IDs that can be normalized to 14 digits; otherwise use unknown_sensor.",
    }
    return profile


def process_aquacheck(interim_dir: Path, output_dir: Path, freq: str) -> dict[str, Any]:
    path = interim_dir / "aquacheck_observations_standardized.csv"
    df = read_standardized_csv(path)
    input_rows = len(df)
    df["sensor_id"] = df["sensor_id"].fillna("unknown_sensor").replace("", "unknown_sensor")
    features = numeric_columns_present(df, AQUACHECK_FEATURES)
    df = coerce_numeric(df, features)
    df, range_summary = apply_range_rules(df, features)
    output = aggregate_existing_intervals(df, ["sensor_id"], features, freq)
    output, impute_counts = impute_short_gaps(output, ["sensor_id"], features, max_gap_steps=4)
    output, norm_stats = add_minmax_normalized_columns(output, features)
    output_path = output_dir / "aquacheck_15min_clean.csv"
    output.to_csv(output_path, index=False)
    profile = compact_profile("aquacheck", input_rows, output, range_summary, impute_counts, norm_stats)
    profile["output_file"] = str(output_path)
    profile["aggregation_frequency"] = freq
    return profile


def process_neusta(interim_dir: Path, output_dir: Path, freq: str) -> dict[str, Any]:
    path = interim_dir / "neusta_observations_standardized.csv"
    df = read_standardized_csv(path)
    input_rows = len(df)
    features = numeric_columns_present(df, NEUSTA_FEATURES)
    target_columns = numeric_columns_present(df, ["vivabilite_binary"])
    all_numeric = features + target_columns
    df = coerce_numeric(df, all_numeric)
    df, range_summary = apply_range_rules(df, all_numeric)
    output = aggregate_existing_intervals(df, [], all_numeric, freq)
    if "vivabilite_binary" in output.columns:
        output["vivabilite_binary_mode"] = output["vivabilite_binary"].round()
        output = output.rename(columns={"vivabilite_binary": "vivabilite_binary_mean"})
    output, impute_counts = impute_short_gaps(output, [], features, max_gap_steps=4)
    output, norm_stats = add_minmax_normalized_columns(output, features)
    output_path = output_dir / "neusta_15min_clean.csv"
    output.to_csv(output_path, index=False)
    profile = compact_profile("neusta", input_rows, output, range_summary, impute_counts, norm_stats)
    profile["output_file"] = str(output_path)
    profile["aggregation_frequency"] = freq
    profile["target_note"] = "vivabilite_binary_mean is the proportion of raw rows equal to 1 inside the interval; vivabilite_binary_mode is the rounded interval label."
    return profile


def process_meteo(interim_dir: Path, output_dir: Path) -> dict[str, Any]:
    path = interim_dir / "meteo_france_observations_standardized.csv"
    df = read_standardized_csv(path)
    input_rows = len(df)
    df["station_id"] = df["station_wmo"].fillna(df["station_wigos"]).astype(str)
    features = numeric_columns_present(df, METEO_FEATURES)
    target_columns = numeric_columns_present(df, ["vivabilite_score_meteo"])
    all_numeric = features + target_columns + numeric_columns_present(df, ["latitude", "longitude"])
    df = coerce_numeric(df, all_numeric)
    df, range_summary = apply_range_rules(df, features + target_columns)

    output = aggregate_existing_intervals(
        df,
        ["station_id", "station_name"],
        all_numeric,
        "1h",
    )
    output, impute_counts = impute_short_gaps(
        output,
        ["station_id", "station_name"],
        features,
        max_gap_steps=2,
    )
    output, norm_stats = add_minmax_normalized_columns(output, features)
    output_path = output_dir / "meteo_france_1h_clean.csv"
    output.to_csv(output_path, index=False)
    profile = compact_profile("meteo_france", input_rows, output, range_summary, impute_counts, norm_stats)
    profile["output_file"] = str(output_path)
    profile["aggregation_frequency"] = "1h"
    profile["target_note"] = "Meteo France vivabilite_score_meteo remains a 0..7 score and is not the same target as Neusta binary livability."
    return profile


def run_uc4_pipeline(interim_dir: Path, output_dir: Path, freq: str = DEFAULT_FREQ) -> dict[str, Any]:
    """Run the complete F8-UC4 source-specific pipeline."""

    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "iot": process_iot(interim_dir, output_dir, freq),
        "aquacheck": process_aquacheck(interim_dir, output_dir, freq),
        "neusta": process_neusta(interim_dir, output_dir, freq),
        "meteo_france": process_meteo(interim_dir, output_dir),
    }
    return summary


def write_summary(summary: dict[str, Any], path: Path) -> None:
    """Write the UC4 JSON summary."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
