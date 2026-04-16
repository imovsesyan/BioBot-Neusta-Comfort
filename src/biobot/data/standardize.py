"""Standardize BioBot raw datasets into consistent CSV files.

The goal of F8-UC3 is not to clean the science yet. It is to make each raw
source readable in a predictable tabular format while keeping provenance fields
that explain where every row came from.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from zoneinfo import ZoneInfo

import pandas as pd


PARIS_TZ = ZoneInfo("Europe/Paris")
UTC = timezone.utc


@dataclass(frozen=True)
class ParsedTimestamp:
    """Timestamp values used in the standardized CSV files."""

    timestamp_utc: str | None
    timestamp_local: str | None
    timestamp_assumption: str


def first_nonspace(path: Path) -> str:
    """Return the first non-space character of a text file."""

    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        while True:
            char = handle.read(1)
            if char == "":
                return ""
            if not char.isspace():
                return char


def iter_json_records(path: Path) -> Iterable[dict[str, Any]]:
    """Yield records from either JSON arrays or line-delimited JSON files."""

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, list):
            for row in loaded:
                if isinstance(row, dict):
                    yield row
            return
        if isinstance(loaded, dict):
            yield loaded
            return
    except json.JSONDecodeError:
        pass

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            cleaned = line.strip().rstrip(",")
            if not cleaned or cleaned in {"[", "]"}:
                continue
            try:
                row = json.loads(cleaned)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                yield row


def to_float(value: Any) -> float | None:
    """Convert raw numeric strings to floats, preserving bad values as nulls."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned.lower() in {"", "nan", "none", "null", "na", "n/a"}:
            return None
        cleaned = cleaned.replace(",", ".")
    else:
        cleaned = value

    try:
        parsed = float(cleaned)
    except (TypeError, ValueError):
        return None
    if math.isnan(parsed) or math.isinf(parsed):
        return None
    return parsed


def parse_timestamp(value: Any) -> ParsedTimestamp:
    """Parse known timestamp formats and normalize them to UTC.

    Naive timestamps are treated as Europe/Paris local time because the project
    data is local operational data. The standardized file keeps this assumption
    explicit in ``timestamp_assumption``.
    """

    if value is None:
        return ParsedTimestamp(None, None, "missing")

    text = str(value).strip()
    if not text:
        return ParsedTimestamp(None, None, "missing")

    aware: datetime | None = None
    assumption = "explicit_timezone"

    for fmt in ("%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        try:
            naive = datetime.strptime(text, fmt)
        except ValueError:
            continue
        aware = naive.replace(tzinfo=PARIS_TZ)
        assumption = "localized_europe_paris"
        break

    if aware is None:
        for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"):
            try:
                parsed = datetime.strptime(text, fmt)
            except ValueError:
                continue
            aware = parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed
            break

    if aware is None:
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return ParsedTimestamp(None, None, "parse_error")
        aware = parsed.replace(tzinfo=PARIS_TZ) if parsed.tzinfo is None else parsed
        assumption = "localized_europe_paris" if parsed.tzinfo is None else "explicit_timezone"

    utc_dt = aware.astimezone(UTC)
    local_dt = utc_dt.astimezone(PARIS_TZ)
    return ParsedTimestamp(
        timestamp_utc=utc_dt.isoformat().replace("+00:00", "Z"),
        timestamp_local=local_dt.isoformat(),
        timestamp_assumption=assumption,
    )


def timestamp_columns(values: Iterable[Any]) -> pd.DataFrame:
    """Return timestamp columns for a sequence of raw timestamp values."""

    parsed = [parse_timestamp(value) for value in values]
    return pd.DataFrame(
        {
            "timestamp_utc": [item.timestamp_utc for item in parsed],
            "timestamp_local": [item.timestamp_local for item in parsed],
            "timestamp_assumption": [item.timestamp_assumption for item in parsed],
        }
    )


def standardize_iot_json(folder: Path) -> pd.DataFrame:
    """Convert IoT JSON records to a standard tabular schema."""

    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        source_format = "json_array" if first_nonspace(path) == "[" else "json_lines"
        for source_row, raw in enumerate(iter_json_records(path), start=1):
            ts = parse_timestamp(raw.get("timestamp"))
            temperature_value = raw.get("temperature", raw.get("Temperature"))
            humidity_value = raw.get("humidity", raw.get("Humidity"))
            rows.append(
                {
                    "source": "iot",
                    "source_file": path.name,
                    "source_format": source_format,
                    "source_row": source_row,
                    "timestamp_raw": raw.get("timestamp"),
                    "timestamp_utc": ts.timestamp_utc,
                    "timestamp_local": ts.timestamp_local,
                    "timestamp_assumption": ts.timestamp_assumption,
                    "sensor_id": raw.get("ID"),
                    "device_type": raw.get("type"),
                    "temperature_c": to_float(temperature_value),
                    "relative_humidity_pct": to_float(humidity_value),
                    "co2_ppm": to_float(raw.get("CO2")),
                    "tvoc_ppb": to_float(raw.get("TVOC")),
                    "pm1_ugm3": to_float(raw.get("PM1.0", raw.get("PM1"))),
                    "pm25_ugm3": to_float(raw.get("PM2.5")),
                    "pm10_ugm3": to_float(raw.get("PM10")),
                    "sound_level_db": to_float(raw.get("sound_level")),
                }
            )

    return pd.DataFrame(rows)


def standardize_aquacheck_json(folder: Path) -> pd.DataFrame:
    """Convert Aquacheck JSON records to a standard tabular schema."""

    rows: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")):
        source_format = "json_array" if first_nonspace(path) == "[" else "json_lines"
        for source_row, raw in enumerate(iter_json_records(path), start=1):
            ts = parse_timestamp(raw.get("timestamp"))
            rows.append(
                {
                    "source": "aquacheck",
                    "source_file": path.name,
                    "source_format": source_format,
                    "source_row": source_row,
                    "timestamp_raw": raw.get("timestamp"),
                    "timestamp_utc": ts.timestamp_utc,
                    "timestamp_local": ts.timestamp_local,
                    "timestamp_assumption": ts.timestamp_assumption,
                    "sensor_id": raw.get("ID"),
                    "device_type": raw.get("type"),
                    "soil_moisture_pct": to_float(raw.get("soilMoisture (%)")),
                    "temperature_c": to_float(raw.get("temperature", raw.get("temperature (degC)"))),
                    "relative_humidity_pct": to_float(raw.get("humidity", raw.get("humidity (%)"))),
                    "humidex_c": to_float(raw.get("humidex")),
                    "battery_level_pct": to_float(raw.get("batteryLevel")),
                }
            )

    return pd.DataFrame(rows)


def standardize_neusta_csv(path: Path) -> pd.DataFrame:
    """Convert the Neusta CSV to the same timestamp and naming conventions."""

    raw = pd.read_csv(path, low_memory=False)
    ts = timestamp_columns(raw["timestamp"])
    out = pd.DataFrame(
        {
            "source": "neusta",
            "source_file": path.name,
            "source_row": range(1, len(raw) + 1),
            "timestamp_raw": raw["timestamp"],
            "timestamp_utc": ts["timestamp_utc"],
            "timestamp_local": ts["timestamp_local"],
            "timestamp_assumption": ts["timestamp_assumption"],
            "temperature_c": pd.to_numeric(raw.get("Temperature"), errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(raw.get("Humidity"), errors="coerce"),
            "temperature_secondary_c": pd.to_numeric(raw.get("temperature"), errors="coerce"),
            "humidity_secondary_pct": pd.to_numeric(raw.get("humidity"), errors="coerce"),
            "pm1_ugm3": pd.to_numeric(raw.get("PM1"), errors="coerce"),
            "pm25_ugm3": pd.to_numeric(raw.get("PM2.5"), errors="coerce"),
            "pm10_ugm3": pd.to_numeric(raw.get("PM10"), errors="coerce"),
            "humidex_c": pd.to_numeric(raw.get("Humidex"), errors="coerce"),
            "vivabilite_binary": pd.to_numeric(raw.get("Vivabilite"), errors="coerce"),
        }
    )

    for column in ["22.57", "22.56", "22.58", "22.54", "22.53", "22.51"]:
        if column in raw.columns:
            safe_name = "unknown_" + column.replace(".", "_")
            out[safe_name] = pd.to_numeric(raw[column], errors="coerce")

    return out


def standardize_meteo_csv(path: Path) -> pd.DataFrame:
    """Convert the Meteo France CSV to standard names used by the pipeline."""

    raw = pd.read_csv(path, low_memory=False)
    ts = timestamp_columns(raw["validity_time"])
    return pd.DataFrame(
        {
            "source": "meteo_france",
            "source_file": path.name,
            "source_row": range(1, len(raw) + 1),
            "timestamp_raw": raw["validity_time"],
            "timestamp_utc": ts["timestamp_utc"],
            "timestamp_local": ts["timestamp_local"],
            "timestamp_assumption": ts["timestamp_assumption"],
            "station_name": raw.get("name"),
            "station_wmo": raw.get("geo_id_wmo"),
            "station_wigos": raw.get("geo_id_wigos"),
            "latitude": pd.to_numeric(raw.get("lat"), errors="coerce"),
            "longitude": pd.to_numeric(raw.get("lon"), errors="coerce"),
            "temperature_c": pd.to_numeric(raw.get("Temp_C"), errors="coerce"),
            "temperature_k": pd.to_numeric(raw.get("t"), errors="coerce"),
            "dew_point_c": pd.to_numeric(raw.get("Dew_C"), errors="coerce"),
            "relative_humidity_pct": pd.to_numeric(raw.get("u"), errors="coerce"),
            "wind_speed_mps": pd.to_numeric(raw.get("ff"), errors="coerce"),
            "wind_direction_deg": pd.to_numeric(raw.get("dd"), errors="coerce"),
            "pressure_pa": pd.to_numeric(raw.get("pres"), errors="coerce"),
            "sea_level_pressure_pa": pd.to_numeric(raw.get("pmer"), errors="coerce"),
            "rain_1h_mm": pd.to_numeric(raw.get("rr1"), errors="coerce"),
            "rain_3h_mm": pd.to_numeric(raw.get("rr3"), errors="coerce"),
            "rain_6h_mm": pd.to_numeric(raw.get("rr6"), errors="coerce"),
            "humidex_c": pd.to_numeric(raw.get("humidex"), errors="coerce"),
            "vivabilite_score_meteo": pd.to_numeric(raw.get("Vivabilite"), errors="coerce"),
        }
    )


def profile_dataframe(df: pd.DataFrame, timestamp_column: str = "timestamp_utc") -> dict[str, Any]:
    """Build a compact profile for a standardized output."""

    timestamps = pd.to_datetime(df[timestamp_column], errors="coerce", utc=True)
    profile: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "column_names": list(df.columns),
        "invalid_timestamps": int(timestamps.isna().sum()),
        "date_min_utc": timestamps.min().isoformat() if timestamps.notna().any() else None,
        "date_max_utc": timestamps.max().isoformat() if timestamps.notna().any() else None,
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "sensor_id" in df.columns:
        profile["sensor_count"] = int(df["sensor_id"].nunique(dropna=True))
    if "station_name" in df.columns:
        profile["station_count"] = int(df["station_name"].nunique(dropna=True))
    return profile


def write_standardized_csvs(dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Write all F8-UC3 standardized CSV files and return a summary."""

    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "iot": (
            output_dir / "iot_observations_standardized.csv",
            standardize_iot_json(dataset_dir / "iot-data"),
        ),
        "aquacheck": (
            output_dir / "aquacheck_observations_standardized.csv",
            standardize_aquacheck_json(dataset_dir / "aquacheck"),
        ),
        "neusta": (
            output_dir / "neusta_observations_standardized.csv",
            standardize_neusta_csv(dataset_dir / "donnees_neusta.csv"),
        ),
        "meteo_france": (
            output_dir / "meteo_france_observations_standardized.csv",
            standardize_meteo_csv(dataset_dir / "data202425_meteo_france.csv"),
        ),
    }

    summary: dict[str, Any] = {}
    for source, (path, df) in outputs.items():
        df.to_csv(path, index=False)
        summary[source] = {
            "output_file": str(path),
            **profile_dataframe(df),
        }

    return summary

