import json
from pathlib import Path

import pytest

from biobot.data.standardize import (
    iter_json_records,
    parse_timestamp,
    standardize_aquacheck_json,
    standardize_iot_json,
    standardize_meteo_csv,
    standardize_neusta_csv,
    to_float,
)


def test_parse_french_local_timestamp_to_utc():
    parsed = parse_timestamp("20-09-2024 02:46:27")

    assert parsed.timestamp_utc == "2024-09-20T00:46:27Z"
    assert parsed.timestamp_assumption == "localized_europe_paris"


def test_parse_explicit_timezone_timestamp_to_utc():
    parsed = parse_timestamp("2026-03-26T00:03:42+0100")

    assert parsed.timestamp_utc == "2026-03-25T23:03:42Z"
    assert parsed.timestamp_assumption == "explicit_timezone"


def test_to_float_handles_nan_strings():
    assert to_float("NaN") is None
    assert to_float("nan") is None
    assert to_float("22,5") == 22.5


# ── FIX 1: iter_json_records ─────────────────────────────────────────────────


def test_iter_json_records_yields_from_json_array(tmp_path: Path):
    p = tmp_path / "records.json"
    p.write_text('[{"a": 1}, {"b": 2}]', encoding="utf-8")
    result = list(iter_json_records(p))
    assert result == [{"a": 1}, {"b": 2}]


def test_iter_json_records_handles_ndjson_and_skips_malformed(tmp_path: Path):
    p = tmp_path / "records.ndjson"
    p.write_text('{"x": 10}\nBAD LINE\n{"y": 20}\n', encoding="utf-8")
    result = list(iter_json_records(p))
    assert len(result) == 2
    assert result[0] == {"x": 10}
    assert result[1] == {"y": 20}


# ── FIX 1: standardize_* functions ───────────────────────────────────────────


def test_standardize_iot_json_produces_standard_columns(tmp_path: Path):
    folder = tmp_path / "iot"
    folder.mkdir()
    record = {
        "timestamp": "2024-01-01 10:00:00",
        "ID": "12345678901234",
        "temperature": 22.5,
        "humidity": 55.0,
        "CO2": 800,
        "TVOC": 100,
        "PM1.0": 5.0,
        "PM2.5": 8.0,
        "PM10": 12.0,
        "sound_level": 45.0,
        "type": "indoor",
    }
    (folder / "iot_test.json").write_text(json.dumps([record]), encoding="utf-8")
    df = standardize_iot_json(folder)
    for col in ("source", "sensor_id", "temperature_c", "relative_humidity_pct", "co2_ppm", "timestamp_utc"):
        assert col in df.columns, f"missing column: {col}"
    assert df["temperature_c"].iloc[0] == pytest.approx(22.5)
    assert df["sensor_id"].iloc[0] == "12345678901234"


def test_standardize_aquacheck_json_maps_soil_moisture(tmp_path: Path):
    folder = tmp_path / "aquacheck"
    folder.mkdir()
    record = {
        "timestamp": "2024-01-01 10:00:00",
        "ID": "PROBE01",
        "soilMoisture (%)": 35.5,
        "temperature": 20.0,
        "humidity": 60.0,
        "humidex": 22.0,
        "batteryLevel": 80.0,
    }
    (folder / "aquacheck_test.json").write_text(json.dumps([record]), encoding="utf-8")
    df = standardize_aquacheck_json(folder)
    assert "soil_moisture_pct" in df.columns
    assert df["soil_moisture_pct"].iloc[0] == pytest.approx(35.5)


def test_standardize_neusta_csv_maps_vivabilite_binary(tmp_path: Path):
    csv_content = "timestamp,Temperature,Humidity,Humidex,Vivabilite\n2024-01-01 10:00:00,22.0,55.0,24.0,1\n"
    p = tmp_path / "neusta.csv"
    p.write_text(csv_content, encoding="utf-8")
    df = standardize_neusta_csv(p)
    assert "vivabilite_binary" in df.columns
    assert df["vivabilite_binary"].iloc[0] == pytest.approx(1.0)


def test_standardize_meteo_csv_maps_temperature_and_score(tmp_path: Path):
    header = "validity_time,name,geo_id_wmo,geo_id_wigos,lat,lon,Temp_C,t,Dew_C,u,ff,dd,pres,pmer,rr1,rr3,rr6,humidex,Vivabilite"
    row = "2024-01-01T10:00:00Z,Paris,07149,,48.85,2.35,18.5,291.5,12.0,65.0,3.5,180.0,101325.0,101325.0,0.0,0.0,0.0,20.5,2"
    p = tmp_path / "meteo.csv"
    p.write_text(f"{header}\n{row}\n", encoding="utf-8")
    df = standardize_meteo_csv(p)
    assert "temperature_c" in df.columns
    assert "vivabilite_score_meteo" in df.columns
    assert df["temperature_c"].iloc[0] == pytest.approx(18.5)
    assert df["vivabilite_score_meteo"].iloc[0] == pytest.approx(2.0)

