# F8-UC3 and F8-UC4 Pipeline

This document describes the clean F8 work that should be shown as the validated rebuild. Older model files and previous notebooks are reference material only.

## Scope

For F8, the active tasks are:

| Use case | Task | Output |
|---|---|---|
| F8-UC3 | Convert and format data into standardized CSV files | `data/interim/*_standardized.csv` |
| F8-UC4 | Clean, impute, normalize, and aggregate data | `data/processed/*_clean.csv` |

F8-UC1 and F8-UC2 are used only as context. This work does not train ML models and does not create alerts.

## F8-UC3: Convert and Format Standardized CSV

### Objective

Convert the raw datasets into simple CSV files with consistent column names, timestamp handling, and provenance fields.

### Inputs

| Source | Input path | Format |
|---|---|---|
| IoT sensors | `/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset/iot-data/` | JSON array and JSON-lines |
| Aquacheck | `/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset/aquacheck/` | JSON-lines |
| Neusta | `/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset/donnees_neusta.csv` | CSV |
| Meteo France | `/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset/data202425_meteo_france.csv` | CSV |

### Expected Outputs

| Output file | Meaning |
|---|---|
| `data/interim/iot_observations_standardized.csv` | IoT observations with standard air quality and comfort columns |
| `data/interim/aquacheck_observations_standardized.csv` | Soil moisture observations with optional temperature/humidity fields |
| `data/interim/neusta_observations_standardized.csv` | Neusta labeled local observations |
| `data/interim/meteo_france_observations_standardized.csv` | Weather station observations |
| `reports/tables/f8_uc3_standardization_summary.json` | Reproducible row counts and timestamp checks |

### Standard Columns

Common columns:

| Column | Meaning |
|---|---|
| `source` | Dataset name |
| `source_file` | Original file name |
| `source_row` | Row number inside the source |
| `timestamp_raw` | Original timestamp text |
| `timestamp_utc` | Canonical UTC timestamp used for joining later |
| `timestamp_local` | Europe/Paris timestamp for human inspection |
| `timestamp_assumption` | Whether timezone was explicit or assumed |

IoT columns:

| Column | Meaning |
|---|---|
| `sensor_id` | Sensor identifier when present |
| `device_type` | Device type when present |
| `temperature_c` | Air temperature in degC |
| `relative_humidity_pct` | Relative humidity in percent |
| `co2_ppm` | Carbon dioxide concentration |
| `tvoc_ppb` | Total volatile organic compounds |
| `pm1_ugm3`, `pm25_ugm3`, `pm10_ugm3` | Particulate matter measurements |
| `sound_level_db` | Sound level |

Aquacheck columns:

| Column | Meaning |
|---|---|
| `sensor_id` | Soil station/sensor identifier |
| `soil_moisture_pct` | Soil moisture percentage |
| `temperature_c` | Optional local temperature |
| `relative_humidity_pct` | Optional local humidity |
| `humidex_c` | Optional humidex |
| `battery_level_pct` | Battery level, useful for reliability checks |

Neusta columns:

| Column | Meaning |
|---|---|
| `temperature_c`, `relative_humidity_pct` | Main local comfort variables |
| `temperature_secondary_c`, `humidity_secondary_pct` | Sparse secondary variables |
| `pm1_ugm3`, `pm25_ugm3`, `pm10_ugm3` | Sparse particulate matter variables |
| `humidex_c` | Thermal comfort index |
| `vivabilite_binary` | Binary livability target |
| `unknown_22_57`, etc. | Unknown original numeric columns kept for audit only |

Meteo France columns:

| Column | Meaning |
|---|---|
| `station_name`, `station_wmo`, `station_wigos` | Weather station identifiers |
| `latitude`, `longitude` | Station coordinates |
| `temperature_c`, `dew_point_c`, `relative_humidity_pct` | Weather comfort variables |
| `wind_speed_mps`, `wind_direction_deg` | Wind variables |
| `pressure_pa`, `sea_level_pressure_pa` | Pressure variables |
| `rain_1h_mm`, `rain_3h_mm`, `rain_6h_mm` | Rainfall windows |
| `humidex_c` | Weather humidex |
| `vivabilite_score_meteo` | Meteo livability score, 0 to 7 scale |

### Command

```bash
.venv/bin/python scripts/f8_uc3_convert_to_standard_csv.py
```

### Verification

Check that the generated summary has the expected row counts:

```bash
python3 -m json.tool reports/tables/f8_uc3_standardization_summary.json
```

Expected approximate rows:

| Source | Expected rows |
|---|---:|
| IoT | 857,574 |
| Aquacheck | 298,703 |
| Neusta | 395,039 |
| Meteo France | 468,570 |

## F8-UC4: Clean, Impute, Normalize, Aggregate

### Objective

Prepare source-specific datasets that are safe for later analysis and baseline modeling.

This step does four things:

1. Cleans impossible values using transparent physical ranges.
2. Imputes only short gaps inside each sensor or source.
3. Adds min-max normalized columns for later ML baselines.
4. Aggregates high-frequency observations into readable time intervals.

### Inputs

The F8-UC3 standardized CSV files in `data/interim/`.

### Expected Outputs

| Output file | Meaning |
|---|---|
| `data/processed/iot_15min_clean.csv` | IoT data aggregated by `sensor_id` and 15-minute interval |
| `data/processed/aquacheck_15min_clean.csv` | Aquacheck data aggregated by `sensor_id` and 15-minute interval |
| `data/processed/neusta_15min_clean.csv` | Neusta data aggregated to 15-minute intervals |
| `data/processed/meteo_france_1h_clean.csv` | Meteo France data aggregated by station and hour |
| `reports/tables/f8_uc4_cleaning_summary.json` | Cleaning, imputation, and normalization report |

### Cleaning Rules

Values outside these broad physical ranges are replaced with missing values before aggregation:

| Variable | Accepted range |
|---|---|
| Temperature | -20 to 60 degC |
| Relative humidity | 0 to 100 percent |
| CO2 | 250 to 5000 ppm |
| TVOC | 0 to 10000 ppb |
| PM1, PM2.5, PM10 | 0 to 1000 ug/m3 |
| Sound level | 20 to 120 dB |
| Soil moisture | 0 to 100 percent |
| Humidex | -20 to 80 |
| Battery level | 0 to 100 percent |
| Meteo wind speed | 0 to 75 m/s |
| Meteo wind direction | 0 to 360 degrees |
| Meteo vivability | 0 to 7 |
| Neusta vivability | 0 to 1 |

These ranges are intentionally conservative. They remove physically impossible sensor artifacts without making strict comfort judgments yet.

### Imputation Rule

Only short gaps are imputed:

- IoT, Aquacheck, Neusta: up to 4 aggregated steps.
- Meteo France: up to 2 hourly steps.

Each imputed feature gets a boolean column named like `temperature_c_was_imputed`.

### Normalization Rule

Each feature gets a min-max normalized version named like `temperature_c_norm`. Targets are not normalized.

### Command

```bash
.venv/bin/python scripts/f8_uc4_clean_impute_normalize_aggregate.py
```

### Verification

Check the generated summary:

```bash
python3 -m json.tool reports/tables/f8_uc4_cleaning_summary.json
```

Also inspect row counts:

```bash
wc -l data/interim/*_standardized.csv data/processed/*_clean.csv
```

## Recommended Visualizations

Produce these after the CSV files exist:

| Visualization | Purpose |
|---|---|
| Raw vs cleaned outlier counts | Show why cleaning was necessary |
| Missing values before and after imputation | Prove imputation is limited |
| Timeline coverage per source | Show where sources overlap |
| Temperature and humidity time series | Validate aggregation visually |
| PM, CO2, TVOC boxplots | Identify remaining air-quality spikes |
| Neusta `vivabilite_binary_mode` distribution | Prepare the future ML target |

The current rebuild includes an automatic figure script:

```bash
MPLCONFIGDIR=/Users/inesamovsesyan/Documents/Playground/BioBot/.cache/matplotlib \
  .venv/bin/python scripts/f8_uc4_make_quality_figures.py
```

Generated figures:

```text
reports/figures/f8_uc4_row_counts.png
reports/figures/f8_uc4_out_of_range_counts.png
reports/figures/f8_uc4_imputation_counts.png
```

## Manager Summary

F8-UC3 makes the raw data readable and consistent. F8-UC4 makes it analytically usable while keeping sources separate. This is the correct foundation before F9, because the project now has reproducible CSV outputs, timestamp assumptions, cleaning rules, imputation flags, and normalized variables.

## Current Run Results

The pipeline was run successfully on the raw reference data.

### F8-UC3 Results

| Source | Standardized rows | Invalid timestamps | Notes |
|---|---:|---:|---|
| IoT | 857,574 | 4,427 | Invalid timestamps are kept in the interim CSV but excluded from UC4 time aggregation |
| Aquacheck | 298,703 | 0 | 17 sensor/station IDs |
| Neusta | 395,039 | 0 | Binary `vivabilite_binary` target |
| Meteo France | 468,570 | 0 | 189 weather stations |

### F8-UC4 Results

| Source | Input rows | Clean aggregated rows | Aggregation |
|---|---:|---:|---|
| IoT | 857,574 | 77,124 | 15 minutes by sensor |
| Aquacheck | 298,703 | 106,332 | 15 minutes by sensor |
| Neusta | 395,039 | 5,679 | 15 minutes |
| Meteo France | 468,570 | 459,291 | 1 hour by weather station |

Main quality findings:

- IoT had 78 impossible temperature values, 34 impossible CO2 values, and a few impossible PM/sound values replaced with missing values.
- IoT sensor IDs were normalized from 7 raw non-null forms to 5 cleaned IDs including `unknown_sensor`.
- Aquacheck had 5 impossible temperature values and 5 impossible humidex values replaced with missing values.
- Neusta values were mostly physically valid, but secondary temperature/humidity and PM columns are very sparse.
- Meteo France rainfall columns had many out-of-range values under the current conservative rules; these should be checked against Meteo France metadata before using rainfall as a modeling feature.

Generated data files are intentionally not committed because they are large. Generated summary reports are small and can be committed:

```text
reports/tables/f8_uc3_standardization_summary.json
reports/tables/f8_uc4_cleaning_summary.json
```
