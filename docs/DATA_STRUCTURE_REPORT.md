# Data Structure Report

This report explains the structure of the BioBot / Neusta data pipeline after the clean F8 rebuild.

The current validated scope is:

- F8-UC3: convert and format raw data into standardized CSV files.
- F8-UC4: clean, impute, normalize, and aggregate the standardized files.

Machine learning is intentionally not included in this report.

## 1. Raw Data Sources

The raw reference data is stored outside the repository:

```text
/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset
```

| Source | Raw path | Format | Main role |
|---|---|---|---|
| IoT sensors | `iot-data/` | JSON arrays and JSON-lines | Indoor/environment sensor observations |
| Aquacheck | `aquacheck/` | JSON-lines | Soil moisture and related environmental observations |
| Neusta | `donnees_neusta.csv` | CSV | Local environmental data with binary livability target |
| Meteo France | `data202425_meteo_france.csv` | CSV | Outdoor weather station data with weather variables |

## 2. F8-UC3 Standardized CSV Structure

F8-UC3 converts raw data to consistent CSV files in:

```text
data/interim/
```

Generated files:

| File | Rows | Purpose |
|---|---:|---|
| `iot_observations_standardized.csv` | 857,574 | Standardized IoT sensor observations |
| `aquacheck_observations_standardized.csv` | 298,703 | Standardized Aquacheck observations |
| `neusta_observations_standardized.csv` | 395,039 | Standardized Neusta labeled observations |
| `meteo_france_observations_standardized.csv` | 468,570 | Standardized weather station observations |

### Common UC3 Columns

These columns appear in standardized files where relevant:

| Column | Meaning |
|---|---|
| `source` | Dataset name, for example `iot`, `aquacheck`, `neusta`, or `meteo_france` |
| `source_file` | Original raw file name |
| `source_format` | JSON format when relevant: `json_array` or `json_lines` |
| `source_row` | Row number inside the original source file |
| `timestamp_raw` | Original timestamp string |
| `timestamp_utc` | Parsed timestamp normalized to UTC |
| `timestamp_local` | Timestamp shown in Europe/Paris local time |
| `timestamp_assumption` | Whether timezone was explicit or assumed |

The timestamp strategy is important:

- ISO timestamps with timezone offsets are converted directly to UTC.
- Naive timestamps such as `20-09-2024 02:46:27` are treated as Europe/Paris local time.
- Invalid timestamps remain visible in UC3 but are excluded from UC4 aggregation.

### IoT Standardized Columns

| Column | Meaning | Unit |
|---|---|---|
| `sensor_id` | Sensor identifier when available | ID |
| `device_type` | Device type when available | Text |
| `temperature_c` | Air temperature | degC |
| `relative_humidity_pct` | Relative humidity | Percent |
| `co2_ppm` | Carbon dioxide | ppm |
| `tvoc_ppb` | Total volatile organic compounds | ppb |
| `pm1_ugm3` | PM1 particulate matter | ug/m3 |
| `pm25_ugm3` | PM2.5 particulate matter | ug/m3 |
| `pm10_ugm3` | PM10 particulate matter | ug/m3 |
| `sound_level_db` | Sound level | dB |

Data quality note: IoT has 4,427 invalid timestamps and several malformed sensor ID formats.

### Aquacheck Standardized Columns

| Column | Meaning | Unit |
|---|---|---|
| `sensor_id` | Aquacheck station/sensor identifier | ID |
| `device_type` | Device type when available | Text |
| `soil_moisture_pct` | Soil moisture | Percent |
| `temperature_c` | Local temperature when available | degC |
| `relative_humidity_pct` | Local relative humidity when available | Percent |
| `humidex_c` | Humidex when available | degC-like index |
| `battery_level_pct` | Device battery level | Percent |

### Neusta Standardized Columns

| Column | Meaning | Unit |
|---|---|---|
| `temperature_c` | Main temperature | degC |
| `relative_humidity_pct` | Main relative humidity | Percent |
| `temperature_secondary_c` | Sparse secondary temperature field | degC |
| `humidity_secondary_pct` | Sparse secondary humidity field | Percent |
| `pm1_ugm3` | PM1 particulate matter | ug/m3 |
| `pm25_ugm3` | PM2.5 particulate matter | ug/m3 |
| `pm10_ugm3` | PM10 particulate matter | ug/m3 |
| `humidex_c` | Humidex | degC-like index |
| `vivabilite_binary` | Binary livability target | 0 or 1 |
| `unknown_22_57`, etc. | Unknown original columns kept for audit | Unknown |

Important target note: Neusta `vivabilite_binary` is a binary target. It should not be mixed with Meteo France `vivabilite_score_meteo`.

### Meteo France Standardized Columns

| Column | Meaning | Unit |
|---|---|---|
| `station_name` | Weather station name | Text |
| `station_wmo` | WMO station ID | ID |
| `station_wigos` | WIGOS station ID | ID |
| `latitude`, `longitude` | Station coordinates | Degrees |
| `temperature_c` | Air temperature | degC |
| `temperature_k` | Raw air temperature | Kelvin |
| `dew_point_c` | Dew point temperature | degC |
| `relative_humidity_pct` | Relative humidity | Percent |
| `wind_speed_mps` | Wind speed | m/s |
| `wind_direction_deg` | Wind direction | Degrees |
| `pressure_pa` | Station pressure | Pa |
| `sea_level_pressure_pa` | Sea-level pressure | Pa |
| `rain_1h_mm`, `rain_3h_mm`, `rain_6h_mm` | Rain accumulation windows | mm |
| `humidex_c` | Humidex | degC-like index |
| `vivabilite_score_meteo` | Meteo livability score | 0 to 7 |

## 3. F8-UC4 Processed CSV Structure

F8-UC4 produces cleaned and aggregated files in:

```text
data/processed/
```

Generated files:

| File | Rows | Aggregation |
|---|---:|---|
| `iot_15min_clean.csv` | 77,124 | 15 minutes by `sensor_id` |
| `aquacheck_15min_clean.csv` | 106,332 | 15 minutes by `sensor_id` |
| `neusta_15min_clean.csv` | 5,679 | 15 minutes |
| `meteo_france_1h_clean.csv` | 459,291 | 1 hour by weather station |

### Common UC4 Columns

| Column type | Meaning |
|---|---|
| Original cleaned numeric columns | Aggregated mean values after range cleaning |
| `record_count` | Number of raw observations inside the interval |
| `*_was_imputed` | Boolean flag showing whether that interval value was imputed |
| `*_norm` | Min-max normalized version of the feature |

Example:

```text
temperature_c
temperature_c_was_imputed
temperature_c_norm
```

This design keeps three pieces of information:

1. the real cleaned value,
2. whether the value was filled,
3. the normalized value for future machine learning.

## 4. Cleaning Rules

UC4 replaces physically impossible values with missing values before aggregation.

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
| Wind speed | 0 to 75 m/s |
| Wind direction | 0 to 360 degrees |
| Neusta livability | 0 to 1 |
| Meteo livability | 0 to 7 |

These are broad plausibility rules, not comfort rules. Comfort and risk thresholds belong later in F10.

## 5. Imputation Strategy

The pipeline only imputes short gaps:

- IoT, Aquacheck, Neusta: up to 4 aggregated steps.
- Meteo France: up to 2 hourly steps.

Imputation is done after aggregation. This prevents the pipeline from inventing many high-frequency raw observations.

Every imputed feature is flagged with a column ending in:

```text
_was_imputed
```

## 6. Normalization Strategy

Min-max normalization is added for future ML models.

Example:

```text
temperature_c_norm = (temperature_c - min_temperature_c) / (max_temperature_c - min_temperature_c)
```

Targets are not normalized.

## 7. Why Datasets Are Not Merged Yet

The datasets are intentionally kept separate in F8.

Reason:

- IoT sensors and Aquacheck stations may not measure the same location.
- Meteo France stations are outdoor weather stations and may be geographically distant.
- Neusta contains the main binary livability target.
- Meteo France contains a different 0 to 7 livability score.

Merging should happen only after the project defines:

- which target is being predicted,
- which location each sensor represents,
- what time tolerance is scientifically acceptable,
- whether the task is current comfort prediction or future forecasting.

## 8. Existing Visualizations

The current repo already contains these figures:

| Figure | Meaning |
|---|---|
| `reports/figures/f8_uc4_row_counts.png` | Rows before and after aggregation |
| `reports/figures/f8_uc4_out_of_range_counts.png` | Values removed by physical range rules |
| `reports/figures/f8_uc4_imputation_counts.png` | Values imputed after aggregation |

Generate them with:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f8_uc4_make_quality_figures.py
```

## 9. Recommended Next Visualizations

Before F9, add:

| Visualization | Purpose |
|---|---|
| Source coverage timeline | Show date overlap between sources |
| Temperature/humidity time series | Validate aggregation visually |
| CO2/TVOC/PM boxplots | Detect remaining air-quality spikes |
| Neusta livability distribution | Understand the future ML target |
| Missingness heatmap | Show remaining missing values after UC4 |

## 10. Main Reports

Reproducible summary files:

```text
reports/tables/f8_uc3_standardization_summary.json
reports/tables/f8_uc4_cleaning_summary.json
```

Human-readable documentation:

```text
README.md
docs/f8/F8_UC3_UC4_pipeline.md
docs/f8/F8_UC3_UC4_manager_summary.md
docs/DATA_STRUCTURE_REPORT.md
```

