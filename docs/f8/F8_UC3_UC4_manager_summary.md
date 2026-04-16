# F8-UC3 and F8-UC4 Manager Summary

## Objective

The active F8 work was limited to two tasks:

- F8-UC3: convert and format raw data into standardized CSV files.
- F8-UC4: clean, impute, normalize, and aggregate the standardized data.

The goal was to create a reliable data foundation before any machine learning work.

## Work Completed

### F8-UC3

Four standardized CSV files were generated:

| Source | Output |
|---|---|
| IoT | `data/interim/iot_observations_standardized.csv` |
| Aquacheck | `data/interim/aquacheck_observations_standardized.csv` |
| Neusta | `data/interim/neusta_observations_standardized.csv` |
| Meteo France | `data/interim/meteo_france_observations_standardized.csv` |

The standardization step created consistent timestamp columns, source provenance columns, and harmonized variable names such as `temperature_c`, `relative_humidity_pct`, `pm25_ugm3`, and `humidex_c`.

### F8-UC4

Four cleaned and aggregated CSV files were generated:

| Source | Output | Rows |
|---|---|---:|
| IoT | `data/processed/iot_15min_clean.csv` | 77,124 |
| Aquacheck | `data/processed/aquacheck_15min_clean.csv` | 106,332 |
| Neusta | `data/processed/neusta_15min_clean.csv` | 5,679 |
| Meteo France | `data/processed/meteo_france_1h_clean.csv` | 459,291 |

The cleaning step applied physical plausibility rules, limited short-gap imputation, min-max normalization, and time aggregation.

## Important Decisions

- The datasets were not merged yet because source location and timestamp alignment need scientific justification.
- Neusta `vivabilite_binary` and Meteo France `vivabilite_score_meteo` are kept separate because they use different target scales.
- Imputed values are marked with explicit boolean columns such as `temperature_c_was_imputed`.
- Large generated CSV files are ignored by git; small JSON summaries are kept for reproducibility.

## Main Data Quality Findings

- IoT contains 4,427 invalid timestamps.
- IoT contains malformed sensor IDs, now normalized during UC4.
- IoT has a small number of physically impossible values, such as invalid temperatures and CO2 readings.
- Aquacheck has a few impossible temperature and humidex values.
- Neusta PM variables are very sparse.
- Meteo rainfall columns require additional metadata verification before being used as reliable model features.

## Verification Files

The reproducible reports are:

```text
reports/tables/f8_uc3_standardization_summary.json
reports/tables/f8_uc4_cleaning_summary.json
```

## Next Recommended Step

Before starting F9, create visual checks from the processed files: missingness before/after imputation, coverage timelines, time-series plots for temperature/humidity, and the distribution of Neusta `vivabilite_binary_mode`.
