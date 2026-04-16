# BioBot / Neusta Comfort Prediction

Clean data-science rebuild for the BioBot / Neusta internship project.

The project prepares environmental sensor datasets for future comfort, livability, and risk prediction. The current validated scope is **F8-UC3** and **F8-UC4**:

- **F8-UC3:** convert raw datasets into standardized CSV files.
- **F8-UC4:** clean, impute, normalize, and aggregate the standardized data.

Machine learning is intentionally not included yet. The data pipeline must be trusted before model training begins.

## Project Goals

The long-term project goal is to build a data science and machine learning pipeline for environmental comfort and livability prediction using:

- IoT indoor/environmental sensors.
- Meteo France weather observations.
- Neusta environmental/livability data.
- Aquacheck soil moisture data.

The pipeline should eventually support:

- comfortable and risky period detection,
- livability or thermal comfort prediction,
- model comparison,
- alerts and recommendations.

## Current Repository Status

This clean repo contains the validated F8 data-preparation work only.

| Area | Status |
|---|---|
| Raw dataset study | Used as context |
| F8-UC3 standardized CSV conversion | Complete |
| F8-UC4 cleaning, imputation, normalization, aggregation | Complete |
| F9 machine learning | Not started |
| F10 risk detection and alerts | Not started |

## Repository Structure

```text
BioBot-Neusta-Comfort/
  README.md
  requirements.txt
  pyproject.toml
  data/
    raw/                  # optional local raw data, not committed
    interim/              # generated standardized CSV files, not committed
    processed/            # generated cleaned CSV files, not committed
  docs/
    f8/                   # F8 documentation and manager summary
    GITHUB_SETUP.md
    PROJECT_ROADMAP.md
    VSCODE_SETUP.md
  notebooks/              # future exploratory notebooks
  reports/
    figures/              # small report figures
    tables/               # small reproducibility summaries
  scripts/                # runnable pipeline scripts
  src/
    biobot/
      data/               # data processing package
  tests/                  # lightweight validation tests
```

## Raw Data Location

The raw reference data currently lives outside the repository:

```text
/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset
```

Large raw and generated CSV files should not be committed to GitHub.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run F8-UC3

Convert raw data into standardized CSV files:

```bash
python scripts/f8_uc3_convert_to_standard_csv.py
```

Main outputs:

```text
data/interim/iot_observations_standardized.csv
data/interim/aquacheck_observations_standardized.csv
data/interim/neusta_observations_standardized.csv
data/interim/meteo_france_observations_standardized.csv
reports/tables/f8_uc3_standardization_summary.json
```

## Run F8-UC4

Clean, impute, normalize, and aggregate the standardized CSV files:

```bash
python scripts/f8_uc4_clean_impute_normalize_aggregate.py
```

Main outputs:

```text
data/processed/iot_15min_clean.csv
data/processed/aquacheck_15min_clean.csv
data/processed/neusta_15min_clean.csv
data/processed/meteo_france_1h_clean.csv
reports/tables/f8_uc4_cleaning_summary.json
```

## Generate Quality Figures

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f8_uc4_make_quality_figures.py
```

Main outputs:

```text
reports/figures/f8_uc4_row_counts.png
reports/figures/f8_uc4_out_of_range_counts.png
reports/figures/f8_uc4_imputation_counts.png
```

## Verified Results

F8-UC3 standardized:

| Source | Rows | Invalid timestamps |
|---|---:|---:|
| IoT | 857,574 | 4,427 |
| Aquacheck | 298,703 | 0 |
| Neusta | 395,039 | 0 |
| Meteo France | 468,570 | 0 |

F8-UC4 aggregated:

| Source | Input rows | Output rows | Aggregation |
|---|---:|---:|---|
| IoT | 857,574 | 77,124 | 15 minutes by sensor |
| Aquacheck | 298,703 | 106,332 | 15 minutes by sensor |
| Neusta | 395,039 | 5,679 | 15 minutes |
| Meteo France | 468,570 | 459,291 | 1 hour by station |

## Key Scientific Decisions

- Datasets are kept separate during F8. They are not merged until location and timestamp alignment are justified.
- Neusta `vivabilite_binary` and Meteo France `vivabilite_score_meteo` are different targets and are not mixed.
- Imputation is limited to short gaps and every imputed value is flagged.
- Normalized columns are added for future baseline ML models, but raw cleaned values are preserved.

## Documentation

- [F8-UC3 and F8-UC4 Pipeline](docs/f8/F8_UC3_UC4_pipeline.md)
- [F8 Manager Summary](docs/f8/F8_UC3_UC4_manager_summary.md)
- [Project Roadmap](docs/PROJECT_ROADMAP.md)
- [VS Code Setup](docs/VSCODE_SETUP.md)
- [GitHub Setup](docs/GITHUB_SETUP.md)

## Next Recommended Step

Before starting F9 machine learning, create exploratory visual checks from the processed CSV files:

- coverage timelines,
- temperature and humidity trends,
- missing values before and after imputation,
- Neusta livability target distribution,
- PM, CO2, TVOC outlier plots.

