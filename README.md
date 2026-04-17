# BioBot / Neusta Comfort Prediction

Clean data-science rebuild for the BioBot / Neusta internship project.

The project prepares environmental sensor datasets for comfort, livability, and risk prediction. The current validated scope includes **F8-UC3**, **F8-UC4**, and an initial **F9 livability-score prediction baseline**:

- **F8-UC3:** convert raw datasets into standardized CSV files.
- **F8-UC4:** clean, impute, normalize, and aggregate the standardized data.
- **F9:** define the livability prediction problem, compare model families, analyze humidex thresholds, and test baseline prediction models.

Recommendation systems, alert generation, and model interpretation are intentionally postponed.

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

This clean repo contains the validated F8 data-preparation work and the first F9 livability-score prediction workflow.

| Area | Status |
|---|---|
| Raw dataset study | Used as context |
| F8-UC3 standardized CSV conversion | Complete |
| F8-UC4 cleaning, imputation, normalization, aggregation | Complete |
| F9 machine learning | Initial livability prediction complete |
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
    f9/                   # F9 prediction documentation and manager summary
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
      modeling/           # F9 feature preparation and metrics
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

## Run F9 Livability Prediction

Analyze humidex threshold bands:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_uc6_humidex_threshold_analysis.py
```

Test baseline livability-score models:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_uc7_test_livability_models.py
```

Optional advanced CNN-LSTM experiment:

```bash
python -m pip install -r requirements-advanced.txt
python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8
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

F9-UC7 livability prediction:

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Random Forest | 0.0033 | 0.0218 | 0.9967 |
| Histogram Gradient Boosting | 0.0033 | 0.0188 | 0.9975 |
| Ridge Regression | 0.1753 | 0.2473 | 0.5748 |
| Mean Baseline | 0.3450 | 0.3887 | -0.0503 |

## Key Scientific Decisions

- Datasets are kept separate during F8. They are not merged until location and timestamp alignment are justified.
- Neusta `vivabilite_binary` and Meteo France `vivabilite_score_meteo` are different targets and are not mixed.
- Imputation is limited to short gaps and every imputed value is flagged.
- Normalized columns are added for future baseline ML models, but raw cleaned values are preserved.
- The first F9 target is Neusta `vivabilite_binary_mean`.
- Very high tree-model scores are treated as a scientific warning: the target may be formula-derived from environmental variables, so the current model is best described as reproducing the Neusta score, not yet proving perceived human comfort.

## Documentation

- [F8-UC3 and F8-UC4 Pipeline](docs/f8/F8_UC3_UC4_pipeline.md)
- [F8 Manager Summary](docs/f8/F8_UC3_UC4_manager_summary.md)
- [Project Roadmap](docs/PROJECT_ROADMAP.md)
- [Data Structure Report](docs/DATA_STRUCTURE_REPORT.md)
- [VS Code Setup](docs/VSCODE_SETUP.md)
- [GitHub Setup](docs/GITHUB_SETUP.md)
- [Chat Handoff Guide](docs/CHAT_HANDOFF_GUIDE.md)
- [F9 Manager Summary](docs/f9/F9_MANAGER_SUMMARY.md)
- [F9 Problem Definition](docs/f9/F9_UC2_problem_definition.md)
- [F9 Model Review](docs/f9/F9_UC3_model_review.md)
- [F9 Humidex Thresholds](docs/f9/F9_UC6_humidex_thresholds.md)
- [F9 Model Testing Results](docs/f9/F9_UC7_model_testing_results.md)
- [F9 Advanced Models](docs/f9/F9_UC8_advanced_models.md)

## Next Recommended Step

Before moving into F10 alerts or recommendation systems:

- validate the meaning of Neusta `vivabilite_binary_mean`,
- run an ablation test without `humidex_c`,
- test next-step prediction, for example predicting livability 15 or 60 minutes ahead,
- define F10 risk labels from humidex and environmental thresholds.
