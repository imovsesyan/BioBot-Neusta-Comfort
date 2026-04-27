# BioBot / Neusta Comfort Prediction

Clean data-science rebuild for the BioBot / Neusta internship project.

The project prepares environmental sensor datasets for comfort, livability, risk prediction, and personalized comfort recommendations. The current validated scope includes **F8-UC3**, **F8-UC4**, **F9 livability-score prediction**, **F10 risk detection**, and **F11 personalized recommendations**:

- **F8-UC3:** convert raw datasets into standardized CSV files.
- **F8-UC4:** clean, impute, normalize, and aggregate the standardized data.
- **F9:** define the livability prediction problem, compare model families, analyze humidex thresholds, and test baseline prediction models.
- **F10:** define humidex risk periods, generate rule-based alerts, and test ML risk classification.
- **F11:** build a personalized comfort recommendation system using synthetic user profiles, rule-based logic, and AI-assisted rephrasing.

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

| Area | Status |
|---|---|
| Raw dataset study | Used as context |
| F8-UC3 standardized CSV conversion | Complete |
| F8-UC4 cleaning, imputation, normalization, aggregation | Complete |
| F9 machine learning | Livability prediction + ablation complete |
| F10 risk detection and alerts | Rule and ML workflow + ablation complete |
| F11 personalized recommendations | Rule-based + AI rephrasing complete |

## Repository Structure

```text
BioBot-Neusta-Comfort/
  README.md
  CLAUDE.md               # architecture and scientific constraints reference
  requirements.txt
  pyproject.toml
  data/
    raw/                  # optional local raw data, not committed
    interim/              # generated standardized CSV files, not committed
    processed/            # generated cleaned CSV files (committed as .csv.gz)
    outputs/              # F11 recommendation output CSV
  docs/
    f8/                   # F8 documentation and manager summary
    f9/                   # F9 prediction documentation, ablation, and model review
    f10/                  # F10 risk detection documentation and ablation
    f11/                  # F11 recommendation system research findings
    GITHUB_SETUP.md
    PROJECT_ROADMAP.md
    VSCODE_SETUP.md
  notebooks/              # exploratory notebooks
  reports/
    figures/              # report figures
    tables/               # reproducibility summaries and prediction examples
  scripts/                # runnable pipeline scripts
  src/
    biobot/
      data/               # F8 data processing package
      modeling/           # F9 feature preparation and metrics
      risk/               # F10 risk rules and alert helpers
      recommendations/    # F11 user profile, rule recommender, AI recommender
  tests/                  # validation tests
```

## Raw Data Location

The raw reference data currently lives outside the repository:

```text
/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset
```

Large raw and generated CSV files should not be committed to GitHub.

The final cleaned F8-UC4 datasets are available in GitHub as compressed CSV files:

```text
data/processed/iot_15min_clean.csv.gz
data/processed/aquacheck_15min_clean.csv.gz
data/processed/neusta_15min_clean.csv.gz
data/processed/meteo_france_1h_clean.csv.gz
```

They can be read directly with pandas, for example:

```python
import pandas as pd

neusta = pd.read_csv("data/processed/neusta_15min_clean.csv.gz")
```

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

For GitHub sharing, the four final CSV files are compressed as `.csv.gz`.

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

Compare classical ML, XGBoost, and CNN-LSTM:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_compare_ml_dl_models.py
```

Optional advanced CNN-LSTM experiment:

```bash
python -m pip install -r requirements-advanced.txt
python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8
```

## Run F10 Risk Detection

Define livable, discomfort, high-risk, and dangerous periods:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc1_define_livable_dangerous_periods.py
```

Generate rule-based alerts from the risk labels:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc3_generate_rule_alerts.py
```

Train ML classifiers for the rule-derived risk labels:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc4_train_risk_classifier.py
```

## Run F11 Personalized Recommendations

Generate rule-based and AI-personalized comfort recommendations for every row in the F10 output:

```bash
python scripts/f11_uc1_uc5_recommendations.py
```

To enable AI rephrasing via Claude Haiku (optional, requires an Anthropic API key):

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/f11_uc1_uc5_recommendations.py
```

Main output:

```text
data/outputs/f11_recommendations.csv
```

Output columns include `rec_action`, `rec_clothing_advice`, `rec_activity_advice`, `rec_alert` (non-empty for vulnerable profiles at elevated risk), and `rec_ai_text` (populated only when AI is enabled).

> This output is informational only and does not constitute medical advice.

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
| Validation-weighted tree blend | 0.0034 | 0.0201 | 0.9972 |
| Equal-weight tree blend | 0.0034 | 0.0200 | 0.9972 |
| XGBoost | 0.0036 | 0.0210 | 0.9969 |
| Ridge Regression | 0.1753 | 0.2473 | 0.5748 |
| CNN-LSTM | 0.3000 | 0.4174 | -0.1879 |
| Mean Baseline | 0.3450 | 0.3887 | -0.0503 |

F10 risk detection:

| Task | Main result |
|---|---|
| F10-UC1 | Score-based livability status added: Neusta actual score has 3,237 livable and 1,078 not-livable rows; F9 predicted test score has 532 livable and 116 not-livable rows. Humidex heat-risk labels are kept as a separate safety layer. |
| F10-UC3 | 97,694 simulated rule-based alerts generated from Meteo France, including 3 critical humidex alerts. |
| F10-UC4 | Random Forest classifier reached macro F1 1.0000 on rule-derived risk labels. |
| F10 ablation | Removing `humidex_c` from F10-UC4 drops macro F1 from 1.0000 to 0.6667, confirming humidex is the primary signal. |

F11 personalized recommendations:

| Task | Main result |
|---|---|
| F11-UC1 | Rule-based recommender covers all 5 humidex risk levels × 8 synthetic profiles (40 cells). Actions sourced from WHO, Santé publique France, and ASHRAE 55. Vulnerable profiles at high_risk/dangerous/critical always receive a non-empty alert. |
| F11-UC5 | AI rephrasing layer uses Claude Haiku with prompt caching. Graceful fallback to rule text when API key is absent. In-memory cache avoids redundant API calls within a run. |

## Key Scientific Decisions

- Datasets are kept separate during F8. They are not merged until location and timestamp alignment are justified.
- Neusta `vivabilite_binary` and Meteo France `vivabilite_score_meteo` are different targets and are not mixed.
- Imputation is limited to short gaps and every imputed value is flagged.
- Normalized columns are added for future baseline ML models, but raw cleaned values are preserved.
- The first F9 target is Neusta `vivabilite_binary_mean`.
- Very high tree-model scores are treated as a scientific warning: the target may be formula-derived from environmental variables, so the current model is best described as reproducing the Neusta score, not yet proving perceived human comfort.
- F10-UC1 separates overall score-based livability status from humidex heat danger. The current Neusta binary target is interpreted as `score < 0.5 = livable` and `score >= 0.5 = not livable`, but this direction should be confirmed with the project owner.
- F10-UC4 currently classifies humidex-rule labels. It should not be interpreted as independent medical risk prediction.

## Documentation

### Project

- [Project Roadmap](docs/PROJECT_ROADMAP.md)
- [Data Structure Report](docs/DATA_STRUCTURE_REPORT.md)
- [Chat Handoff Guide](docs/CHAT_HANDOFF_GUIDE.md)
- [VS Code Setup](docs/VSCODE_SETUP.md)
- [GitHub Setup](docs/GITHUB_SETUP.md)

### F8 — Data Preparation

- [F8-UC3 and F8-UC4 Pipeline](docs/f8/F8_UC3_UC4_pipeline.md)
- [F8 Manager Summary](docs/f8/F8_UC3_UC4_manager_summary.md)

### F9 — Livability Prediction

- [F9 Manager Summary](docs/f9/F9_MANAGER_SUMMARY.md)
- [F9 Problem Definition](docs/f9/F9_UC2_problem_definition.md)
- [F9 Model Review](docs/f9/F9_UC3_model_review.md)
- [F9 Humidex Thresholds](docs/f9/F9_UC6_humidex_thresholds.md)
- [F9 Model Testing Results](docs/f9/F9_UC7_model_testing_results.md)
- [F9 Ablation — No Humidex](docs/f9/F9_UC7_ablation.md)
- [F9 Advanced Models (CNN-LSTM)](docs/f9/F9_UC8_advanced_models.md)
- [F9 ML vs Deep Learning Comparison](docs/f9/F9_ML_VS_DL_COMPARISON.md)
- [F9 Ensemble Modeling](docs/f9/F9_ENSEMBLE_MODELING.md)

### F10 — Risk Detection and Alerts

- [F10 Manager Summary](docs/f10/F10_MANAGER_SUMMARY.md)
- [F10 Livable and Dangerous Periods](docs/f10/F10_UC1_livable_dangerous_periods.md)
- [F10 Rule-Based Alerts](docs/f10/F10_UC3_rule_based_alerts.md)
- [F10 Risk Classification](docs/f10/F10_UC4_risk_classification.md)
- [F10 Ablation — No Humidex](docs/f10/F10_UC4_ablation.md)

### F11 — Personalized Recommendations

- [F11 Research Findings and Strategy](docs/f11/F11_research_recommendations.md)

## Next Recommended Steps

- Confirm the meaning of Neusta `vivabilite_binary_mean` with the data provider — the current direction (score ≥ 0.5 = not livable) is unverified.
- Extend F11 profiles from binary fields (8 combos) to 3-bin categorical (activity/clothing/vulnerability → 36 buckets × 5 risk levels) as recommended in the F11 research document.
- Author a complete rule table from WHO / Santé publique France / ASHRAE 55 sources and add a qualitative evaluation against published guidance.
- Test next-step prediction — for example predicting livability 15 or 60 minutes ahead.
- Add non-humidex risk factors if validated thresholds are available.
