# Project Roadmap

This roadmap keeps the internship project understandable and scientifically ordered.

## Current Principle

Do not start advanced machine learning until the data pipeline and baseline models are validated.

The previous project work is useful reference material, but this clean repository should contain only validated and explainable steps.

## Phase F8: Data Acquisition and Preparation

| Use case | Status | Objective | Output |
|---|---|---|---|
| F8-UC1 Study datasets | Context only | Understand raw sources, schemas, missingness, and target variables | Dataset study notes |
| F8-UC2 Import JSON | Context only | Understand JSON formats and timestamp issues | Import logic reused in UC3 |
| F8-UC3 Convert and format CSV | Complete | Produce standardized CSV files with consistent names and timestamps | `data/interim/*_standardized.csv` |
| F8-UC4 Clean, impute, normalize, aggregate | Complete | Produce analysis-ready source-specific tables | `data/processed/*_clean.csv` |

## Phase F9: Prediction and Machine Learning

F9 has now started with Neusta livability score prediction.

Recommended order:

| Use case | Status | Objective | Output |
|---|---|---|---|
| F9-UC2 Problem definition | Complete | Define livability-score prediction and challenges | `docs/f9/F9_UC2_problem_definition.md` |
| F9-UC3 Model review | Complete | Compare relevant model families | `docs/f9/F9_UC3_model_review.md` |
| F9-UC6 Humidex thresholds | Complete | Identify critical humidex zones | `docs/f9/F9_UC6_humidex_thresholds.md` |
| F9-UC7 Test prediction models | Complete | Compare baseline score-prediction models, including XGBoost | `reports/tables/f9_uc7_livability_model_results.json` |
| F9-UC8 Advanced models | Initial test complete | Test CNN-LSTM as an advanced candidate | `reports/tables/f9_uc8_sequence_model_results.json` |

Additional comparison output:

```text
reports/tables/f9_ml_vs_dl_comparison.json
reports/figures/f9_ml_vs_dl_comparison.png
```

Ensemble modeling note:

```text
docs/f9/F9_ENSEMBLE_MODELING.md
```

## Phase F10: Risk Detection

F10 uses explainable humidex rules before ML classification.

| Use case | Status | Objective | Output |
|---|---|---|---|
| F10-UC1 Livable and dangerous periods | Complete | Define humidex-based risk periods | `reports/tables/f10_uc1_livable_dangerous_summary.json` |
| F10-UC3 Rule-based alerts | Complete | Generate simulated alerts from thresholds | `reports/tables/f10_uc3_rule_alert_summary.json` |
| F10-UC4 Risk classification | Complete | Train ML classifiers to reproduce rule-derived risk levels | `reports/tables/f10_uc4_risk_classifier_results.json` |

Important limitation: F10-UC4 currently classifies rule-derived labels, not independent real-world medical incident labels.

## Documentation Standard

Every task should include:

- objective,
- input files,
- expected output,
- implementation steps,
- verification command,
- results,
- limitations,
- next step.

## Immediate Next Step

The immediate next improvements are:

- verify how Neusta `vivabilite_binary_mean` was created,
- rerun F9-UC7 without humidex features to test target leakage risk,
- define a future-horizon target, for example livability 15 or 60 minutes ahead,
- add non-humidex risk rules only when their thresholds are scientifically justified.
