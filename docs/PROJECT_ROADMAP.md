# Project Roadmap

This roadmap keeps the internship project understandable and scientifically ordered.

## Current Principle

Do not start machine learning until the data pipeline is validated.

The previous project work is useful reference material, but this clean repository should contain only validated and explainable steps.

## Phase F8: Data Acquisition and Preparation

| Use case | Status | Objective | Output |
|---|---|---|---|
| F8-UC1 Study datasets | Context only | Understand raw sources, schemas, missingness, and target variables | Dataset study notes |
| F8-UC2 Import JSON | Context only | Understand JSON formats and timestamp issues | Import logic reused in UC3 |
| F8-UC3 Convert and format CSV | Complete | Produce standardized CSV files with consistent names and timestamps | `data/interim/*_standardized.csv` |
| F8-UC4 Clean, impute, normalize, aggregate | Complete | Produce analysis-ready source-specific tables | `data/processed/*_clean.csv` |

## Phase F9: Prediction and Machine Learning

F9 should start only after visual checks confirm the F8 outputs.

Recommended order:

1. Define the prediction problem.
2. Decide which target to use:
   - Neusta binary livability,
   - Meteo France 0 to 7 comfort score,
   - future rule-based risk level.
3. Create a time-based train/validation/test split.
4. Train simple baselines first:
   - majority class,
   - logistic regression,
   - decision tree,
   - random forest or gradient boosting.
5. Compare models using clear metrics.
6. Only then consider sequence models.

## Phase F10: Risk Detection

F10 should use explainable rules before ML alerts.

Recommended order:

1. Define comfort and danger thresholds.
2. Create low, medium, high risk labels.
3. Build rule-based alert examples.
4. Validate labels with plots and examples.
5. Train ML risk classification only if labels are reliable.

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

Create exploratory visualizations from F8-UC4 outputs:

- source coverage timeline,
- temperature and humidity time series,
- missingness after cleaning,
- imputation counts,
- Neusta livability distribution.

