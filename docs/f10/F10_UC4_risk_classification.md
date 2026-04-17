# F10-UC4 Risk Classification

## Objective

Train machine learning classifiers to predict the rule-derived risk level.

The target is:

```text
risk_level
```

The target classes are:

```text
livable, discomfort, high_risk, dangerous
```

## Input Files

```text
data/processed/f10_meteo_risk_labels.csv
```

## Implementation

The classifier workflow is implemented in:

```text
scripts/f10_uc4_train_risk_classifier.py
src/biobot/risk/rules.py
```

Code examples:

```python
models, skipped_models = build_classifiers(args.random_state, len(RISK_LEVEL_ORDER))
```

```python
test_metrics = evaluate_classifier(model_name, y_test, test_pred, RISK_LEVEL_ORDER)
```

## Features

The model uses Meteo France weather features:

```text
temperature_c
dew_point_c
relative_humidity_pct
wind_speed_mps
pressure_pa
rain_1h_mm
humidex_c
hour_sin
hour_cos
month_sin
month_cos
```

## Command

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc4_train_risk_classifier.py
```

## Outputs

```text
reports/tables/f10_uc4_risk_classifier_results.json
reports/tables/f10_uc4_risk_classifier_prediction_examples.csv
reports/figures/f10_uc4_classifier_comparison.png
reports/figures/f10_uc4_best_classifier_confusion_matrix.png
data/processed/f10_risk_classifier_test_predictions.csv
```

## Model Results

| Model | Macro F1 | Balanced accuracy | Accuracy |
|---|---:|---:|---:|
| Random Forest classifier | 1.0000 | 1.0000 | 1.0000 |
| XGBoost classifier | 0.9891 | 0.9835 | 0.9998 |
| Histogram Gradient Boosting classifier | 0.9859 | 0.9862 | 0.9997 |
| Balanced logistic regression | 0.9458 | 0.9885 | 0.9950 |
| Most frequent baseline | 0.2308 | 0.2500 | 0.8571 |

Best model:

```text
random_forest_classifier
```

## Visualization

Recommended figures for the report:

```text
reports/figures/f10_uc4_classifier_comparison.png
reports/figures/f10_uc4_best_classifier_confusion_matrix.png
```

## Scientific Limitation

The target is derived from humidex thresholds, and `humidex_c` is also included as a feature. Therefore, this classifier mainly learns to reproduce the rule-based label.

This is useful for testing a future ML risk-classification pipeline, but it should not be presented as an independently validated health-risk model.

## Suivi PM Action

```text
Entraînement et comparaison de modèles de classification pour prédire les niveaux de risque définis par les règles.
```
