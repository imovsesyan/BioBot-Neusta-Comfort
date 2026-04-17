# F9 Manager Summary

## Scope

The current F9 work focuses only on livability score prediction.

Out of scope for now:

- recommendation system,
- model interpretation,
- alert generation,
- full risk-classification engine.

## Completed Tasks

| Use case | Status | Output |
|---|---|---|
| F9-UC2 | Complete | Problem definition and key challenges |
| F9-UC3 | Complete | Comparative model review |
| F9-UC6 | Complete | Humidex threshold analysis |
| F9-UC7 | Complete | Baseline model testing |
| F9-UC8 | Initial test complete | CNN-LSTM smoke test |

## Main Decision

The first prediction target is:

```text
Neusta vivabilite_binary_mean
```

This is a 0 to 1 livability score from the processed Neusta dataset.

## Main Result

Tree-based tabular models perform best on the current dataset.

| Model | Family | Test MAE | Test RMSE | Test R2 |
|---|---|---:|---:|---:|
| Random Forest | Classical ML | 0.0033 | 0.0218 | 0.9967 |
| Histogram Gradient Boosting | Classical ML | 0.0033 | 0.0188 | 0.9975 |
| Validation-weighted tree blend | Ensemble blend | 0.0034 | 0.0201 | 0.9972 |
| Equal-weight tree blend | Ensemble blend | 0.0034 | 0.0200 | 0.9972 |
| XGBoost | Classical ML | 0.0036 | 0.0210 | 0.9969 |
| Ridge Regression | Classical ML | 0.1753 | 0.2473 | 0.5748 |
| CNN-LSTM | Deep Learning | 0.3000 | 0.4174 | -0.1879 |
| Mean Baseline | Baseline | 0.3450 | 0.3887 | -0.0503 |

## Scientific Caution

The very high model score probably means the Neusta target is strongly derived from temperature, humidity, and humidex.

Therefore, the correct conclusion is:

> The pipeline can reproduce the current Neusta livability score very well, but we still need to verify whether the score represents real perceived comfort or a rule-based label.

## Humidex Finding

Neusta does not contain high-humidex critical periods after processing. Meteo France does contain high-humidex periods.

This means humidex is useful for future F10 risk detection, but Neusta alone is not enough to train dangerous-heat classification.

## Advanced Model Finding

The CNN-LSTM smoke test did not beat tabular baselines. XGBoost performs similarly to Random Forest and Histogram Gradient Boosting.

The blended tree ensembles are also strong, but they do not beat Random Forest by MAE. This is useful evidence: the individual tree ensembles are already strong enough that a second blending layer gives stability, not a clear accuracy improvement.

Current decision:

- keep tabular models as the main F9 baseline,
- include XGBoost as a strong tabular benchmark,
- keep CNN-LSTM for future experiments,
- postpone Transformer models,
- postpone recommendation and interpretation work.
