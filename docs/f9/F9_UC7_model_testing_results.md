# F9-UC7 Livability Prediction Model Testing

## Objective

Test baseline models for predicting the Neusta livability score.

Target:

```text
vivabilite_binary_mean
```

Input:

```text
data/processed/neusta_15min_clean.csv
```

## Command

```bash
python scripts/f9_uc7_test_livability_models.py
```

## Outputs

```text
reports/tables/f9_uc7_livability_model_results.json
reports/tables/f9_uc7_livability_test_predictions.csv
reports/figures/f9_uc7_model_comparison.png
reports/figures/f9_uc7_test_predictions_timeseries.png
```

XGBoost is included in this benchmark. On macOS, XGBoost may require:

```bash
brew install libomp
```

## Split Strategy

The data is split chronologically:

| Split | Rows |
|---|---:|
| Train | 3,020 |
| Validation | 647 |
| Test | 648 |

Random shuffling is not used.

## Model Results

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Random Forest | 0.0033 | 0.0218 | 0.9967 |
| Histogram Gradient Boosting | 0.0033 | 0.0188 | 0.9975 |
| Validation-weighted tree blend | 0.0034 | 0.0201 | 0.9972 |
| Equal-weight tree blend | 0.0034 | 0.0200 | 0.9972 |
| XGBoost | 0.0036 | 0.0210 | 0.9969 |
| Ridge Regression | 0.1753 | 0.2473 | 0.5748 |
| Mean Baseline | 0.3450 | 0.3887 | -0.0503 |

The best model by test MAE is:

```text
random_forest
```

The best model by test RMSE is:

```text
hist_gradient_boosting
```

## Interpretation

The tree models perform extremely well. This is promising technically, but it is also a warning sign.

The two blended tree ensembles are strong, but they do not improve the best MAE. In this dataset, the individual tree ensemble models are already very close to the target rule.

The most likely explanation is that the Neusta livability label is highly dependent on temperature, humidity, and humidex. Therefore, the model may be learning a label-generation rule rather than independent human comfort.

This does not make the result useless. It means the current model is a strong baseline for reproducing the existing Neusta livability score, but not yet proof of real perceived comfort prediction.

## Next Testing Improvements

Before treating the model as final:

1. Run ablation without humidex.
2. Run a future-horizon prediction task, for example predicting the next 15 or 60 minutes.
3. Validate the meaning of Neusta `vivabilite`.
4. Compare results on Meteo France separately if using the 0 to 7 target.
