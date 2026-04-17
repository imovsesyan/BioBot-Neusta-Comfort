# F9 Ensemble Modeling

## Objective

Test whether combining strong tabular models improves livability-score prediction.

## Ensemble Methods Tested

Two prediction-level blends were added after training the individual tree models:

| Ensemble | Members | Weighting |
|---|---|---|
| Equal-weight tree blend | Random Forest, Histogram Gradient Boosting, XGBoost | Each model gets one third |
| Validation-weighted tree blend | Random Forest, Histogram Gradient Boosting, XGBoost | Lower validation MAE gets higher weight |

The validation-weighted blend used these weights:

| Member model | Weight |
|---|---:|
| Random Forest | 0.3902 |
| Histogram Gradient Boosting | 0.2998 |
| XGBoost | 0.3100 |

## Results

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| Random Forest | 0.0033 | 0.0218 | 0.9967 |
| Histogram Gradient Boosting | 0.0033 | 0.0188 | 0.9975 |
| Validation-weighted tree blend | 0.0034 | 0.0201 | 0.9972 |
| Equal-weight tree blend | 0.0034 | 0.0200 | 0.9972 |
| XGBoost | 0.0036 | 0.0210 | 0.9969 |

## Interpretation

The blended ensembles are very strong, but they do not beat the best individual model by MAE.

This is not a failure. It means the strongest tree models are probably learning almost the same signal from temperature, humidity, and humidex. When models make very similar predictions, blending usually improves stability more than accuracy.

## Current Decision

For the current project stage:

- use Random Forest as the best model by MAE,
- keep Histogram Gradient Boosting as the best model by RMSE and R2,
- keep XGBoost as an additional strong benchmark,
- keep blended tree ensembles as a robustness check,
- do not switch to CNN-LSTM for the current tabular target.

## Next Improvement

The next scientifically useful test is not a larger ensemble. It is an ablation experiment without `humidex_c`.

If performance stays high without humidex, the model may be learning broader environmental patterns. If performance drops strongly, the current target is probably very dependent on humidex.
