# F9 ML vs Deep Learning Comparison

## Objective

Compare the current classical machine learning models with XGBoost and the CNN-LSTM advanced model.

The question is:

```text
Are classical machine learning models better than deep learning for the current livability-score prediction task?
```

## Compared Models

| Model | Family | Role |
|---|---|---|
| Mean Baseline | Classical ML baseline | Sanity check |
| Ridge Regression | Classical ML | Simple linear baseline |
| Random Forest | Classical ML | Nonlinear tree ensemble |
| Histogram Gradient Boosting | Classical ML | Boosted tree ensemble |
| XGBoost | Classical ML | Strong boosted tree benchmark |
| Equal-weight tree blend | Ensemble blend | Average of Random Forest, Histogram Gradient Boosting, and XGBoost |
| Validation-weighted tree blend | Ensemble blend | Blend weighted by validation MAE |
| CNN-LSTM | Deep learning | Sequence model experiment |

## Commands

First run the tabular models:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_uc7_test_livability_models.py
```

Then compare ML and deep learning:

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_compare_ml_dl_models.py
```

The CNN-LSTM result comes from:

```bash
python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8 --window-size 16
```

## Outputs

```text
reports/tables/f9_ml_vs_dl_comparison.json
reports/tables/f9_ml_vs_dl_comparison.csv
reports/figures/f9_ml_vs_dl_comparison.png
```

## Current Result

| Model | Family | Test MAE | Test RMSE | Test R2 |
|---|---|---:|---:|---:|
| Random Forest | Classical ML | 0.0033 | 0.0218 | 0.9967 |
| Histogram Gradient Boosting | Classical ML | 0.0033 | 0.0188 | 0.9975 |
| Validation-weighted tree blend | Ensemble blend | 0.0034 | 0.0201 | 0.9972 |
| Equal-weight tree blend | Ensemble blend | 0.0034 | 0.0200 | 0.9972 |
| XGBoost | Classical ML | 0.0036 | 0.0210 | 0.9969 |
| Ridge Regression | Classical ML | 0.1753 | 0.2473 | 0.5748 |
| CNN-LSTM | Deep Learning | 0.3000 | 0.4174 | -0.1879 |
| Mean Baseline | Classical ML baseline | 0.3450 | 0.3887 | -0.0503 |

## Interpretation

For the current Neusta dataset, classical tabular machine learning is better than the tested CNN-LSTM.

The strongest current family is tree-based tabular machine learning:

- Random Forest,
- Histogram Gradient Boosting,
- XGBoost.

The blended ensembles are also strong, but they do not beat the best individual tree models. This suggests that the models are learning almost the same signal.

This is expected because:

- the dataset is small for deep learning,
- the target is tabular and probably formula-derived,
- the strongest signals are current temperature, humidity, and humidex,
- the current task predicts the same interval, not a difficult long-horizon sequence.

This does not mean deep learning is useless for the full project. It means deep learning is not justified as the main model yet.

Deep learning can become useful later if the project has:

- much more data,
- multiple sensors aligned over time,
- a true forecasting task,
- validated human comfort labels,
- richer temporal context.

## Fairness Note

The classical ML models and XGBoost use the same chronological train, validation, and test split.

The CNN-LSTM uses the same chronological split, but it evaluates fewer test rows because each prediction needs a previous time window. A stricter future comparison can align all models on exactly the same test timestamps.

## Decision

For the current project phase:

- keep Random Forest, Histogram Gradient Boosting, and XGBoost as the main model candidates,
- keep blended tree ensembles as a stability check, not as the primary model,
- keep CNN-LSTM as an advanced experiment,
- postpone Transformer models,
- do not build a recommendation system yet,
- do not present deep learning as better unless future data proves it.
