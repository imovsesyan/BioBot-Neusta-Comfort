# F9-UC8 Advanced Models

## Objective

Identify and test advanced models for livability prediction.

Current scope:

- advanced prediction models only,
- no recommendation system,
- no model interpretation module.

## Implemented Advanced Test

The repository includes an optional CNN-LSTM/LSTM script:

```bash
python -m pip install -r requirements-advanced.txt
python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8 --window-size 16
```

Output:

```text
reports/tables/f9_uc8_sequence_model_results.json
reports/tables/f9_uc8_sequence_test_predictions.csv
```

## Current CNN-LSTM Smoke Test Result

| Model | Test MAE | Test RMSE | Test R2 |
|---|---:|---:|---:|
| CNN-LSTM | 0.3000 | 0.4174 | -0.1879 |

Interpretation:

The CNN-LSTM smoke test does not outperform the tabular models. With the current data and target, advanced deep learning is not justified as the main model yet.

## Why Advanced Models Still Matter Later

The research documents support CNN-LSTM and Transformer models when the project has:

- larger time-series datasets,
- richer sensor history,
- true temporal forecasting,
- multiple modalities,
- validated comfort labels.

## Recommended Advanced Model Order

1. LSTM.
2. CNN-LSTM.
3. CNN-LSTM plus residual tree model.
4. Transformer only when data becomes truly multimodal and large enough.

## Current Decision

For the current project state:

- keep Random Forest / gradient boosting as the practical F9 baseline,
- keep CNN-LSTM as an experimental future path,
- do not implement Transformer yet,
- do not implement recommendations yet.
