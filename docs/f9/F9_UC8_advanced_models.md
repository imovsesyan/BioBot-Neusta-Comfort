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

## Why Transformer Is Deferred

Three specific reasons apply to this project:

1. **Dataset size**: The Neusta processed data has ~4,315 usable rows (15-minute intervals). Transformers require substantially more data to outperform gradient-boosted trees and to avoid overfitting their attention mechanisms.
2. **Target validity**: `vivabilite_binary_mean` may be formula-derived from humidex (confirmed by the F9-UC7 humidex ablation — see `F9_UC7_ablation.md`). Improving model architecture on a formula-reproduction target does not improve scientific validity.
3. **Current evidence**: CNN-LSTM underperforms Ridge Regression on this dataset (MAE 0.30 vs 0.175), consistent with the target-leakage warning. A Transformer would face the same structural problem.

## Fair Comparison Without Humidex

The current CNN-LSTM result (R² = -0.19) and the tabular baselines (R² = 0.997 for tree models) are not a fair comparison because tabular models have direct access to `humidex_c` while the LSTM must infer it from a temporal window.

To compare fairly, run both scripts with `--no-humidex`:

```bash
python scripts/f9_uc7_test_livability_models.py --input data/processed/neusta_15min_clean.csv.gz --no-humidex
python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8 --no-humidex
```

The F9-UC7 ablation without humidex gives best-model R² ≈ 0.88 (see `F9_UC7_ablation.md`). A fair CNN-LSTM comparison should be run on the same feature set to determine whether temporal context adds value beyond temperature and humidity alone.
