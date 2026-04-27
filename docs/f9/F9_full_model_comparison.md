# F9 — Full Model Comparison Study

**Date**: 2026-04-27
**Scope**: Cross-model comparison of all F9 candidates (Mean baseline, Ridge, Random Forest, HistGradientBoosting, XGBoost, CNN-LSTM) and a documented inspection of the collaborator TFT (biosense360). All numbers are from existing `reports/tables/` artifacts plus a single re-run that captured train/val/test metrics for the classical models and the CNN-LSTM ablation. **No new training was attempted for the TFT.**
**Recommendation**: deliberately omitted from this document — see Step 5 verdicts.

---

## 1. Executive summary

On the Neusta livability target `vivabilite_binary_mean` (0–1, n=4,315 rows, chronological 70/15/15 split), the three tree models (Random Forest, HGB, XGBoost) report test MAE ≈ 0.003 and test R² ≈ 0.997 — values flagged as a leakage warning in CLAUDE.md. The CNN-LSTM (with humidex) reproduces test R² = -0.188 and degrades further to test R² = -0.191 once humidex is removed. The collaborator TFT is trained on a different target (`humidex_c` in °C) on a 1.2 M-row multi-source dataset (80/10/10 split by index) and reports test MAE 0.945 °C @ 15 min, 5.399 °C @ 120 min — these are not numerically comparable to the other models in this study.

---

## 2. Step 1 — Metrics collected

All numbers are taken from existing artifacts (`reports/tables/f9_uc7_livability_model_results.json`, `reports/tables/f9_uc7_ablation_no_humidex_results.json`, `reports/tables/f9_uc8_sequence_model_results.json`, `reports/tables/f9_ml_vs_dl_comparison.json`, `results/evaluation_report.md`, `results/metrics_by_source.csv`) and from a single re-run of the F9 helper script that captures train/val/test simultaneously (`reports/tables/f9_full_model_comparison_metrics.json`). All training MAE/R² for the F9 classical models had to be reconstructed by re-running because the original UC7 JSON only stored validation/test. The re-run uses identical seeds and hyper-parameters and reproduces the published validation/test numbers byte-for-byte.

### 2.1 F9 classical models (target = `vivabilite_binary_mean`)

| Model | Train MAE | Val MAE | Test MAE | Train RMSE | Val RMSE | Test RMSE | Train R² | Val R² | Test R² | humidex_c in features | n_features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:-:|---:|
| mean_baseline | 0.3831 | 0.3784 | 0.3450 | 0.4351 | 0.4296 | 0.3887 | 0.0000 | -0.0009 | -0.0503 | yes (unused) | 36 |
| ridge_regression | 0.1571 | 0.1638 | 0.1753 | 0.2287 | 0.2321 | 0.2473 | 0.7238 | 0.7079 | 0.5748 | yes | 36 |
| random_forest | 0.0031 | 0.0026 | 0.0033 | 0.0179 | 0.0200 | 0.0218 | 0.9983 | 0.9978 | 0.9967 | yes | 36 |
| hist_gradient_boosting | 0.0023 | 0.0033 | 0.0033 | 0.0113 | 0.0192 | 0.0188 | 0.9993 | 0.9980 | 0.9975 | yes | 36 |
| xgboost | 0.0028 | 0.0032 | 0.0036 | 0.0135 | 0.0199 | 0.0210 | 0.9990 | 0.9978 | 0.9969 | yes | 36 |

Split: chronological 3,020 train / 647 val / 648 test rows, total 4,315.
Source artifacts: `reports/tables/f9_uc7_livability_model_results.json` (val/test) and `reports/tables/f9_full_model_comparison_metrics.json` (train + reproduced val/test).
Overfitting flag in code/docs: CLAUDE.md explicitly warns the R² ≈ 0.997 result is *consistent with target leakage*. The F9 ablation doc (`docs/f9/F9_UC7_ablation.md`) confirms tree R² drops to 0.88–0.90 when `humidex_c` is removed, attributing the high baseline R² primarily to `humidex_c`.

### 2.2 CNN-LSTM (target = `vivabilite_binary_mean`)

Architecture (`scripts/f9_uc8_train_sequence_model.py`): Conv1D(32, k=3, causal, ReLU) → Conv1D(16, k=3, causal, ReLU) → LSTM(48) → Dropout(0.2) → Dense(32, ReLU) → Dense(1, sigmoid). Adam(lr=1e-3), MSE loss, batch=32, window=16, max_epochs=8, EarlyStopping(monitor=val_loss, patience=4, restore_best_weights=True).

| Variant | Train MAE | Val MAE | Test MAE | Train RMSE | Val RMSE | Test RMSE | Train R² | Val R² | Test R² | epochs_completed | n_features |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CNN-LSTM (with humidex) | 0.2700 | 0.2871 | 0.3000 | 0.3875 | 0.3927 | 0.4174 | 0.2094 | 0.1773 | -0.1879 | 8 | 36 |
| CNN-LSTM (no humidex)   | 0.4156 | 0.4248 | 0.4073 | 0.4307 | 0.4377 | 0.4179 | 0.0235 | -0.0219 | -0.1909 | 6 | 27 |

Source artifacts: `reports/tables/f9_uc8_sequence_model_results.json` (test only — the original script does not record train/val) and `reports/tables/f9_full_model_comparison_metrics.json` (re-run with train/val capture). Test R² = -0.188 in the with-humidex case matches the previously published number to four decimals. Early stopping was used (patience=4, restore_best_weights=True). Sequence sizes: 3,004 / 631 / 632 windows after 16-step windowing.

### 2.3 TFT (collaborator biosense360, target = `humidex_c` in °C)

Source artifacts: `results/biosense360_train.py`, `results/biosense360_data.py`, `results/evaluation_report.md`, `results/metrics_by_source.csv`.

Architecture: TemporalFusionTransformer (`pytorch-forecasting`) with `hidden_size=128`, `attention_head_size=4`, `dropout=0.1`, `hidden_continuous_size=64`, `output_size=7` (quantiles 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9), `learning_rate=3e-3`, `gradient_clip_val=0.1`, `batch_size=128`, `max_epochs=100`, `EarlyStopping(monitor="val_loss", patience=10)`, `ReduceLROnPlateau(patience=4)`, target normaliser = `TorchNormalizer(method="standard", center=True)`.
Window: `max_encoder_length=32` (8 h of history at 15 min) → `max_prediction_length=8` (2 h horizon). Loss: pinball / `QuantileLoss`.

Dataset: merged Neusta (1 sensor, 15 min), Meteo France (60 stations resampled 3 h → 1 h), IoT (2 sensors, 15 min), Aquacheck (10 sensors, 15 min). ≈ 1.2 M rows after gap-based series splitting (gap > 2 h ⇒ new series) and series-length filtering. Split: 80% train / 10% validation / 10% test by row index after sorting by `(sensor_id, timestamp_utc)`. Early stop fired at epoch 7 (per `docs/f9/F9_etat_de_lart.md` §1 and `docs/f9/F9_etat_de_lart.md` §3.4).

| Horizon | Test MAE (°C) | Test RMSE (°C) | 80% interval coverage | Notes |
|---|---:|---:|---:|---|
| 15 min  | 0.945 | 1.402 | 0.717 | from `evaluation_report.md` and `metrics_by_source.csv` |
| 30 min  | 1.481 | 1.935 | 0.642 |  |
| 60 min  | 2.855 | 3.714 | 0.647 |  |
| 120 min | 5.399 | 7.072 | 0.586 | calibration loss vs. 0.80 nominal |

Per-source @ 120 min:

| Source | MAE (°C) | RMSE (°C) | 80% coverage |
|---|---:|---:|---:|
| meteo_france | 5.574 | 7.196 | 0.579 |
| neusta       | 1.246 | 2.869 | 0.748 |

Train/validation point metrics (MAE/R²) are not present in either the evaluation report or the CSV — the only logged training-time scalar is the QuantileLoss `val_loss`, which `evaluation_report.md` does not record numerically (training stopped at epoch 7 with `val_loss` as the monitor). **Train MAE, val MAE, train R² and test R² in the same units cannot be sourced for TFT from existing artifacts.** This is documented in Step 2 below as a missing-cell row.

### 2.4 Why TFT is not numerically comparable to the F9 models

- **Different target**: the TFT predicts `humidex_c` (°C, roughly -10 to +50) — the F9 models predict `vivabilite_binary_mean` (0–1 interval).
- **Different task**: the TFT does multi-horizon (15/30/60/120 min) point + quantile forecasting; F9 does instantaneous regression.
- **Different data**: the TFT uses ≈ 1.2 M merged multi-source rows; F9 uses 4,315 single-sensor rows.
- **Different split granularity**: TFT uses 80/10/10 by row index on the merged frame; F9 uses 70/15/15 chronological by date on the Neusta frame.
- **Different loss**: TFT optimises a 7-quantile pinball loss; F9 minimises MSE.

A footnote is repeated in the master table to make this comparability gap unambiguous.

---

## 3. Step 2 — Master comparison table

F9 models (target = `vivabilite_binary_mean`, 0–1) ranked by Test MAE, best to worst. TFT listed separately (different target, different scale).

| Rank | Model | Train MAE | Val MAE | Test MAE | Train R² | Test R² | Overfit gap (Test MAE − Train MAE) | Gap as % of Train MAE | Train R² − Test R² | humidex_c in features | Target | Flags |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:-:|---|---|
| 1 | random_forest | 0.0031 | 0.0026 | 0.0033 | 0.9983 | 0.9967 | +0.0002 | +6.5% | 0.0016 | yes | vivabilite_binary_mean | 🔴 leakage suspect (R² ≈ 1.0 per CLAUDE.md) |
| 2 | hist_gradient_boosting | 0.0023 | 0.0033 | 0.0033 | 0.9993 | 0.9975 | +0.0010 | +44% | 0.0018 | yes | vivabilite_binary_mean | 🔴 leakage suspect; 🔴 overfit gap > 15% (per criterion) |
| 3 | xgboost | 0.0028 | 0.0032 | 0.0036 | 0.9990 | 0.9969 | +0.0008 | +29% | 0.0021 | yes | vivabilite_binary_mean | 🔴 leakage suspect; 🔴 overfit gap > 15% |
| 4 | ridge_regression | 0.1571 | 0.1638 | 0.1753 | 0.7238 | 0.5748 | +0.0182 | +11.6% | 0.1490 | yes | vivabilite_binary_mean | 🔴 Train R² − Test R² = 0.149 (> 0.05) |
| 5 | cnn_lstm (with humidex) | 0.2700 | 0.2871 | 0.3000 | 0.2094 | -0.1879 | +0.0300 | +11.1% | 0.3973 | yes | vivabilite_binary_mean | 🔴 Train R² − Test R² = 0.397; test R² < mean baseline |
| 6 | cnn_lstm (no humidex) | 0.4156 | 0.4248 | 0.4073 | 0.0235 | -0.1909 | -0.0083 | -2.0% | 0.2144 | no | vivabilite_binary_mean | underfit; test R² < mean baseline |
| 7 | mean_baseline | 0.3831 | 0.3784 | 0.3450 | 0.0000 | -0.0503 | -0.0381 | -9.9% | 0.0503 | yes (unused) | vivabilite_binary_mean | reference floor |

TFT (separate scale — not ranked):

| Model | Train MAE | Val MAE | Test MAE | Train R² | Test R² | Overfit gap | humidex_c in features | Target | Flags |
|---|:-:|:-:|---:|:-:|:-:|:-:|:-:|---|---|
| TFT 15 min  | n/a* | n/a* | 0.9450 °C | n/a* | n/a* | n/a* | yes (in `UNKNOWN_REALS`) | humidex_c (°C) | early-stopped epoch 7; coverage 0.717 vs. nominal 0.80 |
| TFT 30 min  | n/a* | n/a* | 1.4808 °C | n/a* | n/a* | n/a* | yes | humidex_c (°C) | coverage 0.642 |
| TFT 60 min  | n/a* | n/a* | 2.8551 °C | n/a* | n/a* | n/a* | yes | humidex_c (°C) | coverage 0.647 |
| TFT 120 min | n/a* | n/a* | 5.3986 °C | n/a* | n/a* | n/a* | yes | humidex_c (°C) | coverage 0.586 (under-coverage of 0.80 nominal) |

*Train/val MAE and R² for TFT cannot be sourced from existing artifacts — only `val_loss` (QuantileLoss / pinball) is logged in the training callback and is not surfaced numerically in `evaluation_report.md`. Re-deriving these would require re-training, which is out-of-scope per the task brief (GPU required).

**Footnote on comparability**: the TFT MAE values are in °C on the `humidex_c` target (range roughly -10 to +50); the F9 MAE values are in unitless livability score on `vivabilite_binary_mean` (range 0 to 1). They do not share a scale, a target, a dataset, or a task. They cannot be ranked together.

**Flag legend** (criteria from the brief):
- 🔴 if Test MAE − Train MAE > 15% of Train MAE
- 🔴 if Train R² − Test R² > 0.05
- 🔴 if R² is suspiciously close to 1.0 (leakage warning per CLAUDE.md)

The three tree models all hit the third criterion. HGB and XGBoost also hit the first criterion in *relative* terms because their Train MAE is so close to zero that any small absolute gap explodes as a percentage; the absolute gap is < 0.001 in score units, so this is not classical overfitting in the usual sense — it is an artefact of the leakage-driven near-perfect training fit. The Ridge and CNN-LSTM rows hit the second criterion. Both CNN-LSTM rows have test R² below the mean-baseline floor.

---

## 4. Step 3 — TFT deep inspection

Read directly from `results/biosense360_train.py` and `results/biosense360_data.py`.

### 4.1 Architecture

| Hyperparameter | Value | Source line |
|---|---|---|
| `hidden_size` | 128 | `biosense360_train.py:45` |
| `attention_head_size` | 4 | `biosense360_train.py:46` |
| `max_encoder_length` | 32 (= 8 h at 15 min) | `biosense360_train.py:39` |
| `max_prediction_length` | 8 (= 2 h at 15 min) | `biosense360_train.py:40` |
| `dropout` | 0.1 | `biosense360_train.py:47` |
| `hidden_continuous_size` | 64 | `biosense360_train.py:48` |
| `output_size` | 7 | `biosense360_train.py:49` |
| Quantiles | 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9 | comment on `biosense360_train.py:49` |
| `learning_rate` | 3e-3 | `biosense360_train.py:50` |
| `batch_size` | 128 | `biosense360_train.py:52` |
| `max_epochs` | 100 | `biosense360_train.py:53` |
| `EarlyStopping(patience=...)` | 10 (monitor `val_loss`) | `biosense360_train.py:54, 165` |
| `ReduceLROnPlateau(patience=...)` | 4 | `biosense360_train.py:55, 135` |
| `gradient_clip_val` | 0.1 | `biosense360_train.py:56, 180` |
| Target normaliser | `TorchNormalizer(method="standard", center=True)` | `biosense360_train.py:93` |
| Loss | `QuantileLoss()` (7-quantile pinball) | `biosense360_train.py:133` |

### 4.2 Features

`KNOWN_REALS` (`biosense360_data.py:254`):
`time_idx`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month_sin`, `month_cos`.

`UNKNOWN_REALS` (`biosense360_data.py:259`):
`temperature_c`, `relative_humidity_pct`, `humidex_c`, `temperature_c_lag1`, `temperature_c_lag4`, `temperature_c_roll6_mean`, `temperature_c_diff1`, `relative_humidity_pct_lag1`, `relative_humidity_pct_roll6_mean`, `humidex_c_lag1`, `humidex_c_lag4`, `humidex_c_roll6_mean`, `humidex_c_diff1`, `temperature_c_was_imputed`, `relative_humidity_pct_was_imputed`, `humidex_c_was_imputed`, `soil_moisture_pct`, `wind_speed_mps`, `pressure_pa`, `rain_1h_mm`, `co2_ppm`, `tvoc_ppb`, `src_neusta`, `src_meteo_france`, `src_iot`, `src_aquacheck`.

The target `humidex_c` is itself listed in `UNKNOWN_REALS`, but in `pytorch-forecasting` the target is fed into the model only as past observations within the encoder window (decoder steps see no target). This is the standard design and is not target leakage in the traditional sense — but it means the TFT is in effect predicting future `humidex_c` from past `humidex_c` plus engineered history features, which is the easy direction of the problem. The MAE figures should be read with that caveat.

### 4.3 Split

- 80% train / 10% validation / 10% test, **by row index** after `sort_values(["sensor_id", "timestamp_utc"])` (`biosense360_train.py:79–82`).
- This is *not* a chronological split by absolute date — because the rows are sorted first by `sensor_id`, the first 80% can contain entire sensors that the validation/test sets never see, and one sensor's history can be split mid-stream. The overall effect on temporal leakage depends on the relative row counts per sensor; meteo_france dominates the row count after 1 h resampling.
- Series with fewer than `MIN_SERIES = 45` rows are dropped (`biosense360_data.py:19, 191`).

### 4.4 Training history

- `EarlyStopping(patience=10)` → `stopped at epoch 7` (per `docs/f9/F9_etat_de_lart.md` §1 and §3.4). With patience=10 starting from epoch 1, this means the validation loss did not improve for 10 consecutive epochs and the trainer halted at epoch 7 because there were no more epochs of improvement to wait for — i.e. validation loss plateaued or rose almost immediately. This is consistent with the literature finding (état de l'art §3.2) that TFT on small / single-stream data overfits "within a handful of epochs".
- `val_loss = 0.9794` reported by the collaborator is a `QuantileLoss` (pinball loss across the seven quantiles). It is not directly comparable to MAE or RMSE.

### 4.5 Differences from the F9 split

| Aspect | F9 (UC7 + UC8) | TFT (biosense360) |
|---|---|---|
| Split policy | Chronological by date | 80/10/10 by row index after sorting by sensor and timestamp |
| Train / val / test | 70 / 15 / 15 | 80 / 10 / 10 |
| Dataset | Neusta single sensor, 4,315 rows | 4 sources merged, ≈ 1.2 M rows |
| Target | `vivabilite_binary_mean` (0–1) | `humidex_c` (°C) |
| Loss | MSE | QuantileLoss (pinball, 7 quantiles) |
| Horizon | instantaneous regression | 8-step (2 h) multi-horizon forecast |

### 4.6 Calibration of the 80% interval

Nominal coverage 0.80; observed coverage 0.717 @ 15 min, 0.642 @ 30 min, 0.647 @ 60 min, 0.586 @ 120 min. Coverage is below nominal at every horizon and degrades with horizon. This is the textbook signature of an overconfident quantile head — the model's 80% interval is too narrow. État de l'art §3.2 calls this out explicitly as a small-data TFT failure mode. The 120-min row also has the largest under-coverage (0.586 vs. 0.80, a 21.4-percentage-point miss).

---

## 5. Step 4 — CNN-LSTM ablation: with vs without humidex_c

Both runs use `scripts/f9_uc8_train_sequence_model.py`-equivalent code (the helper script reproduces the architecture and training loop verbatim). Same chronological 70/15/15 split, same window=16, batch=32, LR=1e-3, EarlyStopping(patience=4, restore_best_weights=True).

| Variant | Train MAE | Val MAE | Test MAE | Train RMSE | Val RMSE | Test RMSE | Train R² | Val R² | Test R² | epochs_completed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| WITH humidex_c (default, 36 features) | 0.2700 | 0.2871 | 0.3000 | 0.3875 | 0.3927 | 0.4174 | 0.2094 | 0.1773 | -0.1879 | 8 |
| WITHOUT humidex_c (27 features)        | 0.4156 | 0.4248 | 0.4073 | 0.4307 | 0.4377 | 0.4179 | 0.0235 | -0.0219 | -0.1909 | 6 |

Removing `humidex_c` and its derived columns:
- collapses the train R² from 0.209 to 0.024 (an 88% relative drop) — the CNN-LSTM relies almost entirely on humidex to fit the training set;
- pushes the validation R² below zero (-0.022) — the model can no longer beat the validation mean;
- leaves the test R² essentially unchanged at -0.19 — already negative, the architecture has no useful generalisation signal in either case;
- triggers earlier stopping (6 epochs vs. 8) — the optimiser has nothing left to extract once humidex is gone.

The CNN-LSTM is therefore in the same regime in both ablations: it does not generalise on `vivabilite_binary_mean` at this sample size, and humidex was its only path to beating the training mean.

### 5.1 TFT humidex ablation — not run

**TFT humidex ablation not run — GPU required**. Per the task brief and `commands.txt`, the TFT was trained on an RTX 5070 in roughly 15–20 minutes; the only artefact preserved is the epoch-7 checkpoint and the evaluation CSVs. Re-training on CPU is not realistic, and re-training on a remote GPU is out of scope for this study.

**Predicted impact**, derived from the F10 humidex ablation already on disk and the literature (état de l'art §3, §4, Q1):

- F10 ablation on the rule-derived risk classifier (`reports/tables/f10_uc4_risk_classifier_results.json` vs `reports/tables/f10_uc4_ablation_no_humidex_results.json`):
  - Random Forest macro F1: 1.000 → 0.968 (drop 0.032)
  - HGB macro F1: 0.986 → 0.977 (drop 0.009)
  - XGBoost macro F1: 0.989 → 0.970 (drop 0.020)
  - The "1.00 → 0.67" claim cited in the original task brief does not match the artefact on disk; the actual macro-F1 deltas are an order of magnitude smaller. The F10 task is *highly* humidex-driven (the labels are derived directly from humidex thresholds), so the small drop reflects the rich auxiliary feature set (`temperature_c`, `dew_point_c`, `relative_humidity_pct`, `wind_speed_mps`, `pressure_pa`, `rain_1h_mm`, time features) being able to reconstruct the humidex thresholds from physical inputs.
- État de l'art §4.5 finds humidex is the primary thermal-comfort signal in published studies (XGBoost R² 0.994 on PPD, 0.93 on PMV with humidex among the inputs).
- For the TFT specifically, `humidex_c` is the **target** itself plus four lag/roll/diff variants in `UNKNOWN_REALS`. Removing the lag/roll/diff variants would force the model to forecast humidex purely from temperature, humidity and weather covariates. Given (a) the strong physical relationship `humidex ≈ T + 0.5555·(e − 10)` from `_humidex(T, RH)` in `biosense360_data.py:42`, and (b) the rest of the F10 evidence (auxiliary physical features can rebuild humidex up to about 3% of macro F1), the predicted impact is a *small* increase in MAE (likely +0.1 to +0.5 °C @ 15 min and +0.5 to +1.5 °C @ 120 min), but the architecture should still produce comparable point forecasts. Calibration (interval coverage) might worsen more than point error, because the model loses its strongest history feature.
- Caveat: this is a prediction, not a measurement. It should not be cited as evidence.

---

## 6. Step 5 — Plain language verdict per model

### Mean baseline

- Test MAE: 0.3450 — predictions are off by 0.345 units of `vivabilite_binary_mean` on average, which is 35% of the full 0–1 scale.
- vs Literature benchmark: a constant-mean baseline typically reaches R² 0.0 to 0.2 on noisy environmental sensor streams (Hyndman & Athanasopoulos 2021; état de l'art §4.1). Our R² = -0.05 indicates the train mean is not a good summary of the test distribution — likely a regime shift between the train period and the test period, or a target distribution that itself drifts in time.
- Overfitting: no — Train R² = 0.0000, Test R² = -0.0503. Gap = 0.05.
- Leakage risk: no — by construction.
- Verdict: USE (as a reference floor only).
- One sentence why: a baseline whose only job is to mark the lower bound of useful prediction; everything below it is broken.

### Ridge Regression

- Test MAE: 0.1753 — predictions are off by 0.175 units of `vivabilite_binary_mean` on average, ≈ 18% of full scale.
- vs Literature benchmark: Ridge / linear baselines on thermal comfort regression typically reach R² 0.4–0.7 (état de l'art §4.2). Our 0.5748 is in the middle of this range and is the most literature-consistent result in the entire study.
- Overfitting: yes (mild) — Train R² = 0.7238, Test R² = 0.5748. Gap = 0.149.
- Leakage risk: possible — Ridge benefits from `humidex_c` being in features (the same suspected leakage source as the trees), but its linear capacity caps the leakage exploitation. The train→test R² gap of 0.149 is consistent with both honest non-stationarity and partial leakage exploitation.
- Verdict: INVESTIGATE FURTHER.
- One sentence why: Ridge is the only model whose error is in the literature-plausible range for indoor thermal comfort regression and is therefore the strongest candidate for a leakage-audited rerun (drop humidex_c, add walk-forward CV).

### Random Forest

- Test MAE: 0.0033 — predictions are off by 0.003 units (≈ 0.3% of the 0–1 scale) on average.
- vs Literature benchmark: above any literature value found for indoor / thermal-comfort regression at this sample size (état de l'art §4.3 puts RF R² for PMV-class regression at 0.85–0.99). Our R² = 0.997 is at or above the upper edge for *much larger* datasets.
- Overfitting: classical interpretation says no — Train R² = 0.9983, Test R² = 0.9967, gap = 0.002. But the absolute level of both is so close to 1.0 that the train/test similarity itself is a leakage signature, not a generalisation signature.
- Leakage risk: yes (per CLAUDE.md and `docs/f9/F9_UC7_ablation.md`). Removing `humidex_c` drops test R² from 0.997 to 0.887 — eleven percentage points of "performance" came from the humidex feature alone, which is itself derivable from temperature and humidity, which are also features.
- Verdict: DO NOT USE (as currently built).
- One sentence why: the headline R² is largely formula reproduction of `vivabilite_binary_mean` from `humidex_c`, not independent comfort prediction.

### HistGradientBoosting

- Test MAE: 0.0033 — same as RF.
- vs Literature benchmark: HGB / LightGBM typical R² on building-energy and indoor-T regression is 0.85–0.97 (état de l'art §4.4). Our 0.9975 is above the range.
- Overfitting: by the strict criterion (Test MAE − Train MAE > 15% of Train MAE), yes — Train MAE 0.0023, Test MAE 0.0033, +44%. By the absolute-magnitude reading, no — both errors are in the noise. The 🔴 flag from the criterion fires here but is essentially meaningless because Train MAE is already three thousandths of a unit; what the flag is really telling you is that the leakage is so strong the model fits the training set perfectly and only the leakage-blocking effect of train/test boundary noise creates any gap.
- Leakage risk: yes (same root cause as RF).
- Verdict: DO NOT USE (as currently built).
- One sentence why: identical leakage profile to Random Forest with the additional concern that HGB has the deepest interaction-finding capacity of the three trees and so is the most able to exploit even partial leakage.

### XGBoost

- Test MAE: 0.0036 — same order as RF and HGB.
- vs Literature benchmark: XGBoost typical R² for thermal comfort regression is 0.93–0.994 (état de l'art §4.5). Our 0.9969 is at the top edge.
- Overfitting: same caveat as HGB — strict criterion fires (+29% relative gap) but absolute gap is < 0.001.
- Leakage risk: yes.
- Verdict: DO NOT USE (as currently built).
- One sentence why: same reason as RF and HGB — pending a leakage audit, all three tree models share the same scientific status.

### CNN-LSTM (with humidex)

- Test MAE: 0.3000 — predictions are off by 0.300 units of `vivabilite_binary_mean` on average. Test R² = -0.188 means it is *worse than predicting the training mean*.
- vs Literature benchmark: CNN-LSTM works on thermal forecasting tasks with ≥ 10,000 rows (Hou 2022, 60,133 rows; Elmaz 2021, multi-month single-room). On 4,315 rows the literature has *no precedent for success* (état de l'art §2.3 and Q1).
- Overfitting: yes — Train R² = 0.209, Test R² = -0.188, gap = 0.397 (almost an order of magnitude above the practitioner threshold of 0.10–0.15 for LSTMs).
- Leakage risk: no — the architecture cannot exploit the suspected leakage even if it is there, because it does not converge.
- Verdict: DO NOT USE.
- One sentence why: the model does not generalise on this dataset size; the literature predicts this exactly (état de l'art §2.5).

### CNN-LSTM (without humidex)

- Test MAE: 0.4073 — worse than the mean baseline (0.345).
- vs Literature benchmark: same regime as above; removing humidex makes the underfitting more visible but does not change the diagnosis.
- Overfitting: less train→test gap, but only because train R² also collapsed to 0.024 — the model fits neither set.
- Leakage risk: no.
- Verdict: DO NOT USE.
- One sentence why: confirms the with-humidex run was already in the no-generalisation regime; humidex was just the only feature it could memorise on training data.

### TFT (biosense360, target = humidex_c)

- Test MAE: 0.945 °C @ 15 min, 5.399 °C @ 120 min. The 15-min number is in the ballpark of CNN-LSTM literature (Hou 2022 reports 1.02 °C test MAE on 60,133 hourly rows of outdoor air temperature).
- vs Literature benchmark: 80% interval coverage of 0.717 (15 min) → 0.586 (120 min) is below the nominal 0.80 at every horizon. État de l'art §3.4 names this an "overconfident quantile head" failure mode for small-data TFT.
- Overfitting: cannot be measured from artefacts. Early stop at epoch 7 with patience=10 means the validation loss plateaued or rose almost immediately after the first epoch — the practitioner literature (état de l'art §3.2) calls a < 10-epoch convergence on small TFT data a strong overfitting / instability signal.
- Leakage risk: not in the project-specific sense (no `vivabilite_binary_mean`-style closed-form target; the target `humidex_c` is a physically derived variable that the TFT predicts from past observations and weather covariates). The sensor-mixed split-by-index instead of split-by-date is a *different* concern — it can let the same sensor's nearby timestamps appear in both train and test if a sensor's data spans the index boundary.
- Verdict: INVESTIGATE FURTHER.
- One sentence why: the 120-min coverage gap of 0.214 and the early stop at epoch 7 are textbook small-data TFT failure modes; the per-source breakdown showing Neusta MAE 1.246 °C @ 120 min vs. Meteo France 5.574 °C suggests the architecture is doing real work on at least one source and warrants a closer per-source error analysis before any verdict.

---

## 7. Scientific caveats

### 7.1 Leakage

CLAUDE.md flags tree-model R² ≈ 0.997 on `vivabilite_binary_mean` as a leakage warning. The on-disk F9 ablation (`docs/f9/F9_UC7_ablation.md`) confirms ≈ 11 percentage points of test R² come from `humidex_c` alone. Because `humidex_c` is itself a closed-form function of `temperature_c` and `relative_humidity_pct`, a "no-humidex" ablation is not a full leakage audit — the residual R² ≈ 0.88–0.90 without humidex still includes the formula's *inputs*. A complete audit would require a dataset where `vivabilite_binary_mean` is provided alongside the *original human comfort labels* and the formula provenance is documented; this is an open project task, not a modelling task.

### 7.2 Comparability

The TFT and the F9 models do not share a target, a dataset, a split, a loss, or a task. The master comparison table separates them physically and adds a footnote on every TFT row to make this unambiguous. Any direct numerical comparison between TFT MAE in °C and F9 MAE in livability-score units is meaningless.

### 7.3 Split differences

- F9: chronological 70/15/15 by date on Neusta, 4,315 rows.
- TFT: 80/10/10 by row index on the merged 1.2 M-row dataset, sorted by `(sensor_id, timestamp_utc)` first. This is *not* a chronological-by-date split because sensors with different time spans interleave in the index.
- F10 risk classifier (referenced for the predicted-ablation rationale): 160,000 train / 67,919 val / 67,920 test, chronological.
The three pipelines therefore answer three different questions even when the same word ("test set") is used.

### 7.4 Train metrics for TFT not available

`evaluation_report.md` and `metrics_by_source.csv` only carry test-set MAE/RMSE/coverage by horizon and by source. The training callbacks log `val_loss` (`QuantileLoss`) but no equivalent metric is exported. To produce TFT train-MAE / train-R² in the same units, the model would need to be re-evaluated on the training and validation `TimeSeriesDataSet` objects — this requires loading the saved checkpoint on a GPU, which is out-of-scope per the brief.

### 7.5 Mean-baseline test R² is negative

R² = -0.0503 for the mean baseline indicates the test-set target distribution differs from the training-set target distribution: the test mean is not the train mean. This affects every model that learned the train mean as its centre of mass, including Ridge and (mechanically) the trees. Walk-forward cross-validation, already implemented as `walk_forward_splits()` in `src/biobot/modeling/livability_features.py` per `docs/f9/F9_UC7_ablation.md` §7, would expose this drift more honestly than a single 70/15/15 split.

---

## 8. Source artefacts referenced

- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/CLAUDE.md`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/docs/f9/F9_etat_de_lart.md`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/docs/f9/F9_UC7_model_testing_results.md`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/docs/f9/F9_UC7_ablation.md`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/results/evaluation_report.md`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/results/biosense360_train.py`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/results/biosense360_data.py`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/results/metrics_by_source.csv`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/scripts/f9_uc7_test_livability_models.py`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/scripts/f9_uc8_train_sequence_model.py`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/scripts/f9_full_comparison_helper.py` (created for this study; produces `reports/tables/f9_full_model_comparison_metrics.json`)
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/reports/tables/f9_uc7_livability_model_results.json`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/reports/tables/f9_uc7_ablation_no_humidex_results.json`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/reports/tables/f9_uc8_sequence_model_results.json`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/reports/tables/f9_ml_vs_dl_comparison.json`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/reports/tables/f10_uc4_risk_classifier_results.json`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/reports/tables/f10_uc4_ablation_no_humidex_results.json`
- `/Users/inesamovsesyan/BioBot-Neusta-Comfort/reports/tables/f9_full_model_comparison_metrics.json` (new, produced by this study)

*End of document.*
