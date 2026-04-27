# F9-UC7 Humidex Ablation Study

## Scientific Question

The F9-UC7 baseline results showed tree models achieving R² ≈ 0.997 on `vivabilite_binary_mean`. CLAUDE.md flags this as a scientific warning: the target may be formula-derived from the same environmental variables used as features, specifically `humidex_c`.

This ablation answers: **is R² = 0.997 real comfort prediction or formula reproduction via humidex?**

## Method

Run the full F9-UC7 model comparison twice with the same chronological split:

1. **With humidex** (standard run): all features including `humidex_c`, `humidex_c_norm`, and all lag/rolling variants.
2. **Without humidex** (`--no-humidex` flag): all humidex-derived columns removed. Remaining features: temperature, humidity, record count, time features, and their lag/rolling variants.

Run command:
```bash
# Baseline (with humidex)
python scripts/f9_uc7_test_livability_models.py \
  --input data/processed/neusta_15min_clean.csv.gz

# Ablation (without humidex)
python scripts/f9_uc7_test_livability_models.py \
  --input data/processed/neusta_15min_clean.csv.gz \
  --results reports/tables/f9_uc7_ablation_no_humidex_results.json \
  --no-humidex
```

## Results

### With humidex_c (standard)

| Model | Test MAE | Test R² |
|---|---:|---:|
| Hist Gradient Boosting | 0.0033 | 0.9975 |
| Random Forest | 0.0033 | 0.9967 |
| XGBoost | 0.0036 | 0.9969 |
| Tree blend (validation-weighted) | 0.0034 | 0.9972 |
| Ridge Regression | 0.1753 | 0.5748 |
| Mean Baseline | 0.3450 | -0.0503 |

### Without humidex_c (ablation)

| Model | Test MAE | Test R² |
|---|---:|---:|
| Hist Gradient Boosting | 0.0320 | 0.8809 |
| Random Forest | 0.0369 | 0.8871 |
| XGBoost | 0.0479 | 0.9003 |
| Tree blend (validation-weighted) | 0.0366 | 0.8934 |
| Ridge Regression | 0.2293 | 0.4273 |
| Mean Baseline | 0.3450 | -0.0503 |

## Interpretation

The R² drop from **0.997 → 0.881** (Random Forest) confirms that `humidex_c` is the primary driver of the near-perfect tree model scores. When humidex is removed:

- **Tree models**: R² drops by ~11 percentage points (0.997 → ~0.88–0.90). MAE rises tenfold (0.003 → 0.032–0.048). This is a large degradation — the models were largely reproducing the humidex-based formula.
- **Ridge Regression**: R² drops from 0.575 → 0.427. The linear model loses less because it was already limited in capturing the nonlinear rule.
- **Mean Baseline**: unchanged at R² = -0.050, confirming it never used any features.

The remaining R² ≈ 0.88–0.90 without humidex is still meaningful — temperature, humidity, and their temporal lags do carry real predictive signal. However, this is not independent of humidex either, since humidex is itself derived from temperature and humidity. The target `vivabilite_binary_mean` is highly correlated with the input environmental features by construction.

**Conclusion**: The R² = 0.997 result is predominantly formula reproduction, not evidence of real independent comfort prediction. The ablation reduces but does not eliminate predictability, because the underlying physical variables (temperature, humidity) are themselves correlated with the label-generating process.

## Recommended Next Steps

1. Validate the meaning of `vivabilite_binary_mean` with the project owner to confirm whether it is a human-annotated label or a formula output.
2. Use `walk_forward_splits()` (now available in `livability_features.py`) as the primary evaluation method to get uncertainty bounds across multiple temporal windows.
3. If the target is confirmed formula-derived, the appropriate F9 model is a formula approximator — useful for imputation and interpolation but not for inferring real human comfort.

## Walk-Forward Cross-Validation

`walk_forward_splits(df, n_splits=5, min_train_fraction=0.40)` is now implemented in `src/biobot/modeling/livability_features.py`. It produces expanding-window train/test pairs — a more reliable uncertainty estimate than a single chronological split on 4,315 rows.

Future evaluation runs should report mean ± std of MAE and R² across folds, both with and without humidex, to distinguish real predictive performance from variance due to the train/test boundary location.
