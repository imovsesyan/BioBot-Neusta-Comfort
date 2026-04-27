# F10-UC4 Circular Feature Ablation

## The Circular Feature Problem

The F10-UC4 classifier achieves macro F1 = 1.0000 (Random Forest) in its standard configuration. The reason is structural:

- The **target** (`risk_level`) is derived from humidex thresholds: `< 30 = livable`, `30–40 = discomfort`, `40–45 = high_risk`, `> 45 = dangerous`.
- `humidex_c` is also included as a **feature**.

When the label-generating variable is a feature, the classifier only needs to learn the threshold boundaries on that single variable. This is a lookup table, not a generalizable model. The macro F1 = 1.0 result tells us the implementation is correct, but not that ML adds scientific value over the rule itself.

## Ablation Method

Run the classifier twice with the same chronological split:

1. **With humidex_c** (standard): confirms circular behavior (expected F1 ≈ 1.0).
2. **Without humidex_c** (`--no-humidex`): tests whether temperature, dew point, humidity, wind speed, pressure, rain, and calendar features can reproduce the humidex risk zones.

```bash
# Standard (circular, expected perfect)
python scripts/f10_uc4_train_risk_classifier.py \
  --fallback-meteo data/processed/meteo_france_1h_clean.csv.gz

# Ablation (honest — no direct humidex)
python scripts/f10_uc4_train_risk_classifier.py \
  --fallback-meteo data/processed/meteo_france_1h_clean.csv.gz \
  --results reports/tables/f10_uc4_ablation_no_humidex_results.json \
  --no-humidex
```

## Results

### With humidex_c (standard, circular)

| Model | Macro F1 | Balanced Accuracy |
|---|---:|---:|
| Random Forest | 1.0000 | 1.0000 |
| XGBoost | 0.9891 | 0.9835 |
| Hist Gradient Boosting | 0.9859 | 0.9862 |
| Logistic Regression (balanced) | 0.9458 | 0.9885 |
| Most Frequent Baseline | 0.2308 | 0.2500 |

### Without humidex_c (ablation, honest)

| Model | Macro F1 | Balanced Accuracy |
|---|---:|---:|
| Hist Gradient Boosting | 0.9770 | 0.9905 |
| XGBoost | 0.9695 | 0.9786 |
| Random Forest | 0.9681 | 0.9692 |
| Logistic Regression (balanced) | 0.9133 | 0.9821 |
| Most Frequent Baseline | 0.2308 | 0.2500 |

## Interpretation

**Random Forest** drops from F1 = 1.0000 → 0.9681 without humidex, a significant drop confirming the circular feature was driving perfect performance.

**Gradient Boosting and XGBoost** remain very high (0.97+) even without humidex. This is physically expected: humidex is computed from temperature and dew point, so these variables encode nearly the same information as humidex. A tree model with enough depth can reconstruct the humidex risk zones from temperature + humidity + dew point alone.

**Scientific conclusion**: The humidex risk zones are learnable from raw meteorological variables at 97%+ macro F1. This means the ML pipeline has genuine value — it can classify risk periods even when humidex is not explicitly available. However, the result is still classification of a rule-derived label, not an independently validated health-risk outcome.

## Why Deep Learning Is Deferred

1. **Dataset adequacy**: Classical ML with class balancing achieves 97%+ macro F1 on this 452,795-row dataset. Adding a deep learning layer does not improve scientific validity or generalization.
2. **Target nature**: The target is rule-derived humidex labels, not independent medical or human-comfort annotations. Deep models optimized on this target would learn the same rule — just less efficiently than gradient boosting.
3. **Justified path to DL**: Deep learning becomes the right tool when the target is (a) independently labeled human comfort or health data, (b) multimodal (combining IoT, Aquacheck, Meteo France), and (c) the dataset is large enough to prevent overfitting. None of those conditions currently hold.
