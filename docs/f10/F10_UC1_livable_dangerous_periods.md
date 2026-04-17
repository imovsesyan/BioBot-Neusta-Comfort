# F10-UC1 Livable and Dangerous Periods

## Objective

Determine whether periods are livable or not livable using the project livability score, while keeping humidex thresholds as a separate heat-danger layer.

## Input Files

```text
data/processed/meteo_france_1h_clean.csv
data/processed/neusta_15min_clean.csv
reports/tables/f9_uc7_livability_test_predictions.csv
```

## Implementation

The risk rules are implemented in:

```text
src/biobot/risk/rules.py
scripts/f10_uc1_define_livable_dangerous_periods.py
```

Code examples:

```python
score_status = assign_livability_score_status(df["vivabilite_binary_mean"])
```

```python
prediction_df = add_livability_score_status(prediction_df, score_column="predicted")
```

Humidex danger labels are still generated as a separate safety layer:

```python
risk_labels = assign_humidex_risk_level(df["humidex_c"])
```

## Score-Based Status Rule

For the current Neusta binary score:

| Score range | Status | Meaning |
|---|---|---|
| < 0.5 | `livable` | Formula/model considers the period livable |
| >= 0.5 | `not_livable` | Formula/model considers the period not livable |

This direction is based on observed data behavior. Higher values occur during warmer, less comfortable periods. The meaning of `vivabilite_binary_mean` should still be confirmed with the project owner.

## Humidex Heat-Risk Levels

| Humidex range | Risk level | Meaning |
|---|---|---|
| < 30 | `livable` | Little or no discomfort |
| 30 to 39 | `discomfort` | Some discomfort |
| 40 to 45 | `high_risk` | Great discomfort; reduce exertion |
| > 45 | `dangerous` | Dangerous heat stress conditions |
| > 54 | critical flag | Imminent heat-stress concern |

## Command

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc1_define_livable_dangerous_periods.py
```

## Outputs

```text
data/processed/f10_meteo_risk_labels.csv
data/processed/f10_neusta_risk_labels.csv
reports/tables/f10_uc1_livable_dangerous_summary.json
reports/tables/f10_uc1_risk_period_examples.csv
reports/tables/f10_uc1_livability_score_status_summary.json
reports/tables/f10_uc1_livability_score_examples.csv
reports/figures/f10_uc1_risk_level_distribution.png
reports/figures/f10_uc1_high_risk_timeline.png
reports/figures/f10_uc1_livability_score_status.png
```

## Results

### Score-Based Livability Status

| Source | Livable | Not livable |
|---|---:|---:|
| Neusta actual score | 3,237 | 1,078 |
| F9 predicted test score | 532 | 116 |

### Humidex Heat-Risk Layer

| Source | Livable | Discomfort | High risk | Dangerous |
|---|---:|---:|---:|---:|
| Meteo France | 355,101 | 74,067 | 22,030 | 1,597 |
| Neusta | 4,451 | 0 | 0 | 0 |

Meteo France contains high-risk and dangerous humidex periods. Neusta does not contain high-humidex risk periods after processing.

## Visualization

Recommended figures for the report:

```text
reports/figures/f10_uc1_risk_level_distribution.png
reports/figures/f10_uc1_high_risk_timeline.png
reports/figures/f10_uc1_livability_score_status.png
```

## Limitation

The score threshold depends on the meaning of the Neusta binary score. The current implementation treats higher values as less livable because that matches observed data behavior, but this should be validated with the project owner.

Humidex labels do not include activity level, clothing, age, health condition, radiant heat, or direct human feedback.

## Suivi PM Action

```text
Détermination des plages vivables et non vivables à partir du score de vivabilité prédit, avec ajout d’une couche de risque thermique basée sur l’humidex.
```
