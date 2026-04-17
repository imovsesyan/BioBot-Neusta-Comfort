# F10-UC1 Livable and Dangerous Periods

## Objective

Determine livable and dangerous environmental periods using humidex thresholds.

## Input Files

```text
data/processed/meteo_france_1h_clean.csv
data/processed/neusta_15min_clean.csv
```

## Implementation

The risk rules are implemented in:

```text
src/biobot/risk/rules.py
scripts/f10_uc1_define_livable_dangerous_periods.py
```

Code examples:

```python
risk_labels = assign_humidex_risk_level(df["humidex_c"])
```

```python
meteo_df = add_risk_labels(meteo_df)
```

## Risk Levels

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
reports/figures/f10_uc1_risk_level_distribution.png
reports/figures/f10_uc1_high_risk_timeline.png
```

## Results

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
```

## Limitation

These labels are rule-based humidex labels. They do not include activity level, clothing, age, health condition, radiant heat, or direct human feedback.

## Suivi PM Action

```text
Détermination des plages vivables et dangereuses à partir des seuils d’humidex et génération des labels de risque.
```
