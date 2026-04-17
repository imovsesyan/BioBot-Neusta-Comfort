# F10-UC3 Rule-Based Alerts

## Objective

Trigger simulated alerts using threshold rules derived from humidex risk labels.

## Input Files

```text
data/processed/f10_meteo_risk_labels.csv
```

If this file does not exist, the script can recreate labels from:

```text
data/processed/meteo_france_1h_clean.csv
```

## Implementation

The alert rules are implemented in:

```text
src/biobot/risk/rules.py
scripts/f10_uc3_generate_rule_alerts.py
```

Code examples:

```python
alerts = create_rule_alerts(risk_df)
```

```python
alerts["alert_level"] = alerts["risk_level"].where(
    ~alerts["is_critical_humidex"],
    "critical",
)
```

## Alert Rules

| Risk level | Alert severity | Alert type |
|---|---|---|
| `discomfort` | `info` | `thermal_discomfort` |
| `high_risk` | `warning` | `heat_stress_warning` |
| `dangerous` | `danger` | `dangerous_heat_alert` |
| humidex > 54 | `critical` | `critical_heat_alert` |

## Command

```bash
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc3_generate_rule_alerts.py
```

## Outputs

```text
data/processed/f10_meteo_rule_alerts.csv
reports/tables/f10_uc3_rule_alert_summary.json
reports/tables/f10_uc3_alert_examples.csv
reports/figures/f10_uc3_alert_severity_distribution.png
```

## Results

| Alert severity | Count |
|---|---:|
| Info | 74,067 |
| Warning | 22,030 |
| Danger | 1,594 |
| Critical | 3 |

Total simulated alerts:

```text
97,694
```

## Visualization

Recommended figure for the report:

```text
reports/figures/f10_uc3_alert_severity_distribution.png
```

## Limitation

This script creates simulated alert rows. It does not send emails, notifications, SMS, or dashboard alerts yet.

## Suivi PM Action

```text
Mise en place d’un système d’alertes simulées basé sur des règles et des seuils d’humidex.
```
