# F10 Manager Summary

## Scope

This F10 phase focuses on initial risk detection using humidex thresholds.

Included:

- livable and dangerous period definition,
- rule-based alert generation,
- ML classification of rule-derived risk labels.

Out of scope:

- production notification delivery,
- medical-risk validation,
- recommendation system,
- deep learning risk model.

## Completed Tasks

| Use case | Status | Main output |
|---|---|---|
| F10-UC1 | Complete | Livable, discomfort, high-risk, and dangerous humidex labels |
| F10-UC3 | Complete | Simulated rule-based alert table |
| F10-UC4 | Complete | ML classifier comparison for risk labels |

## Risk Rules

| Humidex range | Risk level | Meaning |
|---|---|---|
| < 30 | `livable` | Little or no discomfort |
| 30 to 39 | `discomfort` | Some discomfort |
| 40 to 45 | `high_risk` | Great discomfort; reduce exertion |
| > 45 | `dangerous` | Dangerous heat stress conditions |
| > 54 | critical alert flag | Immediate safety response required |

## Main Results

F10-UC1 showed that Meteo France contains risk periods, while Neusta does not contain high-humidex risk periods after processing.

| Source | Livable | Discomfort | High risk | Dangerous |
|---|---:|---:|---:|---:|
| Meteo France | 355,101 | 74,067 | 22,030 | 1,597 |
| Neusta | 4,451 | 0 | 0 | 0 |

F10-UC3 generated:

| Alert severity | Count |
|---|---:|
| Info | 74,067 |
| Warning | 22,030 |
| Danger | 1,594 |
| Critical | 3 |

F10-UC4 best classifier:

```text
random_forest_classifier
```

It reached macro F1 = 1.0000 on the rule-derived Meteo risk labels.

## Scientific Caution

The classifier is learning labels created from humidex thresholds. Therefore, the perfect classification result means the model reproduced the rule successfully. It does not prove independent real-world medical risk prediction.

## Suivi PM Actions

| Task | Action |
|---|---|
| F10-UC1 | Détermination des plages vivables et dangereuses à partir des seuils d’humidex et génération des labels de risque. |
| F10-UC3 | Mise en place d’un système d’alertes simulées basé sur des règles et des seuils d’humidex. |
| F10-UC4 | Entraînement et comparaison de modèles de classification pour prédire les niveaux de risque définis par les règles. |
