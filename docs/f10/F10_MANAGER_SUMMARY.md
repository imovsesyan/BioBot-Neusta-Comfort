# F10 Manager Summary

## Scope

This F10 phase focuses on initial risk detection using two related layers:

- overall livability status from the Neusta/F9 score,
- humidex heat danger as a safety threshold layer.

Included:

- livable and not-livable period definition from score thresholds,
- humidex danger period definition,
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

### Score-Based Livability Status

For the current Neusta binary score:

| Score range | Status | Meaning |
|---|---|---|
| < 0.5 | `livable` | Formula/model considers the period livable |
| >= 0.5 | `not_livable` | Formula/model considers the period not livable |

This direction is based on observed data behavior: higher score values occur during warmer, less comfortable periods.

### Humidex Heat-Risk Layer

| Humidex range | Risk level | Meaning |
|---|---|---|
| < 30 | `livable` | Little or no discomfort |
| 30 to 39 | `discomfort` | Some discomfort |
| 40 to 45 | `high_risk` | Great discomfort; reduce exertion |
| > 45 | `dangerous` | Dangerous heat stress conditions |
| > 54 | critical alert flag | Immediate safety response required |

## Main Results

F10-UC1 showed that Meteo France contains risk periods, while Neusta does not contain high-humidex risk periods after processing.

Score-based F10-UC1 result:

| Source | Livable | Not livable |
|---|---:|---:|
| Neusta actual score | 3,237 | 1,078 |
| F9 predicted test score | 532 | 116 |

Humidex heat-risk result:

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

The score-based livability status depends on the meaning of `vivabilite_binary_mean`. The current interpretation should be confirmed with the project owner.

The classifier is learning labels created from humidex thresholds. Therefore, the perfect classification result means the model reproduced the rule successfully. It does not prove independent real-world medical risk prediction.

## Suivi PM Actions

| Task | Action |
|---|---|
| F10-UC1 | Détermination des plages vivables et non vivables à partir du score de vivabilité prédit, avec ajout d’une couche de risque thermique basée sur l’humidex. |
| F10-UC3 | Mise en place d’un système d’alertes simulées basé sur des règles et des seuils d’humidex. |
| F10-UC4 | Entraînement et comparaison de modèles de classification pour prédire les niveaux de risque définis par les règles. |
