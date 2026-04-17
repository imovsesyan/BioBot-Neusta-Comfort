# F9-UC2 Problem Definition and Key Challenges

## Objective

Define the machine learning problem for the next project phase.

Current scope:

- Predict a livability score.
- Use Neusta as the first labeled local dataset.
- Do not build recommendations yet.
- Do not add model interpretation yet.

## Prediction Target

The first F9 target is:

```text
vivabilite_binary_mean
```

Source:

```text
data/processed/neusta_15min_clean.csv
```

Meaning:

`vivabilite_binary_mean` is the average Neusta binary livability label inside a 15-minute interval. It is a score between 0 and 1:

- `0` means the interval is fully non-livable according to the original label.
- `1` means the interval is fully livable according to the original label.
- values between 0 and 1 can occur when the 15-minute interval contains mixed raw labels.

## First ML Problem Statement

Given the cleaned Neusta environmental variables at or before time `t`, predict the 15-minute livability score:

```text
y_t = vivabilite_binary_mean_t
```

where:

```text
0 <= y_t <= 1
```

This is treated as a regression problem first.

## Input Features

The first model uses:

- temperature,
- relative humidity,
- humidex,
- record count,
- time features,
- lagged features,
- rolling mean and rolling standard deviation features.

The first model does not use recommendations, personal preferences, or model explanation features.

## Current Dataset Size

From the processed Neusta table:

| Item | Value |
|---|---:|
| Aggregated rows | 5,679 |
| Rows with livability target | 4,315 |
| Target scale | 0 to 1 |
| Split strategy | Chronological train/validation/test |

## Key Challenges

### 1. Target Validity

The current Neusta target may be derived from temperature, humidity, or humidex rules rather than independent human feedback. This means very strong model scores may indicate that the model learned the rule used to generate the target, not true human comfort.

This is the most important scientific limitation of the current F9 work.

### 2. Temporal Leakage Risk

Random shuffling is not appropriate for this time-series dataset because it can leak future patterns into the training data. The current scripts therefore use chronological splitting.

### 3. Limited Target Rows

Only 4,315 aggregated intervals currently have a usable target. This is enough for baseline models but small for deep learning.

### 4. Sparse PM Variables

The Neusta PM columns are sparse. They are not reliable primary predictors yet.

### 5. Different Target Definitions Across Sources

Neusta and Meteo France do not use the same livability target:

| Source | Target | Scale |
|---|---|---|
| Neusta | `vivabilite_binary_mean` | 0 to 1 |
| Meteo France | `vivabilite_score_meteo` | 0 to 7 |

These targets must not be merged without a clear scientific definition.

## Current Decision

For the first F9 implementation, use Neusta only and predict:

```text
vivabilite_binary_mean
```

This gives a clean and explainable first ML task.

