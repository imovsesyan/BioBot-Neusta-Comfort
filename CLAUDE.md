# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BioBot / Neusta Comfort Prediction — a data science and ML pipeline for indoor environmental comfort and livability prediction. It processes IoT sensors, Meteo France weather, Neusta environmental data, and Aquacheck soil moisture through a three-phase pipeline: F8 (data preparation), F9 (livability prediction), and F10 (heat-risk detection and alerts).

Raw data lives outside the repo at `/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset`. Compressed processed datasets are committed under `data/processed/*.csv.gz` and can be read directly with `pd.read_csv(path)`.

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
# For CNN-LSTM experiments only:
pip install -r requirements-advanced.txt
```

## Commands

**Run tests:**
```bash
pytest
# Single test file:
pytest tests/test_clean_aggregate.py
# Single test:
pytest tests/test_clean_aggregate.py::test_normalize_iot_sensor_id_keeps_14_digits
```

**Lint (ruff, line length 100):**
```bash
ruff check src/ scripts/ tests/
ruff format src/ scripts/ tests/
```

**Run pipeline scripts** (matplotlib needs a cache dir on some systems):
```bash
python scripts/f8_uc3_convert_to_standard_csv.py
python scripts/f8_uc4_clean_impute_normalize_aggregate.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f8_uc4_make_quality_figures.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_uc6_humidex_threshold_analysis.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_uc7_test_livability_models.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f9_compare_ml_dl_models.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc1_define_livable_dangerous_periods.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc3_generate_rule_alerts.py
MPLCONFIGDIR=.cache/matplotlib python scripts/f10_uc4_train_risk_classifier.py
# Advanced sequence model (requires requirements-advanced.txt):
python scripts/f9_uc8_train_sequence_model.py --model cnn_lstm --epochs 8
```

## Architecture

### Package: `src/biobot/`

**`data/standardize.py`** — F8-UC3. Converts each raw source (IoT JSON, Aquacheck JSON, Neusta CSV, Meteo France CSV) into a consistent schema. Key behaviors: naive timestamps are treated as `Europe/Paris` and converted to UTC; invalid floats become nulls; every row retains `source_file`, `source_row`, and `timestamp_assumption` for provenance. The four datasets are never merged here.

**`data/clean_aggregate.py`** — F8-UC4. Takes standardized CSVs and applies range-rule nulling → 15-min/1h aggregation by sensor or station → short-gap interpolation (limit: 4 steps for IoT/Aquacheck/Neusta, 2 for Meteo) → min-max normalization. Imputed values are flagged with `{column}_was_imputed` boolean columns. Meteo France aggregates at 1h instead of 15 min. Neusta `vivabilite_binary` is preserved as both `vivabilite_binary_mean` (interval mean) and `vivabilite_binary_mode` (rounded label). The `RANGES` dict in this file controls all physical validity bounds.

**`modeling/livability_features.py`** — F9. Prepares model-ready tabular features from the Neusta clean CSV. Adds cyclic time features (sin/cos for hour, day-of-week, month), lag features (1, 4, 16 steps back), and rolling mean/std (windows 4 and 16). Splits are always chronological (70% train / 15% validation / 15% test). Provides `make_sequence_arrays()` for sliding-window CNN-LSTM experiments.

**`modeling/metrics.py`** — MAE, RMSE, R² for regression evaluation.

**`risk/rules.py`** — F10. Two separate risk dimensions:
1. **Humidex heat risk** (`assign_humidex_risk_level`): `< 30` livable, `30–40` discomfort, `40–45` high_risk, `> 45` dangerous, `> 54` critical. Used for rule-based alerts via `create_rule_alerts()`.
2. **Score-based livability status** (`assign_livability_score_status`): `vivabilite_binary_mean >= 0.5` → `not_livable` (higher score correlates with discomfort in the Neusta data — pending confirmation with project owner).

### Data flow

```
raw data (external)
  └─ F8-UC3 (standardize.py)  →  data/interim/*_standardized.csv
       └─ F8-UC4 (clean_aggregate.py)  →  data/processed/*_clean.csv[.gz]
            ├─ F9 (livability_features.py + scripts)  →  reports/
            └─ F10 (risk/rules.py + scripts)  →  reports/
```

### Key scientific constraints

- The four data sources (IoT, Aquacheck, Neusta, Meteo France) are **not merged** — their spatial/temporal alignment has not been validated.
- `vivabilite_binary_mean` (Neusta) and `vivabilite_score_meteo` (Meteo France, 0–7 scale) are **different targets** and must not be mixed.
- Very high tree model scores on F9 (R² ≈ 0.997) are a **scientific warning**: `vivabilite_binary_mean` may be formula-derived from the same environmental variables used as features (target leakage risk). An ablation without `humidex_c` is a recommended next step.
- F10-UC4 classifies rule-derived humidex labels, not independently labeled medical events.

### Output artifacts

Scripts write to `reports/tables/` (JSON summaries, CSV prediction examples) and `reports/figures/` (PNG plots). These are committed. Large intermediate CSVs (`data/interim/`, `data/processed/*.csv` uncompressed) are gitignored; only `.csv.gz` files are committed.
