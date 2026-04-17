# Chat Handoff Guide

Use this guide when you want to paste project context into a chat tool and ask for a documentation report, manager summary, or review.

## Best Short Context to Paste

Paste these files first:

1. `README.md`
2. `docs/DATA_STRUCTURE_REPORT.md`
3. `docs/f8/F8_UC3_UC4_manager_summary.md`
4. `docs/f8/F8_UC3_UC4_pipeline.md`
5. `docs/f9/F9_MANAGER_SUMMARY.md`
6. `docs/f9/F9_ML_VS_DL_COMPARISON.md`

This is usually enough for a normal written documentation report.

## If the Chat Also Needs Code Context

Add these files:

1. `scripts/f8_uc3_convert_to_standard_csv.py`
2. `scripts/f8_uc4_clean_impute_normalize_aggregate.py`
3. `src/biobot/data/standardize.py`
4. `src/biobot/data/clean_aggregate.py`
5. `scripts/f9_uc6_humidex_threshold_analysis.py`
6. `scripts/f9_uc7_test_livability_models.py`
7. `src/biobot/modeling/livability_features.py`
8. `src/biobot/modeling/metrics.py`

These explain exactly how the pipeline works.

## If the Chat Needs Results

Add these small report files:

1. `reports/tables/f8_uc3_standardization_summary.json`
2. `reports/tables/f8_uc4_cleaning_summary.json`
3. `reports/tables/f9_uc6_humidex_threshold_summary.json`
4. `reports/tables/f9_uc7_livability_model_results.json`

Do not paste large CSV files.

## Suggested Prompt

```text
I am working on an internship project called BioBot / Neusta.
The current scope is F8-UC3, F8-UC4, and the first F9 livability-score prediction workflow:
- convert raw IoT, Aquacheck, Neusta, and Meteo France data to standardized CSV;
- clean, impute, normalize, and aggregate the datasets.
- define and test baseline models for predicting Neusta `vivabilite_binary_mean`.

Please write a clear documentation report in English for my manager.
Include:
- objective,
- input data sources,
- data structure,
- processing steps,
- cleaning rules,
- generated outputs,
- data quality findings,
- F9 problem definition,
- humidex threshold findings,
- baseline model comparison,
- limitations,
- next steps before advanced machine learning, alerts, or recommendations.

Use the files I pasted below as the source of truth.
Do not invent model results. Mention that recommendation systems and model interpretation are out of scope for now.
```

## What Not to Paste

Do not paste:

- raw datasets,
- generated CSV files from `data/interim/` or `data/processed/`,
- `.venv/`,
- cache files,
- old previous-attempt model code from another folder.

## Task-Specific Documentation Packs

For an F9-only report, paste:

1. `docs/f9/F9_UC2_problem_definition.md`
2. `docs/f9/F9_UC3_model_review.md`
3. `docs/f9/F9_UC6_humidex_thresholds.md`
4. `docs/f9/F9_UC7_model_testing_results.md`
5. `docs/f9/F9_UC8_advanced_models.md`
6. `docs/f9/F9_MANAGER_SUMMARY.md`
7. `reports/tables/f9_uc7_livability_model_results.json`
8. `reports/tables/f9_ml_vs_dl_comparison.json`
