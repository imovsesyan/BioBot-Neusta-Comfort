# Chat Handoff Guide

Use this guide when you want to paste project context into a chat tool and ask for a documentation report, manager summary, or review.

## Best Short Context to Paste

Paste these files first:

1. `README.md`
2. `docs/DATA_STRUCTURE_REPORT.md`
3. `docs/f8/F8_UC3_UC4_manager_summary.md`
4. `docs/f8/F8_UC3_UC4_pipeline.md`

This is usually enough for a normal written documentation report.

## If the Chat Also Needs Code Context

Add these files:

1. `scripts/f8_uc3_convert_to_standard_csv.py`
2. `scripts/f8_uc4_clean_impute_normalize_aggregate.py`
3. `src/biobot/data/standardize.py`
4. `src/biobot/data/clean_aggregate.py`

These explain exactly how the pipeline works.

## If the Chat Needs Results

Add these small report files:

1. `reports/tables/f8_uc3_standardization_summary.json`
2. `reports/tables/f8_uc4_cleaning_summary.json`

Do not paste large CSV files.

## Suggested Prompt

```text
I am working on an internship project called BioBot / Neusta.
The current scope is F8-UC3 and F8-UC4 only:
- convert raw IoT, Aquacheck, Neusta, and Meteo France data to standardized CSV;
- clean, impute, normalize, and aggregate the datasets.

Please write a clear documentation report in English for my manager.
Include:
- objective,
- input data sources,
- data structure,
- processing steps,
- cleaning rules,
- generated outputs,
- data quality findings,
- limitations,
- next steps before machine learning.

Use the files I pasted below as the source of truth.
Do not invent model results.
```

## What Not to Paste

Do not paste:

- raw datasets,
- generated CSV files from `data/interim/` or `data/processed/`,
- `.venv/`,
- cache files,
- old previous-attempt model code from another folder.

