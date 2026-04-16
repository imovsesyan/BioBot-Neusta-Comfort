# Data Directory

This directory stores local data files used by the F8 pipeline.

## Folders

| Folder | Purpose | Commit to GitHub? |
|---|---|---|
| `raw/` | Optional local copy of raw datasets | No |
| `interim/` | F8-UC3 standardized CSV outputs | No |
| `processed/` | F8-UC4 cleaned and aggregated CSV outputs | No |

The generated CSV files are intentionally ignored by git because they are large. Small summary reports in `reports/tables/` can be committed.

## Current Raw Data Path

The current raw reference data is expected at:

```text
/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset
```

You can override it when running F8-UC3:

```bash
python scripts/f8_uc3_convert_to_standard_csv.py --dataset-dir "/path/to/dataset"
```

