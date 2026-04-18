# Data Directory

This directory stores data files used by the F8 pipeline.

## Folders

| Folder | Purpose | Commit to GitHub? |
|---|---|---|
| `raw/` | Optional local copy of raw datasets | No |
| `interim/` | F8-UC3 standardized CSV outputs | No |
| `processed/` | F8-UC4 cleaned and aggregated outputs | Yes, compressed final files only |

Plain generated CSV files are intentionally ignored by git because they can be large. The final cleaned datasets are committed as compressed `.csv.gz` files so the repository remains usable without becoming too heavy.

## Final Processed Files On GitHub

The GitHub repository includes:

```text
data/processed/iot_15min_clean.csv.gz
data/processed/aquacheck_15min_clean.csv.gz
data/processed/neusta_15min_clean.csv.gz
data/processed/meteo_france_1h_clean.csv.gz
```

These are the cleaned, imputed, normalized, and aggregated F8-UC4 outputs.

You can read them directly with pandas:

```python
import pandas as pd

df = pd.read_csv("data/processed/neusta_15min_clean.csv.gz")
```

The uncompressed local `.csv` versions may also exist on the development machine, but they are not committed.

## Current Raw Data Path

The current raw reference data is expected at:

```text
/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset
```

You can override it when running F8-UC3:

```bash
python scripts/f8_uc3_convert_to_standard_csv.py --dataset-dir "/path/to/dataset"
```
