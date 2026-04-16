"""F8-UC4: Clean, impute, normalize, and aggregate standardized BioBot CSVs.

Run from the repository root after F8-UC3:
    .venv/bin/python scripts/f8_uc4_clean_impute_normalize_aggregate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.data.clean_aggregate import run_uc4_pipeline, write_summary  # noqa: E402


DEFAULT_INTERIM_DIR = ROOT / "data" / "interim"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "processed"
DEFAULT_SUMMARY_PATH = ROOT / "reports" / "tables" / "f8_uc4_cleaning_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean, impute, normalize, and aggregate BioBot standardized CSV files."
    )
    parser.add_argument("--interim-dir", type=Path, default=DEFAULT_INTERIM_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    parser.add_argument("--freq", default="15min", help="Aggregation frequency for sensor datasets.")
    args = parser.parse_args()

    summary = run_uc4_pipeline(args.interim_dir, args.output_dir, args.freq)
    write_summary(summary, args.summary)

    print(f"Wrote cleaned CSV files to {args.output_dir}")
    print(f"Wrote UC4 summary to {args.summary}")
    for source, profile in summary.items():
        print(
            f"{source}: {profile['input_rows']:,} input rows -> "
            f"{profile['output_rows']:,} aggregated rows"
        )


if __name__ == "__main__":
    main()

