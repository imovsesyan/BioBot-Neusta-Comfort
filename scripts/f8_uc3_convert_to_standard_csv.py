"""F8-UC3: Convert raw BioBot datasets to standardized CSV files.

Run from the repository root:
    .venv/bin/python scripts/f8_uc3_convert_to_standard_csv.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.data.standardize import write_standardized_csvs  # noqa: E402


DEFAULT_DATASET_DIR = Path("/Users/inesamovsesyan/Desktop/Neusta Biosense360/dataset")
DEFAULT_OUTPUT_DIR = ROOT / "data" / "interim"
DEFAULT_SUMMARY_PATH = ROOT / "reports" / "tables" / "f8_uc3_standardization_summary.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw BioBot data to standardized CSV.")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY_PATH)
    args = parser.parse_args()

    summary = write_standardized_csvs(args.dataset_dir, args.output_dir)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote standardized CSV files to {args.output_dir}")
    print(f"Wrote UC3 summary to {args.summary}")
    for source, profile in summary.items():
        print(
            f"{source}: {profile['rows']:,} rows, "
            f"{profile['invalid_timestamps']:,} invalid timestamps"
        )


if __name__ == "__main__":
    main()

