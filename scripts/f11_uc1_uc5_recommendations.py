"""F11-UC1 / F11-UC5: Generate personalized comfort recommendations.

Pipeline
--------
1. Load a single source of F10 humidex-labelled rows. Preference order:
     a. ``reports/tables/f10_uc3_alert_examples.csv`` (already carries
        ``risk_level``, ``alert_severity``, and ``humidex_c``).
     b. ``reports/tables/f10_uc1_risk_period_examples.csv`` (same columns).
     c. ``data/processed/meteo_france_1h_clean.csv.gz`` — labelled on the fly
        via :func:`biobot.risk.rules.add_risk_labels` so the pipeline still
        runs even if the report CSVs have not been regenerated.
2. Normalize the risk level: where ``alert_severity == "critical"`` (humidex
   above 54), promote ``risk_level`` from ``dangerous`` to ``critical`` so the
   F11 recommender sees the full 5-band space.
3. Assign every row the baseline profile
   ``UserProfile(activity_level=0, clothing_level=0, vulnerability=0)``
   (sedentary, light clothing, standard adult). This is the conservative
   default — running the system through this profile demonstrates the
   minimum-action path. Per-row personalization is left to downstream
   delivery.
4. Apply :func:`biobot.recommendations.rules_recommender.get_recommendation`.
5. Optionally apply :func:`biobot.recommendations.ai_recommender.get_ai_recommendation`
   when ``USE_AI`` is true and ``ANTHROPIC_API_KEY`` is set.
6. Write results to ``data/outputs/f11_recommendations.csv``.

Scientific notes
----------------
* The recommender is keyed off F10 humidex bands only. We deliberately do
  NOT consume ``vivabilite_binary_mean`` or any F10-UC4 classifier output
  (CLAUDE.md target-leakage guardrail; see Section 4 of
  ``docs/f11/F11_research_recommendations.md``).
* The four upstream sources (IoT, Aquacheck, Neusta, Meteo France) are not
  spatially/temporally aligned. This script reads ONE source per run.
* The output CSV is informational and NON-MEDICAL. Any downstream consumer
  must surface this disclaimer to end users (already present in the
  rephrased AI text and in the recommendation alert field).

Run from the repository root::

    python scripts/f11_uc1_uc5_recommendations.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import pandas as pd


# Recommendations are informational only. They are not medical advice and
# must not be presented as such by any downstream system.
NON_MEDICAL_DISCLAIMER = (
    "F11 recommendations are informational guidance derived from public-health "
    "humidex thresholds (WHO / Santé publique France / ASHRAE 55). They are "
    "NOT medical advice and must not be presented as such."
)


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from biobot.recommendations.profile import UserProfile  # noqa: E402
from biobot.recommendations.rules_recommender import (  # noqa: E402
    VALID_RISK_LEVELS,
    get_recommendation,
)
from biobot.risk.rules import add_risk_labels  # noqa: E402

# AI personalization is optional and gated by both this flag AND the env var.
USE_AI = True

OUTPUT_DIR = ROOT / "data" / "outputs"
OUTPUT_PATH = OUTPUT_DIR / "f11_recommendations.csv"

# Candidate inputs in preference order.
CANDIDATE_INPUTS = [
    ROOT / "reports" / "tables" / "f10_uc3_alert_examples.csv",
    ROOT / "reports" / "tables" / "f10_uc1_risk_period_examples.csv",
]
FALLBACK_RAW = ROOT / "data" / "processed" / "meteo_france_1h_clean.csv.gz"


def _select_input(explicit: Optional[Path]) -> Path:
    """Pick the most appropriate F10 humidex-labelled CSV available."""

    if explicit is not None:
        if not explicit.exists():
            raise FileNotFoundError(f"Explicit input not found: {explicit}")
        return explicit

    for candidate in CANDIDATE_INPUTS:
        if candidate.exists():
            return candidate
    if FALLBACK_RAW.exists():
        return FALLBACK_RAW
    raise FileNotFoundError(
        "No F10 humidex-labelled CSV found. Looked at: "
        + ", ".join(str(p) for p in CANDIDATE_INPUTS + [FALLBACK_RAW])
    )


def _load_with_risk_labels(path: Path) -> pd.DataFrame:
    """Load the chosen input and ensure it carries (risk_level, humidex_c)."""

    df = pd.read_csv(path, low_memory=False)

    # If the source is the cleaned Meteo CSV, we have humidex but no risk
    # labels — synthesize them from the canonical F10 rules.
    if "risk_level" not in df.columns:
        if "humidex_c" not in df.columns:
            raise ValueError(
                f"Input {path} has neither 'risk_level' nor 'humidex_c'."
            )
        df = add_risk_labels(df)

    if "humidex_c" not in df.columns:
        raise ValueError(f"Input {path} is missing 'humidex_c'.")

    # Promote dangerous -> critical when alert_severity flags humidex > 54.
    # The F10 risk_level itself collapses everything above 45 to 'dangerous';
    # the `critical` band is only surfaced via alert_severity / is_critical_humidex.
    if "alert_severity" in df.columns:
        crit_mask = df["alert_severity"] == "critical"
        df.loc[crit_mask, "risk_level"] = "critical"
    elif "is_critical_humidex" in df.columns:
        crit_mask = df["is_critical_humidex"].astype("boolean").fillna(False)
        df.loc[crit_mask, "risk_level"] = "critical"

    # Drop rows we cannot recommend on.
    before = len(df)
    df = df[df["risk_level"].isin(VALID_RISK_LEVELS)].copy()
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} row(s) with unknown / null risk_level.")

    return df


def _apply_recommender(df: pd.DataFrame, profile: UserProfile) -> pd.DataFrame:
    """Apply the rule recommender row-by-row and append output columns."""

    actions, clothing, activity, alerts = [], [], [], []
    for risk in df["risk_level"].astype(str):
        rec = get_recommendation(risk, profile)
        actions.append(rec["action"])
        clothing.append(rec["clothing_advice"])
        activity.append(rec["activity_advice"])
        alerts.append(rec["alert"])

    df = df.copy()
    df["profile_activity"] = profile.activity_level
    df["profile_clothing"] = profile.clothing_level
    df["profile_vulnerability"] = profile.vulnerability
    df["rec_action"] = actions
    df["rec_clothing_advice"] = clothing
    df["rec_activity_advice"] = activity
    df["rec_alert"] = alerts
    return df


def _apply_ai_layer(df: pd.DataFrame, profile: UserProfile, enabled: bool) -> pd.DataFrame:
    """Optionally layer Claude-rephrased text onto each row.

    Disabled by default unless ``USE_AI`` is true at module level AND the
    environment exposes ``ANTHROPIC_API_KEY``. Failures fall back to the
    rule-based action with an explicit note (see ai_recommender).
    """

    if not enabled or not os.environ.get("ANTHROPIC_API_KEY", "").strip():
        df = df.copy()
        df["rec_ai_text"] = ""
        return df

    # Imported lazily so the script remains usable in environments without
    # the anthropic SDK installed.
    from biobot.recommendations.ai_recommender import get_ai_recommendation

    ai_texts = []
    for _, row in df.iterrows():
        base_rec = {
            "action": row["rec_action"],
            "clothing_advice": row["rec_clothing_advice"],
            "activity_advice": row["rec_activity_advice"],
            "alert": row["rec_alert"],
        }
        humidex = float(row.get("humidex_c") or 0.0)
        ai_texts.append(
            get_ai_recommendation(str(row["risk_level"]), profile, humidex, base_rec)
        )
    df = df.copy()
    df["rec_ai_text"] = ai_texts
    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate F11-UC1/UC5 personalized comfort recommendations."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit path to an F10 humidex-labelled CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Output CSV path (default: data/outputs/f11_recommendations.csv).",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Disable the optional AI rephrasing layer regardless of env state.",
    )
    args = parser.parse_args()

    input_path = _select_input(args.input)
    print(f"Reading F10 humidex-labelled rows from: {input_path}")
    df = _load_with_risk_labels(input_path)
    print(f"Loaded {len(df)} row(s) with risk_level in {VALID_RISK_LEVELS}.")

    # Baseline profile per the task spec: sedentary, light clothing, standard adult.
    baseline_profile = UserProfile(
        activity_level=0, clothing_level=0, vulnerability=0
    )

    df = _apply_recommender(df, baseline_profile)

    ai_requested = USE_AI and not args.no_ai
    api_key_present = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    ai_active = ai_requested and api_key_present
    df = _apply_ai_layer(df, baseline_profile, ai_requested)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    print(f"Wrote {args.output} ({len(df)} row(s)).")
    print("Risk level coverage:", df["risk_level"].value_counts().to_dict())
    if ai_active:
        ai_status = "on (Claude rephrasing layer active)"
    elif ai_requested and not api_key_present:
        ai_status = "off (USE_AI=True but ANTHROPIC_API_KEY is not set)"
    else:
        ai_status = "off (disabled by USE_AI flag or --no-ai)"
    print(f"AI personalization: {ai_status}")
    print(NON_MEDICAL_DISCLAIMER)


if __name__ == "__main__":
    main()
