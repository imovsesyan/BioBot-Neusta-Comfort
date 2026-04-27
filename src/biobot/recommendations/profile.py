"""F11-UC1 / F11-UC5 user profile schema.

This module defines the minimal user profile used to personalize comfort
recommendations. Per the task specification, each field is encoded as a binary
integer to keep the cross with the F10 humidex risk levels small and auditable
(2 x 2 x 2 = 8 profile buckets).

Field semantics (binary encoding):

    activity_level   : 1 = active     (walking / light housework / outdoor work)
                        0 = sedentary  (resting / desk work)
    clothing_level   : 1 = heavy      (winter indoor / outerwear)
                        0 = light      (summer / indoor casual)
    vulnerability    : 1 = vulnerable  (elderly 65+, child < 12, chronic condition,
                                        pregnancy)
                        0 = standard    (healthy adult, no acute conditions)

Scientific notes
----------------
* The richer 3-bin categorical encoding (mapped to MET / CLO bands) discussed in
  ``docs/f11/F11_research_recommendations.md`` is the long-term target. For the
  current implementation we use the binary form requested in the F11
  specification, which still preserves the three axes (metabolism, insulation,
  vulnerability) that scientifically drive thermal-comfort personalization in
  ASHRAE 55, ISO 7730, WHO heat-health guidance, and Santé publique France's
  Plan Canicule.
* The vulnerability flag is the single most influential field — heat-related
  health outcomes are dominated by vulnerability (WHO, EuroHEAT, CDC Heat &
  Health). This is why vulnerable + (high_risk | dangerous | critical) carries
  a mandatory non-empty alert in the rule engine (see ``rules_recommender``).
* These features are NOT used to train any model and have no relationship to
  the F9/F10 livability target. They drive deterministic rule lookups only,
  which keeps the F11 layer free of the target-leakage risk flagged in
  CLAUDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import List


@dataclass(frozen=True)
class UserProfile:
    """Binary user profile used for personalized comfort recommendations.

    All three fields are integers in {0, 1}. The dataclass is frozen so
    profiles are hashable and can be used as dict keys for caching.
    """

    activity_level: int  # 1 = active,     0 = sedentary
    clothing_level: int  # 1 = heavy,      0 = light
    vulnerability: int   # 1 = vulnerable, 0 = standard

    def __post_init__(self) -> None:
        for name in ("activity_level", "clothing_level", "vulnerability"):
            value = getattr(self, name)
            if value not in (0, 1):
                raise ValueError(
                    f"UserProfile.{name} must be 0 or 1, got {value!r}"
                )


def generate_synthetic_profiles() -> List[UserProfile]:
    """Return all 8 combinations of the binary user profile (2 x 2 x 2).

    The generator is deterministic and yields profiles in a stable order so
    that downstream rule tables and tests can rely on it. Use this for
    stakeholder demos, rule-coverage tests, and synthetic batch evaluation.
    """

    profiles: List[UserProfile] = []
    for activity, clothing, vulnerability in product((0, 1), repeat=3):
        profiles.append(
            UserProfile(
                activity_level=activity,
                clothing_level=clothing,
                vulnerability=vulnerability,
            )
        )
    return profiles
