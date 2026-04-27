"""F11 personalized comfort recommendations.

Modules:
    profile           — UserProfile schema + synthetic profile generator (F11-UC1/UC5).
    rules_recommender — F11-UC1 deterministic (humidex risk x profile) recommendation engine.
    ai_recommender    — F11-UC5 optional Claude rephrasing layer on top of the rule engine.

Scientific note: the F11 layer is keyed off F10 humidex risk levels (deterministic
public-health bands) and never off `vivabilite_binary_mean` or F10-UC4 classifier
output, in line with the leakage guardrails in CLAUDE.md and the F11 research report.
Outputs are informational and non-medical.
"""

from biobot.recommendations.profile import UserProfile, generate_synthetic_profiles
from biobot.recommendations.rules_recommender import get_recommendation

__all__ = [
    "UserProfile",
    "generate_synthetic_profiles",
    "get_recommendation",
]
