"""Tests for the F11-UC1 / F11-UC5 recommendation system.

Coverage
--------
1. ``test_all_profile_combinations``: ``generate_synthetic_profiles`` returns
   exactly 8 unique binary profiles.
2. ``test_rule_coverage``: every (risk_level x profile) combo returns a dict
   with all four required keys non-None.
3. ``test_vulnerable_alert``: vulnerable=1 + risk in {high_risk, dangerous,
   critical} always returns a non-empty alert string.
4. ``test_standard_no_alert_at_low_risk``: vulnerable=0 + risk_level=livable
   returns the empty string for the alert field.
5. ``test_ai_recommender_fallback``: with ``ANTHROPIC_API_KEY`` unset, the AI
   layer returns a non-empty string and never raises.
"""

from __future__ import annotations

import os

import pytest

from biobot.recommendations.profile import UserProfile, generate_synthetic_profiles
from biobot.recommendations.rules_recommender import (
    ELEVATED_RISK_LEVELS,
    REQUIRED_KEYS,
    VALID_RISK_LEVELS,
    get_recommendation,
)
from biobot.recommendations import ai_recommender


def test_all_profile_combinations():
    profiles = generate_synthetic_profiles()

    # Exactly 2 x 2 x 2 = 8 combinations, all unique.
    assert len(profiles) == 8
    assert len(set(profiles)) == 8

    # Each field is a binary integer.
    for p in profiles:
        assert isinstance(p, UserProfile)
        assert p.activity_level in (0, 1)
        assert p.clothing_level in (0, 1)
        assert p.vulnerability in (0, 1)

    # Spot-check that every binary triple appears.
    triples = {(p.activity_level, p.clothing_level, p.vulnerability) for p in profiles}
    assert len(triples) == 8


def test_rule_coverage():
    profiles = generate_synthetic_profiles()

    # 5 risk levels x 8 profiles = 40 combinations.
    for risk_level in VALID_RISK_LEVELS:
        for profile in profiles:
            rec = get_recommendation(risk_level, profile)

            assert isinstance(rec, dict)
            for key in REQUIRED_KEYS:
                assert key in rec, f"Missing key {key} for {risk_level}, {profile}"
                assert rec[key] is not None, (
                    f"Key {key} is None for {risk_level}, {profile}"
                )
                assert isinstance(rec[key], str), (
                    f"Key {key} is not a str for {risk_level}, {profile}"
                )

            # action / clothing_advice / activity_advice must always be
            # non-empty regardless of profile.
            for key in ("action", "clothing_advice", "activity_advice"):
                assert rec[key].strip() != "", (
                    f"Key {key} is empty for {risk_level}, {profile}"
                )


def test_vulnerable_alert():
    """Vulnerable=1 + elevated risk always carries a non-empty alert."""

    profiles = [p for p in generate_synthetic_profiles() if p.vulnerability == 1]
    assert len(profiles) == 4  # 2 x 2 x 1

    for profile in profiles:
        for risk_level in ELEVATED_RISK_LEVELS:
            rec = get_recommendation(risk_level, profile)
            alert = rec["alert"]
            assert isinstance(alert, str)
            assert alert.strip() != "", (
                f"Empty alert for vulnerable profile {profile} "
                f"at elevated risk {risk_level}"
            )


def test_standard_no_alert_at_low_risk():
    """Vulnerable=0 + risk_level=livable must produce an empty alert string."""

    standard_profiles = [
        p for p in generate_synthetic_profiles() if p.vulnerability == 0
    ]
    assert len(standard_profiles) == 4

    for profile in standard_profiles:
        rec = get_recommendation("livable", profile)
        assert rec["alert"] == "", (
            f"Standard profile {profile} at 'livable' should have empty alert; "
            f"got {rec['alert']!r}"
        )


def test_ai_recommender_fallback(monkeypatch):
    """With ANTHROPIC_API_KEY unset, the AI layer must return a non-empty
    string and never raise. The cache must also not poison the fallback."""

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    ai_recommender.clear_cache()
    # Sanity check that the env truly is empty for this test.
    assert os.environ.get("ANTHROPIC_API_KEY") is None

    profile = UserProfile(activity_level=0, clothing_level=0, vulnerability=0)
    base_rec = get_recommendation("discomfort", profile)

    text = ai_recommender.get_ai_recommendation(
        "discomfort", profile, humidex=35.0, base_rec=base_rec
    )

    assert isinstance(text, str)
    assert text.strip() != ""
    # The fallback should explicitly note that AI is unavailable so that
    # downstream consumers can detect it.
    assert "AI personalization unavailable" in text


@pytest.mark.parametrize("risk_level", list(VALID_RISK_LEVELS))
def test_unknown_risk_level_raises(risk_level):
    """Sanity: known risk levels do NOT raise; unknown ones DO raise."""

    profile = UserProfile(activity_level=0, clothing_level=0, vulnerability=0)
    # Should not raise.
    get_recommendation(risk_level, profile)


def test_unknown_risk_level_value_error():
    profile = UserProfile(activity_level=0, clothing_level=0, vulnerability=0)
    with pytest.raises(ValueError):
        get_recommendation("not_a_level", profile)
