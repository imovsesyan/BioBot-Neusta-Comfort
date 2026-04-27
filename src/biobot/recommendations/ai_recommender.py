"""F11-UC5 AI personalization layer (rephrasing only).

Per the F11 research recommendation
(``docs/f11/F11_research_recommendations.md``, Section 2 Option D / Section 5),
the LLM never invents medical content. The rule engine in
:mod:`biobot.recommendations.rules_recommender` produces the structured
``(action, clothing_advice, activity_advice, alert)`` recommendation, and this
module asks Claude to rephrase that structured payload into a single, friendly
paragraph tailored to the user profile. The LLM's role is *style*, not
*content* — this preserves determinism of the underlying advice and keeps
medical-adjacent statements traceable to the rule table.

Implementation notes
--------------------
* Model: ``claude-haiku-4-5-20251001`` (cheap, fast, sufficient for short
  rephrasing). The model id is exposed as a module-level constant so it can be
  overridden in tests or future migrations.
* Prompt caching: the static system prompt (which contains the persona, the
  rephrasing constraints, and the non-medical-advice disclaimer) is marked
  ``cache_control={"type": "ephemeral"}`` so repeated calls within the 5-minute
  TTL hit the cache and only re-charge for the small per-row user message.
* In-process result cache: since the input space is small
  (5 risk levels x 8 profiles x ~5 humidex bands = 200 cells), we additionally
  cache (risk_level, activity, clothing, vulnerability, humidex_band) ->
  text in a module-level dict, eliminating redundant API calls within a run.
* Graceful fallback: if ``ANTHROPIC_API_KEY`` is unset, or the ``anthropic``
  SDK is not installed, or the API call raises, we return the base
  recommendation's ``action`` string with a clear "AI personalization
  unavailable" suffix. The script continues to produce useful output offline.

This module imports ``anthropic`` lazily inside the function so that environments
without the SDK installed (including the test harness for the fallback path)
can still import the recommender package.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

from biobot.recommendations.profile import UserProfile

# Public model id — kept as a module attribute so callers / tests can override.
MODEL_ID = "claude-haiku-4-5-20251001"

# Soft token limit for the rephrased paragraph. The base recommendation has
# four short fields; ~350 output tokens is comfortable headroom.
MAX_OUTPUT_TOKENS = 400

# Static, cacheable system prompt. Anything that varies per row goes in the
# user message. The non-medical-advice disclaimer is part of the cached prompt
# so every response inherits it.
_SYSTEM_PROMPT = (
    "You are a thermal-comfort assistant that REPHRASES pre-authored advice "
    "for a single user. You will receive a structured base recommendation "
    "(action, clothing_advice, activity_advice, optional alert) plus a user "
    "profile and the current humidex value. Your job is to merge these into "
    "ONE short, friendly paragraph (3-5 sentences) addressed directly to the "
    "user.\n\n"
    "STRICT RULES:\n"
    "1. Do not invent medical thresholds, drugs, dosages, symptoms, or "
    "actions that are not present in the structured base recommendation.\n"
    "2. Do not soften or remove any vulnerability alert — it must be "
    "communicated faithfully.\n"
    "3. Do not contradict the base recommendation. You may reorder and "
    "rephrase, but the meaning must match.\n"
    "4. Keep the tone calm, factual, and supportive. Avoid alarmist "
    "language unless the alert text already carries it.\n"
    "5. End with the disclaimer: \"This is informational guidance, not "
    "medical advice.\"\n"
    "6. Output plain text only, no markdown headings or bullet lists.\n"
)


# In-process cache: keyed by (risk_level, activity, clothing, vulnerability, humidex_band).
# This is intentionally module-level so it survives across calls within a run
# (e.g. a batch script processing the F10 alert CSV).
_CACHE: Dict[Tuple[str, int, int, int, str], str] = {}


def _humidex_band(humidex: float) -> str:
    """Bucket a humidex value to a coarse label for cache-key stability.

    Two readings within the same band produce the same rephrased text, which
    is the desired behaviour for batch personalization (the action plan does
    not change at decimal-point resolution).
    """

    try:
        h = float(humidex)
    except (TypeError, ValueError):
        return "unknown"
    if h < 30:
        return "below_30"
    if h < 40:
        return "30_to_39"
    if h <= 45:
        return "40_to_45"
    if h <= 54:
        return "46_to_54"
    return "above_54"


def _build_user_message(
    risk_level: str,
    profile: UserProfile,
    humidex: float,
    base_rec: dict,
) -> str:
    """Render the per-row user message. This is NOT cached on the API side."""

    activity_label = "active" if profile.activity_level == 1 else "sedentary"
    clothing_label = "heavy" if profile.clothing_level == 1 else "light"
    vulnerability_label = "vulnerable" if profile.vulnerability == 1 else "standard"

    alert_block = base_rec.get("alert") or "(no vulnerability alert)"

    return (
        f"Humidex: {humidex:.1f} (risk level: {risk_level})\n"
        f"User profile:\n"
        f"  - activity: {activity_label}\n"
        f"  - clothing: {clothing_label}\n"
        f"  - vulnerability: {vulnerability_label}\n\n"
        f"Base recommendation to rephrase:\n"
        f"  - action: {base_rec.get('action', '')}\n"
        f"  - clothing_advice: {base_rec.get('clothing_advice', '')}\n"
        f"  - activity_advice: {base_rec.get('activity_advice', '')}\n"
        f"  - alert: {alert_block}\n\n"
        f"Write the rephrased paragraph now."
    )


def _fallback_text(base_rec: dict, reason: str) -> str:
    """Offline / error fallback. Returns the base action plus an explicit note."""

    action = base_rec.get("action", "") if isinstance(base_rec, dict) else ""
    return (
        f"{action} (AI personalization unavailable: {reason}. "
        f"This is informational guidance, not medical advice.)"
    )


def _extract_text(response) -> str:
    """Pull the plain text out of a Messages API response, defensively."""

    try:
        chunks = []
        for block in response.content:
            text = getattr(block, "text", None)
            if text:
                chunks.append(text)
        return "\n".join(chunks).strip()
    except Exception:  # pragma: no cover — defensive only
        return ""


def get_ai_recommendation(
    risk_level: str,
    profile: UserProfile,
    humidex: float,
    base_rec: dict,
) -> str:
    """Return a Claude-rephrased version of ``base_rec`` for ``profile``.

    Falls back to the base action with a clear note when the API key is
    missing, the SDK is not installed, or the call fails for any reason.
    Never raises.
    """

    # Validate enough to fail fast on programmer errors, but never raise on
    # API/runtime failures.
    if not isinstance(profile, UserProfile):
        raise TypeError(
            f"profile must be a UserProfile instance, got {type(profile).__name__}"
        )
    if not isinstance(base_rec, dict):
        raise TypeError(
            f"base_rec must be a dict, got {type(base_rec).__name__}"
        )

    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        return _fallback_text(base_rec, "ANTHROPIC_API_KEY not set")

    cache_key = (
        risk_level,
        profile.activity_level,
        profile.clothing_level,
        profile.vulnerability,
        _humidex_band(humidex),
    )
    cached = _CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Lazy import so the module is usable (and testable) without the SDK
    # installed on the host machine.
    try:
        import anthropic  # type: ignore
    except ImportError:
        return _fallback_text(base_rec, "anthropic SDK not installed")

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=MODEL_ID,
            max_tokens=MAX_OUTPUT_TOKENS,
            system=[
                {
                    "type": "text",
                    "text": _SYSTEM_PROMPT,
                    # Prompt caching: the static system prompt is reused across
                    # every row, so we mark it ephemeral. Only the small user
                    # message is re-billed at full rate.
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": _build_user_message(
                        risk_level, profile, humidex, base_rec
                    ),
                }
            ],
        )
    except Exception as exc:  # pragma: no cover — defensive
        # Any API-side failure (network, rate limit, auth, model id, etc.)
        # falls through to the rule-based action so the pipeline never crashes.
        return _fallback_text(base_rec, f"API call failed: {exc.__class__.__name__}")

    text = _extract_text(response)
    if not text:
        return _fallback_text(base_rec, "empty model response")

    _CACHE[cache_key] = text
    return text


def clear_cache() -> None:
    """Reset the in-process rephrasing cache (useful for tests / batch runs)."""

    _CACHE.clear()
