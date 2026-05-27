"""
Claude AI integration service for BioSense360.

Streams thermal comfort briefings via the Anthropic API using SSE
(Server-Sent Events) so the frontend can render tokens as they arrive.

Model: claude-sonnet-4-20250514
Max tokens: 300 — briefings must be concise (bullet-point style)
"""

import os
from typing import AsyncGenerator

import anthropic
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt — instructs Claude to act as BioSense360's thermal advisor.
# Cites approved protocols and enforces brevity.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are BioSense360's thermal comfort advisor for Toulouse, France. "
    "Provide clear, actionable advice based on Humidex data. "
    "Always cite the relevant protocol (HAS 2023, OHCOW 2022, ACGIH TLV 2023, "
    "INRS R-447, Décret 2025-482) when appropriate. "
    "Be concise. Use bullet points. Maximum 150 words. "
    "Always respond in the same language as the user's prompt."
)


def _build_user_prompt(
    date: str,
    station: str,
    humidex: float,
    comfort_class: str,
    user_type: str,
) -> str:
    """Construct the per-request user prompt sent to Claude."""
    return (
        f"Date: {date}. "
        f"Station: {station}. "
        f"Humidex: {humidex:.1f}. "
        f"Class: {comfort_class}. "
        f"User type: {user_type}. "
        "Give today's thermal safety briefing."
    )


async def stream_ai_explanation(
    date: str,
    station: str,
    humidex: float,
    comfort_class: str,
    user_type: str,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields SSE-formatted text chunks from Claude.

    Each yielded value is a fully-formed SSE data line, e.g.:
        'data: Here is the first chunk\n\n'

    The caller (the FastAPI router) passes this generator to EventSourceResponse.

    Args:
        date:          ISO date string (e.g., "2024-07-15").
        station:       Station name for context.
        humidex:       Current Humidex value.
        comfort_class: Comfort class label (e.g., "Danger").
        user_type:     User profile (e.g., "Worker", "Elderly").

    Yields:
        SSE-formatted string chunks.

    Raises:
        anthropic.APIError on API failures — the caller should handle this.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        # Graceful degradation — return a helpful fallback when no API key is configured
        fallback = (
            f"[AI Advisor offline — ANTHROPIC_API_KEY not set]\n\n"
            f"Manual briefing for {station} on {date}:\n"
            f"- Humidex: {humidex:.1f} ({comfort_class})\n"
            f"- Consult OHCOW 2022 tables for {user_type} action thresholds.\n"
            f"- Contact your safety officer if Humidex > 40."
        )
        yield f"data: {fallback}\n\n"
        return

    client = anthropic.AsyncAnthropic(api_key=api_key)
    user_prompt = _build_user_prompt(date, station, humidex, comfort_class, user_type)

    try:
        async with client.messages.stream(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            async for text_chunk in stream.text_stream:
                # Escape newlines inside the SSE data value so each chunk
                # is a valid single-line SSE event.
                escaped = text_chunk.replace("\n", "\\n")
                yield f"data: {escaped}\n\n"

        # Signal stream completion
        yield "data: [DONE]\n\n"

    except anthropic.APIStatusError as exc:
        error_msg = f"[Claude API error {exc.status_code}]: {exc.message}"
        yield f"data: {error_msg}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as exc:
        yield f"data: [Error]: {str(exc)}\n\n"
        yield "data: [DONE]\n\n"
