"""
AI Advisor router — POST /api/ai/explain (SSE streaming)

Streams a personalised thermal safety briefing from Claude via
Server-Sent Events so the frontend can render tokens incrementally.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from backend.services.claude_service import stream_ai_explanation

router = APIRouter(prefix="/ai", tags=["ai"])


class AiExplainRequest(BaseModel):
    """Request body for the AI explanation endpoint."""
    date: str               # ISO date string, e.g. "2024-07-15"
    station: str            # Station name for context
    humidex: float          # Current Humidex value
    comfort_class: str      # e.g. "Danger"
    user_type: str          # e.g. "Worker", "Elderly", "Athlete", "Child"


@router.post("/explain")
async def ai_explain(req: AiExplainRequest) -> StreamingResponse:
    """
    Stream a Claude-powered thermal safety briefing as Server-Sent Events.

    The response uses text/event-stream content type.
    Each chunk is formatted as:
        data: <token text>\n\n

    The stream ends with:
        data: [DONE]\n\n

    The frontend should accumulate chunks and split on '\\n' to reconstruct
    the original line breaks.
    """
    try:
        generator = stream_ai_explanation(
            date=req.date,
            station=req.station,
            humidex=req.humidex,
            comfort_class=req.comfort_class,
            user_type=req.user_type,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",  # disable nginx buffering for SSE
            },
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(exc)}")
