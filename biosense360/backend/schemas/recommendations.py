"""Recommendation request/response schemas."""

from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Human recommendations
# ---------------------------------------------------------------------------

class HumanRecoRequest(BaseModel):
    """Body for POST /api/recommendations/human."""
    humidex: float = Field(..., description="Current Humidex value")
    comfort_class: str = Field(..., description="Comfort class label")
    age_group: str = Field(
        ...,
        description="Population group: Worker | Elderly | Child | Athlete",
    )
    activity_type: str = Field(
        ...,
        description="Activity type: Outdoor work | Sports | Commuting | Rest",
    )


class HumanRecoResponse(BaseModel):
    """Response for POST /api/recommendations/human."""
    advice: str
    hydration: str
    clothing: str
    risk_level: str
    safe_slots: list[str]
    emergency_protocol: str


# ---------------------------------------------------------------------------
# Irrigation recommendations
# ---------------------------------------------------------------------------

class IrrigationRequest(BaseModel):
    """Body for POST /api/recommendations/irrigation."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    station_id: int = Field(..., description="Station ID to fetch humidex context")
    plant_type: str = Field(
        ...,
        description="Plant type: Vegetables | Flowers | Crops | Grass",
    )
    soil_type: str = Field(
        ...,
        description="Soil type: Clay | Sandy | Loamy",
    )


class IrrigationResponse(BaseModel):
    """Response for POST /api/recommendations/irrigation."""
    should_irrigate: bool
    best_slots: list[str]
    water_liters: float
    frequency: str
    reason: str
    alert: Optional[str] = None
