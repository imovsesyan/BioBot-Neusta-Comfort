"""Comfort class request/response schemas."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class ComfortRequest(BaseModel):
    """Body for POST /api/comfort-class."""
    temperature: float = Field(..., ge=-50, le=60, description="Dry-bulb temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in %")
    station_id: Optional[int] = None
    timestamp: Optional[datetime] = None


class ComfortResponse(BaseModel):
    """Response from POST /api/comfort-class."""
    temperature: float
    humidity: float
    humidex: float
    comfort_class: str
    risk_level: str
    station_id: Optional[int] = None
    timestamp: Optional[datetime] = None
