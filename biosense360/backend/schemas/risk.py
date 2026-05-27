"""Risk-related Pydantic schemas."""

import datetime
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict


class RiskPeriodOut(BaseModel):
    """Read schema for a risk_periods row."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    station_id: int
    date: datetime.date
    risk_level: str
    favorable_hours: list[str]
    dangerous_hours: list[str]
    avg_humidex: Optional[float]
    avg_temp: Optional[float]


class TimeSlot(BaseModel):
    """Single hour-slot forecast entry."""
    hour: int           # 0–23
    label: str          # e.g. "14:00"
    temperature: float
    humidity: float
    humidex: float
    comfort_class: str
    risk_level: str


class TimeSlotForecast(BaseModel):
    """Response for GET /api/forecast/time-slots."""
    station_id: int
    date: str           # ISO date string "YYYY-MM-DD"
    slots: list[TimeSlot]
    favorable_hours: list[str]
    dangerous_hours: list[str]
    day_risk_level: str


class RiskSummaryResponse(BaseModel):
    """Response for GET /api/risk-summary."""
    date: str
    station_id: Optional[int]
    risk_level: str
    avg_humidex: float
    avg_temp: float
    favorable_hours: list[str]
    dangerous_hours: list[str]
    stations_breakdown: list[dict[str, Any]]


class RiskMatrixCell(BaseModel):
    """One cell in the departure × arrival risk matrix."""
    departure: str      # station name
    arrival: str        # station name
    risk_level: str
    avg_humidex: float
    recommendation: str


class RiskMatrixResponse(BaseModel):
    """Response for GET /api/risk-matrix."""
    date: str
    matrix: list[RiskMatrixCell]
