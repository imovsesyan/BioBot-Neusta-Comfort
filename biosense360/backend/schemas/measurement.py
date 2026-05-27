"""Measurement Pydantic schemas."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class MeasurementOut(BaseModel):
    """Read schema for a single measurement row."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    station_id: int
    timestamp: datetime
    temperature: float
    humidity: float
    humidex: float
    comfort_class: str
    source: str
