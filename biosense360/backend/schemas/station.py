"""Station Pydantic schemas."""

from pydantic import BaseModel, ConfigDict


class StationOut(BaseModel):
    """Read schema for a station — returned by GET /api/stations endpoints."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    lat: float
    lon: float
    type: str
    active: bool
