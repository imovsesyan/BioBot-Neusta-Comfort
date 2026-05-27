"""Pydantic schema package — re-exports all response/request models."""

from backend.schemas.comfort import ComfortRequest, ComfortResponse
from backend.schemas.measurement import MeasurementOut
from backend.schemas.recommendations import (
    HumanRecoRequest,
    HumanRecoResponse,
    IrrigationRequest,
    IrrigationResponse,
)
from backend.schemas.risk import (
    RiskMatrixResponse,
    RiskPeriodOut,
    RiskSummaryResponse,
    TimeSlotForecast,
)
from backend.schemas.station import StationOut

__all__ = [
    "StationOut",
    "MeasurementOut",
    "ComfortRequest",
    "ComfortResponse",
    "RiskPeriodOut",
    "RiskSummaryResponse",
    "RiskMatrixResponse",
    "TimeSlotForecast",
    "HumanRecoRequest",
    "HumanRecoResponse",
    "IrrigationRequest",
    "IrrigationResponse",
]
