"""
Recommendations router — POST /api/recommendations/human, POST /api/recommendations/irrigation
"""

import math
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.risk_period import RiskPeriod
from backend.models.station import Station
from backend.schemas.recommendations import (
    HumanRecoRequest,
    HumanRecoResponse,
    IrrigationRequest,
    IrrigationResponse,
)
from backend.services.human_reco_service import generate_human_recommendation
from backend.services.irrigation_service import generate_irrigation_recommendation

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


@router.post("/human", response_model=HumanRecoResponse)
def human_recommendations(req: HumanRecoRequest) -> HumanRecoResponse:
    """
    Generate personalised human thermal safety advice for a given Humidex,
    population group, and activity type.

    Does not require a database — all logic is in human_reco_service.
    """
    try:
        return generate_human_recommendation(req)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(exc)}")


@router.post("/irrigation", response_model=IrrigationResponse)
def irrigation_recommendations(
    req: IrrigationRequest,
    db: Session = Depends(get_db),
) -> IrrigationResponse:
    """
    Generate an irrigation plan based on the thermal conditions at a station
    on a given date, adjusted for plant type and soil type.
    """
    try:
        # Parse date
        try:
            target_date = datetime.strptime(req.date, "%Y-%m-%d").date()
        except ValueError:
            raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")

        # Fetch aggregated conditions from risk_periods
        rp = (
            db.query(RiskPeriod)
            .filter(
                RiskPeriod.station_id == req.station_id,
                RiskPeriod.date == target_date,
            )
            .first()
        )

        if rp and rp.avg_humidex is not None:
            avg_humidex = rp.avg_humidex
            # Average humidity (for high-humidity reduction logic)
            avg_humidity = 55.0  # sensible default when only risk_period aggregation is available
        else:
            # Fallback 1: compute from raw measurements if available.
            from backend.models.measurement import Measurement
            start = datetime(target_date.year, target_date.month, target_date.day, tzinfo=timezone.utc)
            end = datetime(target_date.year, target_date.month, target_date.day, 23, 59, 59, tzinfo=timezone.utc)

            measurements = (
                db.query(Measurement)
                .filter(
                    Measurement.station_id == req.station_id,
                    Measurement.timestamp >= start,
                    Measurement.timestamp <= end,
                )
                .all()
            )

            if measurements:
                avg_humidex = sum(m.humidex for m in measurements) / len(measurements)
                avg_humidity = sum(m.humidity for m in measurements) / len(measurements)
            else:
                # Fallback 2: no DB data at all — synthesise daily average from
                # the same sinusoidal model used by the seed script, so the
                # endpoint never returns 404.
                TEMP_MIN, TEMP_MAX = 18.0, 38.0
                HUMID_MIN, HUMID_MAX = 35.0, 70.0
                PEAK_HOUR = 14.0
                station_obj = db.query(Station).filter(Station.id == req.station_id).first()
                is_indoor = station_obj.type == "indoor" if station_obj else False

                daily_humidex: list[float] = []
                daily_humid: list[float] = []
                for hour in range(24):
                    phase = (hour - PEAK_HOUR) * (math.pi / 12.0)
                    temp = ((TEMP_MAX + TEMP_MIN) / 2.0) - ((TEMP_MAX - TEMP_MIN) / 2.0) * math.cos(phase)
                    humid = ((HUMID_MAX + HUMID_MIN) / 2.0) + ((HUMID_MAX - HUMID_MIN) / 2.0) * math.cos(phase)
                    if is_indoor:
                        temp -= 4.0
                        humid -= 10.0
                    from backend.services.humidex_service import compute_humidex
                    daily_humidex.append(compute_humidex(temp, humid))
                    daily_humid.append(max(10.0, min(100.0, humid)))

                avg_humidex = sum(daily_humidex) / 24
                avg_humidity = sum(daily_humid) / 24

        return generate_irrigation_recommendation(req, avg_humidex, avg_humidity)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Irrigation error: {str(exc)}")
