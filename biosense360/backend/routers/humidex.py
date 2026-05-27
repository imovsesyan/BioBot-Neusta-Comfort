"""
Humidex router — GET /api/measurements, POST /api/comfort-class
"""

from datetime import date, datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.measurement import Measurement
from backend.schemas.comfort import ComfortRequest, ComfortResponse
from backend.schemas.measurement import MeasurementOut
from backend.services.humidex_service import (
    classify_humidex,
    compute_humidex,
    humidex_to_risk_level,
)

router = APIRouter(tags=["humidex"])


@router.get("/measurements", response_model=list[MeasurementOut])
def get_measurements(
    station_id: Optional[int] = Query(None, description="Filter by station ID"),
    date: Optional[str] = Query(None, description="Filter by date YYYY-MM-DD"),
    source: Optional[str] = Query(None, description="Filter by source (neusta/meteo_france)"),
    limit: int = Query(500, ge=1, le=5000, description="Max rows to return"),
    db: Session = Depends(get_db),
) -> list[MeasurementOut]:
    """
    Return measurements with optional filtering by station, date, and source.
    Defaults to the most recent 500 rows if no filters are applied.
    """
    try:
        query = db.query(Measurement)

        if station_id is not None:
            query = query.filter(Measurement.station_id == station_id)

        if date is not None:
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise HTTPException(
                    status_code=422, detail="date must be in YYYY-MM-DD format"
                )
            # Filter to rows whose timestamp date matches
            query = query.filter(
                Measurement.timestamp >= datetime(
                    target_date.year, target_date.month, target_date.day,
                    0, 0, 0, tzinfo=timezone.utc
                ),
                Measurement.timestamp < datetime(
                    target_date.year, target_date.month, target_date.day,
                    23, 59, 59, tzinfo=timezone.utc
                ),
            )

        if source is not None:
            query = query.filter(Measurement.source == source)

        measurements = (
            query.order_by(Measurement.timestamp.desc()).limit(limit).all()
        )
        return measurements

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)}")


@router.post("/comfort-class", response_model=ComfortResponse)
def compute_comfort_class(req: ComfortRequest) -> ComfortResponse:
    """
    Compute Humidex and comfort class from raw temperature and humidity.
    Does not require a database — pure calculation endpoint.
    """
    try:
        humidex_val = compute_humidex(req.temperature, req.humidity)
        comfort_cls = classify_humidex(humidex_val)
        risk_level = humidex_to_risk_level(humidex_val)

        return ComfortResponse(
            temperature=req.temperature,
            humidity=req.humidity,
            humidex=round(humidex_val, 2),
            comfort_class=comfort_cls,
            risk_level=risk_level,
            station_id=req.station_id,
            timestamp=req.timestamp,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Computation error: {str(exc)}")
