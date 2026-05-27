"""
Trends router — GET /api/trends

Returns aggregated daily statistics for a station over a date range.
Used by the Dashboard page's 30-day dual-axis chart.
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.measurement import Measurement
from backend.models.risk_period import RiskPeriod

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("")
def get_trends(
    station_id: int = Query(..., description="Station ID"),
    from_date: str = Query(..., alias="from", description="Start date YYYY-MM-DD"),
    to_date: str = Query(..., alias="to", description="End date YYYY-MM-DD"),
    db: Session = Depends(get_db),
) -> dict:
    """
    Return daily aggregated temperature, humidity, and humidex for a station
    over the specified date range. Used for the 30-day trend chart.

    Response shape:
    {
        "station_id": 1,
        "from": "2024-07-01",
        "to": "2024-07-30",
        "daily": [
            {
                "date": "2024-07-01",
                "avg_temp": 28.5,
                "avg_humidity": 52.0,
                "avg_humidex": 33.1,
                "max_humidex": 41.2,
                "risk_level": "MODERATE"
            }, ...
        ]
    }
    """
    try:
        try:
            start = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            end = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, tzinfo=timezone.utc)
        except ValueError:
            raise HTTPException(status_code=422, detail="from and to must be YYYY-MM-DD")

        if start > end:
            raise HTTPException(status_code=422, detail="from must be before to")

        # Try pre-computed risk_periods first (one row per day — very fast)
        risk_periods = (
            db.query(RiskPeriod)
            .filter(
                RiskPeriod.station_id == station_id,
                RiskPeriod.date >= start.date(),
                RiskPeriod.date <= end.date(),
            )
            .order_by(RiskPeriod.date.asc())
            .all()
        )

        if risk_periods:
            daily = [
                {
                    "date": rp.date.isoformat(),
                    "avg_temp": rp.avg_temp,
                    "avg_humidex": rp.avg_humidex,
                    "risk_level": rp.risk_level,
                }
                for rp in risk_periods
            ]
            return {
                "station_id": station_id,
                "from": from_date,
                "to": to_date,
                "daily": daily,
            }

        # Fallback: aggregate from raw measurements
        rows = (
            db.query(
                func.date(Measurement.timestamp).label("day"),
                func.avg(Measurement.temperature).label("avg_temp"),
                func.avg(Measurement.humidity).label("avg_humidity"),
                func.avg(Measurement.humidex).label("avg_humidex"),
                func.max(Measurement.humidex).label("max_humidex"),
            )
            .filter(
                Measurement.station_id == station_id,
                Measurement.timestamp >= start,
                Measurement.timestamp <= end,
            )
            .group_by(func.date(Measurement.timestamp))
            .order_by(func.date(Measurement.timestamp).asc())
            .all()
        )

        if not rows:
            raise HTTPException(
                status_code=404,
                detail=f"No data for station {station_id} in range {from_date}–{to_date}",
            )

        from backend.services.humidex_service import humidex_to_risk_level

        daily = [
            {
                "date": str(row.day),
                "avg_temp": round(float(row.avg_temp), 2) if row.avg_temp else None,
                "avg_humidity": round(float(row.avg_humidity), 2) if row.avg_humidity else None,
                "avg_humidex": round(float(row.avg_humidex), 2) if row.avg_humidex else None,
                "max_humidex": round(float(row.max_humidex), 2) if row.max_humidex else None,
                "risk_level": humidex_to_risk_level(float(row.avg_humidex)) if row.avg_humidex else "LOW",
            }
            for row in rows
        ]

        return {
            "station_id": station_id,
            "from": from_date,
            "to": to_date,
            "daily": daily,
        }

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {str(exc)}")
