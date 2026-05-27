"""
Risk router — GET /api/risk-summary, GET /api/risk-matrix
"""

from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.measurement import Measurement
from backend.models.risk_period import RiskPeriod
from backend.models.station import Station
from backend.schemas.risk import RiskMatrixCell, RiskMatrixResponse, RiskSummaryResponse
from backend.services.humidex_service import humidex_to_risk_level

router = APIRouter(tags=["risk"])


def _get_date_from_str(date_str: str) -> datetime:
    """Parse a YYYY-MM-DD string into a UTC midnight datetime."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        return d.replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")


@router.get("/risk-summary", response_model=RiskSummaryResponse)
def get_risk_summary(
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    station_id: Optional[int] = Query(None, description="Optional: filter to one station"),
    db: Session = Depends(get_db),
) -> RiskSummaryResponse:
    """
    Return aggregated daily risk summary.

    If station_id is provided, returns risk for that station only.
    Otherwise, aggregates across all stations for the date.
    """
    try:
        target_dt = _get_date_from_str(date)
        target_date = target_dt.date()

        # Try to use pre-computed risk_periods first (fast path)
        rp_query = db.query(RiskPeriod).filter(RiskPeriod.date == target_date)
        if station_id:
            rp_query = rp_query.filter(RiskPeriod.station_id == station_id)

        risk_periods = rp_query.all()

        if risk_periods:
            all_humidex = [rp.avg_humidex for rp in risk_periods if rp.avg_humidex]
            all_temps = [rp.avg_temp for rp in risk_periods if rp.avg_temp]
            avg_humidex = round(sum(all_humidex) / len(all_humidex), 2) if all_humidex else 0.0
            avg_temp = round(sum(all_temps) / len(all_temps), 2) if all_temps else 0.0
            risk_level = humidex_to_risk_level(avg_humidex)

            # Collect favorable/dangerous hours across all stations for the day
            all_favorable: set[str] = set()
            all_dangerous: set[str] = set()
            for rp in risk_periods:
                all_favorable.update(rp.favorable_hours or [])
                all_dangerous.update(rp.dangerous_hours or [])

            stations_breakdown = [
                {
                    "station_id": rp.station_id,
                    "risk_level": rp.risk_level,
                    "avg_humidex": rp.avg_humidex,
                    "avg_temp": rp.avg_temp,
                    "favorable_hours": rp.favorable_hours,
                    "dangerous_hours": rp.dangerous_hours,
                }
                for rp in risk_periods
            ]

            return RiskSummaryResponse(
                date=date,
                station_id=station_id,
                risk_level=risk_level,
                avg_humidex=avg_humidex,
                avg_temp=avg_temp,
                favorable_hours=sorted(all_favorable),
                dangerous_hours=sorted(all_dangerous),
                stations_breakdown=stations_breakdown,
            )

        # Fallback: compute from raw measurements for the requested date
        meas_query = db.query(Measurement).filter(
            Measurement.timestamp >= target_dt,
            Measurement.timestamp < datetime(
                target_dt.year, target_dt.month, target_dt.day, 23, 59, 59,
                tzinfo=timezone.utc
            ),
        )
        if station_id:
            meas_query = meas_query.filter(Measurement.station_id == station_id)

        measurements = meas_query.all()
        if not measurements:
            raise HTTPException(
                status_code=404,
                detail=f"No data found for date={date}" + (f" station_id={station_id}" if station_id else ""),
            )

        humidex_vals = [m.humidex for m in measurements]
        temp_vals = [m.temperature for m in measurements]
        avg_humidex = round(sum(humidex_vals) / len(humidex_vals), 2)
        avg_temp = round(sum(temp_vals) / len(temp_vals), 2)
        risk_level = humidex_to_risk_level(avg_humidex)

        favorable = [f"{m.timestamp.hour:02d}:00" for m in measurements if m.humidex < 30]
        dangerous = [f"{m.timestamp.hour:02d}:00" for m in measurements if m.humidex >= 45]

        return RiskSummaryResponse(
            date=date,
            station_id=station_id,
            risk_level=risk_level,
            avg_humidex=avg_humidex,
            avg_temp=avg_temp,
            favorable_hours=sorted(set(favorable)),
            dangerous_hours=sorted(set(dangerous)),
            stations_breakdown=[],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {str(exc)}")


@router.get("/risk-matrix", response_model=RiskMatrixResponse)
def get_risk_matrix(
    departure: Optional[str] = Query(None, description="Departure station name"),
    arrival: Optional[str] = Query(None, description="Arrival station name"),
    date: Optional[str] = Query(None, description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
) -> RiskMatrixResponse:
    """
    Return a risk matrix for mobility between stations.

    If no departure/arrival filters are given, returns all station pairs.
    Each cell combines the risk of the departure and arrival stations for the day.
    """
    try:
        # Use today if no date provided
        if date is None:
            date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

        target_date = datetime.strptime(date, "%Y-%m-%d").date()

        # Load all stations and their risk periods for the date
        stations = db.query(Station).filter(Station.active == True).all()  # noqa: E712
        station_map = {s.id: s for s in stations}

        risk_periods = (
            db.query(RiskPeriod)
            .filter(RiskPeriod.date == target_date)
            .all()
        )
        risk_by_station: dict[int, RiskPeriod] = {rp.station_id: rp for rp in risk_periods}

        # Filter stations if departure/arrival specified
        if departure:
            dep_stations = [s for s in stations if departure.lower() in s.name.lower()]
        else:
            dep_stations = stations

        if arrival:
            arr_stations = [s for s in stations if arrival.lower() in s.name.lower()]
        else:
            arr_stations = stations

        matrix: list[RiskMatrixCell] = []

        for dep in dep_stations:
            for arr in arr_stations:
                if dep.id == arr.id:
                    continue

                dep_rp = risk_by_station.get(dep.id)
                arr_rp = risk_by_station.get(arr.id)

                # Combined risk = worst of departure and arrival
                risk_levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
                dep_level = dep_rp.risk_level if dep_rp else "MODERATE"
                arr_level = arr_rp.risk_level if arr_rp else "MODERATE"
                combined_level = max(dep_level, arr_level, key=lambda x: risk_levels.index(x))

                dep_h = dep_rp.avg_humidex if dep_rp else 35.0
                arr_h = arr_rp.avg_humidex if arr_rp else 35.0
                avg_h = round((dep_h + arr_h) / 2, 2)

                reco = _trip_recommendation(combined_level, dep.name, arr.name)

                matrix.append(RiskMatrixCell(
                    departure=dep.name,
                    arrival=arr.name,
                    risk_level=combined_level,
                    avg_humidex=avg_h,
                    recommendation=reco,
                ))

        return RiskMatrixResponse(date=date, matrix=matrix)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Error: {str(exc)}")


def _trip_recommendation(risk_level: str, departure: str, arrival: str) -> str:
    """Generate a short trip recommendation string based on risk level."""
    reco_map = {
        "LOW":      f"Trip from {departure} to {arrival} is safe. Stay hydrated.",
        "MODERATE": (
            f"Plan your trip between 06:00–10:00 or 19:00–22:00. "
            "Carry water. Avoid peak heat hours."
        ),
        "HIGH": (
            "High thermal risk during transit. Use air-conditioned transport. "
            "Limit outdoor exposure to under 15 minutes."
        ),
        "CRITICAL": (
            "CRITICAL: Postpone all non-essential travel. "
            "If travel is mandatory, use fully air-conditioned vehicle and "
            "apply HAS 2023 emergency cooling protocol."
        ),
    }
    return reco_map.get(risk_level, "Refer to current thermal advisories.")
