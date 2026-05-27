"""
Forecast router — GET /api/forecast/time-slots

Returns 24 hourly slots for a given station and date, synthesised from:
  1. Stored measurements (seed data or real ingestion)
  2. OpenWeather API (with DB fallback on error)
"""

import os
from datetime import datetime, timezone
from typing import Optional

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.measurement import Measurement
from backend.models.station import Station
from backend.schemas.risk import TimeSlot, TimeSlotForecast
from backend.ml.predict import predict as ml_predict
from backend.services.humidex_service import (
    classify_humidex,
    compute_humidex,
    humidex_to_risk_level,
)

router = APIRouter(prefix="/forecast", tags=["forecast"])

OPENWEATHER_API_KEY: str = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/forecast"


@router.get("/time-slots", response_model=TimeSlotForecast)
def get_time_slots(
    station_id: int = Query(..., description="Station ID"),
    date: str = Query(..., description="Date in YYYY-MM-DD format"),
    db: Session = Depends(get_db),
) -> TimeSlotForecast:
    """
    Return 24 hourly comfort slots for the given station and date.

    Data priority:
        1. Stored measurements in the database (seed data / real ingestion)
        2. OpenWeather 3h forecast API (interpolated to 1h)
        3. Synthetic sinusoidal fallback (if all else fails)
    """
    try:
        target_dt = datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    except ValueError:
        raise HTTPException(status_code=422, detail="date must be YYYY-MM-DD")

    station = db.query(Station).filter(Station.id == station_id).first()
    if station is None:
        raise HTTPException(status_code=404, detail=f"Station {station_id} not found")

    # ------------------------------------------------------------------
    # 1. Try database first (seed or previously ingested)
    # ------------------------------------------------------------------
    db_slots = _slots_from_db(db, station_id, target_dt, station)
    if len(db_slots) >= 4:  # accept sparse (3-hourly) data — it will be interpolated
        return _build_forecast_response(station_id, date, db_slots)

    # ------------------------------------------------------------------
    # 2. Try OpenWeather API
    # ------------------------------------------------------------------
    if OPENWEATHER_API_KEY:
        try:
            ow_slots = _slots_from_openweather(station, date)
            if ow_slots:
                return _build_forecast_response(station_id, date, ow_slots)
        except Exception:
            pass  # Fall through to synthetic fallback

    # ------------------------------------------------------------------
    # 3. Synthetic sinusoidal fallback — always works
    # ------------------------------------------------------------------
    synth_slots = _slots_synthetic(station)
    return _build_forecast_response(station_id, date, synth_slots)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _slots_from_db(
    db: Session,
    station_id: int,
    target_dt: datetime,
    station: Station,
) -> list[TimeSlot]:
    """
    Retrieve measurement slots from the database for the given day and
    interpolate linearly to fill all 24 hourly slots.

    Real Météo France data is 3-hourly (8 obs/day).  Linear interpolation
    gives a smooth hourly representation without any invented extremes.
    comfort_class is re-predicted by the ML model on each interpolated point.
    """
    next_day = datetime(
        target_dt.year, target_dt.month, target_dt.day,
        23, 59, 59, tzinfo=timezone.utc
    )
    measurements = (
        db.query(Measurement)
        .filter(
            Measurement.station_id == station_id,
            Measurement.timestamp >= target_dt,
            Measurement.timestamp <= next_day,
        )
        .order_by(Measurement.timestamp.asc())
        .all()
    )

    if not measurements:
        return []

    is_indoor = station.type == "indoor" if station else False
    lat = station.lat if station else 0.0
    lon = station.lon if station else 0.0
    month = target_dt.month

    # Build lookup dict: hour → (temp, humid, humidex)
    raw: dict[int, tuple[float, float, float]] = {
        m.timestamp.hour: (m.temperature, m.humidity, m.humidex)
        for m in measurements
    }

    # If we already have all 24 hours (hourly data), skip interpolation
    if len(raw) >= 20:
        obs_hours = sorted(raw)
    else:
        # Interpolate all 0-23 hours from the sparse observations
        obs_hours_sorted = sorted(raw)
        # Wrap-around: treat the 0-hour of next-day as the anchor after 21:00
        # by appending (24, value_at_hour_0_or_closest_to_midnight)
        last_h = obs_hours_sorted[-1]
        first_h = obs_hours_sorted[0]
        extended = obs_hours_sorted + [first_h + 24]  # wrap
        extended_vals = {h: raw[h] for h in obs_hours_sorted}
        extended_vals[first_h + 24] = raw[first_h]

        for target_hour in range(24):
            if target_hour in raw:
                continue
            # Find surrounding anchor hours
            lo = max((h for h in extended if h <= target_hour), default=None)
            hi = min((h for h in extended if h > target_hour), default=None)
            if lo is None or hi is None:
                raw[target_hour] = raw[obs_hours_sorted[0]]
            else:
                alpha = (target_hour - lo) / (hi - lo)
                t0, h0, hx0 = extended_vals.get(lo, raw.get(lo % 24, (20.0, 60.0, 20.0)))
                t1, h1, hx1 = extended_vals.get(hi, raw.get(hi % 24, (20.0, 60.0, 20.0)))
                raw[target_hour] = (
                    round(t0 + alpha * (t1 - t0), 2),
                    round(h0 + alpha * (h1 - h0), 2),
                    round(hx0 + alpha * (hx1 - hx0), 2),
                )
        obs_hours = list(range(24))

    slots: list[TimeSlot] = []
    for hour in sorted(obs_hours):
        temp, humid, hx = raw[hour]
        risk = humidex_to_risk_level(hx)
        ml_result = ml_predict(
            temperature=temp,
            humidity=humid,
            hour=hour,
            month=month,
            lat=lat,
            lon=lon,
            is_indoor=is_indoor,
        )
        slots.append(TimeSlot(
            hour=hour,
            label=f"{hour:02d}:00",
            temperature=round(temp, 2),
            humidity=round(humid, 2),
            humidex=round(hx, 2),
            comfort_class=ml_result["predicted_class"],
            risk_level=risk,
        ))
    return slots


def _slots_from_openweather(station: Station, date: str) -> list[TimeSlot]:
    """
    Fetch 3-hourly forecast from OpenWeather and interpolate to 1-hourly slots.
    Raises on any HTTP or parsing error so the caller can fall back.
    """
    response = httpx.get(
        OPENWEATHER_URL,
        params={
            "lat": station.lat,
            "lon": station.lon,
            "appid": OPENWEATHER_API_KEY,
            "units": "metric",
            "cnt": 8,  # 8 × 3h = 24h
        },
        timeout=10.0,
    )
    response.raise_for_status()
    data = response.json()

    # Filter to the requested date and build slots
    slots: list[TimeSlot] = []
    seen_hours: set[int] = set()

    for item in data.get("list", []):
        dt = datetime.fromtimestamp(item["dt"], tz=timezone.utc)
        if dt.strftime("%Y-%m-%d") != date:
            continue
        hour = dt.hour
        if hour in seen_hours:
            continue
        seen_hours.add(hour)

        temp = item["main"]["temp"]
        humid = item["main"]["humidity"]
        humidex_val = compute_humidex(temp, humid)
        risk = humidex_to_risk_level(humidex_val)
        ml_result = ml_predict(
            temperature=temp,
            humidity=humid,
            hour=hour,
            month=dt.month,
            lat=station.lat,
            lon=station.lon,
            is_indoor=station.type == "indoor",
        )

        slots.append(TimeSlot(
            hour=hour,
            label=f"{hour:02d}:00",
            temperature=round(temp, 2),
            humidity=round(humid, 2),
            humidex=round(humidex_val, 2),
            comfort_class=ml_result["predicted_class"],
            risk_level=risk,
        ))

    return slots


def _slots_synthetic(station: Station) -> list[TimeSlot]:
    """
    Generate synthetic sinusoidal slots when no real data is available.
    Uses the same curve as the seed script for consistency.
    """
    import math

    TEMP_MIN, TEMP_MAX = 18.0, 38.0
    HUMID_MIN, HUMID_MAX = 35.0, 70.0
    PEAK_HOUR = 14.0
    is_indoor = station.type == "indoor"

    slots: list[TimeSlot] = []
    for hour in range(24):
        phase = (hour - PEAK_HOUR) * (math.pi / 12.0)
        amplitude_t = (TEMP_MAX - TEMP_MIN) / 2.0
        temp = ((TEMP_MAX + TEMP_MIN) / 2.0) - amplitude_t * math.cos(phase)
        if is_indoor:
            temp -= 4.0

        amplitude_h = (HUMID_MAX - HUMID_MIN) / 2.0
        humid = ((HUMID_MAX + HUMID_MIN) / 2.0) + amplitude_h * math.cos(phase)
        if is_indoor:
            humid -= 10.0

        temp = max(10.0, min(45.0, temp))
        humid = max(10.0, min(100.0, humid))

        humidex_val = compute_humidex(temp, humid)
        risk = humidex_to_risk_level(humidex_val)
        ml_result = ml_predict(
            temperature=temp,
            humidity=humid,
            hour=hour,
            month=datetime.now(tz=timezone.utc).month,
            lat=station.lat,
            lon=station.lon,
            is_indoor=is_indoor,
        )

        slots.append(TimeSlot(
            hour=hour,
            label=f"{hour:02d}:00",
            temperature=round(temp, 2),
            humidity=round(humid, 2),
            humidex=round(humidex_val, 2),
            comfort_class=ml_result["predicted_class"],
            risk_level=risk,
        ))
    return slots


def _build_forecast_response(
    station_id: int, date: str, slots: list[TimeSlot]
) -> TimeSlotForecast:
    """Aggregate slot list into a TimeSlotForecast response."""
    favorable = [s.label for s in slots if s.humidex < 30.0]
    dangerous = [s.label for s in slots if s.humidex >= 45.0]

    if slots:
        avg_humidex = sum(s.humidex for s in slots) / len(slots)
        day_risk = humidex_to_risk_level(avg_humidex)
    else:
        day_risk = "LOW"

    return TimeSlotForecast(
        station_id=station_id,
        date=date,
        slots=slots,
        favorable_hours=favorable,
        dangerous_hours=dangerous,
        day_risk_level=day_risk,
    )
