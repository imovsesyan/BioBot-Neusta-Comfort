"""
Database seed script for BioSense360.

Generates 30 days × 24 hours of realistic thermal comfort data for 8 stations
around Toulouse, France. Temperature follows a sinusoidal daily curve
(minimum ~18 °C at 06:00, peak ~38 °C at 14:00) with Gaussian noise.
Humidity is the inverse curve (70 % at night, 35 % at peak heat).
Humidex and comfort_class are derived from the Masterton & Richardson 1979 formula.

Run directly:
    python -m backend.db.seed
or inside Docker:
    docker exec biosense_backend python -m backend.db.seed
"""

import math
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone

# Allow running as a top-level script from the backend/ directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

from sqlalchemy.orm import Session

from backend.db.session import SessionLocal, create_all_tables
from backend.models.measurement import Measurement
from backend.models.risk_period import RiskPeriod
from backend.models.station import Station
from backend.services.humidex_service import classify_humidex, compute_humidex

# ---------------------------------------------------------------------------
# Station definitions (Toulouse area, mix of outdoor / indoor)
# ---------------------------------------------------------------------------
STATIONS: list[dict] = [
    {"name": "Toulouse Centre",      "lat": 43.6047, "lon": 1.4442, "type": "outdoor"},
    {"name": "Toulouse Blagnac",     "lat": 43.6290, "lon": 1.3638, "type": "outdoor"},
    {"name": "Toulouse Lardenne",    "lat": 43.5889, "lon": 1.3990, "type": "outdoor"},
    {"name": "Toulouse Rangueil",    "lat": 43.5604, "lon": 1.4647, "type": "outdoor"},
    {"name": "Neusta Office Indoor", "lat": 43.6085, "lon": 1.4465, "type": "indoor"},
    {"name": "Neusta Lab Indoor",    "lat": 43.6085, "lon": 1.4465, "type": "indoor"},
    {"name": "Colomiers Outdoor",    "lat": 43.6117, "lon": 1.3373, "type": "outdoor"},
    {"name": "Balma Outdoor",        "lat": 43.6104, "lon": 1.4990, "type": "outdoor"},
]

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SEED_DAYS: int = 30
# Compute start as 29 days before today so the 30-day window ends today.
_today = date.today()
START_DATE: datetime = datetime(
    (_today - timedelta(days=29)).year,
    (_today - timedelta(days=29)).month,
    (_today - timedelta(days=29)).day,
    0, 0, 0, tzinfo=timezone.utc,
)

# Outdoor daily temperature curve parameters
TEMP_MIN: float = 18.0   # °C at 06:00
TEMP_MAX: float = 38.0   # °C at 14:00
TEMP_PEAK_HOUR: float = 14.0
TEMP_NOISE_STD: float = 1.5   # Gaussian noise std

# Indoor offsets — offices / labs run cooler due to A/C
INDOOR_TEMP_OFFSET: float = -4.0    # Indoor temp ~4 °C cooler
INDOOR_HUMID_OFFSET: float = -10.0  # Indoor humidity ~10 % lower

# Humidity curve parameters (inverse of temperature)
HUMID_MIN: float = 35.0   # % at 14:00
HUMID_MAX: float = 70.0   # % at 06:00

# Random seed for reproducibility
RANDOM_SEED: int = 42
random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def sinusoidal_temperature(hour: int, is_indoor: bool = False) -> float:
    """
    Return a realistic temperature for the given hour using a cosine curve.
    Peak at TEMP_PEAK_HOUR, trough 12 hours earlier (02:00).
    Gaussian noise is added for realism.
    """
    # Phase shift so peak lands at TEMP_PEAK_HOUR
    phase = (hour - TEMP_PEAK_HOUR) * (math.pi / 12.0)
    amplitude = (TEMP_MAX - TEMP_MIN) / 2.0
    midpoint = (TEMP_MAX + TEMP_MIN) / 2.0
    temp = midpoint - amplitude * math.cos(phase)
    temp += random.gauss(0, TEMP_NOISE_STD)
    if is_indoor:
        temp += INDOOR_TEMP_OFFSET
    return round(max(10.0, min(45.0, temp)), 2)


def sinusoidal_humidity(hour: int, is_indoor: bool = False) -> float:
    """
    Return realistic relative humidity for the given hour.
    Inverse of temperature: highest at night / dawn, lowest at peak heat.
    """
    phase = (hour - TEMP_PEAK_HOUR) * (math.pi / 12.0)
    amplitude = (HUMID_MAX - HUMID_MIN) / 2.0
    midpoint = (HUMID_MAX + HUMID_MIN) / 2.0
    humid = midpoint + amplitude * math.cos(phase)
    humid += random.gauss(0, 3.0)
    if is_indoor:
        humid += INDOOR_HUMID_OFFSET
    return round(max(10.0, min(100.0, humid)), 2)


def compute_risk_level(avg_humidex: float) -> str:
    """Map average daily humidex to a categorical risk level."""
    if avg_humidex >= 54.0:
        return "CRITICAL"
    elif avg_humidex >= 45.0:
        return "HIGH"
    elif avg_humidex >= 40.0:
        return "MODERATE"
    else:
        return "LOW"


def compute_hourly_slots(
    measurements_by_hour: dict[int, float],
) -> tuple[list[str], list[str]]:
    """
    Classify each hour of the day as favorable or dangerous based on humidex.
    Returns (favorable_hours, dangerous_hours) as lists of "HH:00" strings.
    """
    favorable, dangerous = [], []
    for hour, humidex_val in sorted(measurements_by_hour.items()):
        label = f"{hour:02d}:00"
        if humidex_val < 30.0:
            favorable.append(label)
        elif humidex_val >= 45.0:
            dangerous.append(label)
    return favorable, dangerous


# ---------------------------------------------------------------------------
# Seeding logic
# ---------------------------------------------------------------------------

def seed_stations(db: Session) -> list[Station]:
    """Insert stations if not already present; return list of Station ORM objects."""
    existing = {s.name: s for s in db.query(Station).all()}
    result = []
    for data in STATIONS:
        if data["name"] in existing:
            result.append(existing[data["name"]])
            continue
        station = Station(
            name=data["name"],
            lat=data["lat"],
            lon=data["lon"],
            type=data["type"],
            active=True,
        )
        db.add(station)
        db.flush()  # get station.id before commit
        result.append(station)
    db.commit()
    print(f"  Stations ready: {len(result)}")
    return result


def seed_measurements(db: Session, stations: list[Station]) -> None:
    """
    Generate 30 days × 24 hours per station of synthetic measurements,
    then aggregate into daily risk_periods.
    """
    total_rows = 0

    for station in stations:
        is_indoor = station.type == "indoor"
        print(f"  Seeding station: {station.name} ({'indoor' if is_indoor else 'outdoor'})")

        for day_offset in range(SEED_DAYS):
            day_date = (START_DATE + timedelta(days=day_offset)).date()
            daily_humidex_by_hour: dict[int, float] = {}
            daily_temps: list[float] = []
            daily_humidex_vals: list[float] = []

            hourly_measurements: list[Measurement] = []

            for hour in range(24):
                ts = datetime(
                    day_date.year, day_date.month, day_date.day,
                    hour, 0, 0, tzinfo=timezone.utc
                )
                temp = sinusoidal_temperature(hour, is_indoor)
                humid = sinusoidal_humidity(hour, is_indoor)
                humidex_val = compute_humidex(temp, humid)
                comfort_cls = classify_humidex(humidex_val)

                m = Measurement(
                    station_id=station.id,
                    timestamp=ts,
                    temperature=temp,
                    humidity=humid,
                    humidex=round(humidex_val, 2),
                    comfort_class=comfort_cls,
                    source="neusta",
                )
                hourly_measurements.append(m)
                daily_humidex_by_hour[hour] = humidex_val
                daily_temps.append(temp)
                daily_humidex_vals.append(humidex_val)
                total_rows += 1

            db.bulk_save_objects(hourly_measurements)

            # Build daily risk_period aggregation
            avg_humidex = round(sum(daily_humidex_vals) / len(daily_humidex_vals), 2)
            avg_temp = round(sum(daily_temps) / len(daily_temps), 2)
            risk_level = compute_risk_level(avg_humidex)
            favorable_hours, dangerous_hours = compute_hourly_slots(daily_humidex_by_hour)

            # Check if risk_period already exists for this station + date
            existing_rp = (
                db.query(RiskPeriod)
                .filter(RiskPeriod.station_id == station.id, RiskPeriod.date == day_date)
                .first()
            )
            if not existing_rp:
                rp = RiskPeriod(
                    station_id=station.id,
                    date=day_date,
                    risk_level=risk_level,
                    favorable_hours=favorable_hours,
                    dangerous_hours=dangerous_hours,
                    avg_humidex=avg_humidex,
                    avg_temp=avg_temp,
                )
                db.add(rp)

        db.commit()

    print(f"  Total measurement rows inserted: {total_rows}")


def run_seed() -> None:
    """Entry point — creates tables then seeds all data."""
    print("BioSense360 — Database Seed")
    print("=" * 40)

    print("Creating tables if needed...")
    create_all_tables()

    db: Session = SessionLocal()
    try:
        # Idempotent: only re-seed if today is not yet covered.
        existing_count = db.query(Measurement).count()
        if existing_count > 0:
            # Check whether today's date is already covered in risk_periods.
            latest = db.query(RiskPeriod).order_by(RiskPeriod.date.desc()).first()
            today_date = date.today()
            if latest and latest.date >= today_date:
                print(
                    f"Database already has {existing_count} measurements and covers up to "
                    f"{latest.date} — skipping seed."
                )
                return
            print(
                f"Database exists but only up to "
                f"{latest.date if latest else 'unknown'}. "
                "Re-seeding all data..."
            )
            # Truncate stale data and re-seed from today's rolling window.
            db.query(Measurement).delete()
            db.query(RiskPeriod).delete()
            db.query(Station).delete()
            db.commit()

        print("Seeding stations...")
        stations = seed_stations(db)

        print("Seeding measurements and risk periods...")
        seed_measurements(db, stations)

        print("=" * 40)
        print("Seed complete.")
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
