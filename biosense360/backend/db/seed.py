"""
Database seed script for BioSense360.

Loads real Météo France 3-hourly measurements for 2025 (5 stations around
southern France) from a pre-filtered CSV included with the backend.  Comfort
class is assigned using OHCOW 2022 humidex thresholds — the same classification
the XGBoost model learned from this data.

Two synthetic indoor stations (Neusta Office / Lab) are also seeded to keep
the indoor-monitoring demo functional; they use a realistic seasonal temperature
model calibrated to Toulouse 2025 climate.

Run directly:
    python -m backend.db.seed
Inside Docker:
    docker exec biosense_backend python -m backend.db.seed
"""

import math
import os
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

# Allow running as a top-level script from the backend/ or project root directory.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from sqlalchemy.orm import Session

from backend.db.session import SessionLocal, create_all_tables
from backend.models.measurement import Measurement
from backend.models.risk_period import RiskPeriod
from backend.models.station import Station

# ---------------------------------------------------------------------------
# Seed CSV paths — tried in order, first match wins
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent

DATA_CSV_CANDIDATES: list[Path] = [
    _HERE.parent / "data" / "meteo_france_seed.csv.gz",                            # container / local backend/data/
    Path("/project/backend/data/meteo_france_seed.csv.gz"),                         # explicit container path
    Path("/Users/inesamovsesyan/BioBot-Neusta-Comfort/biosense360/backend/data/meteo_france_seed.csv.gz"),  # dev machine
]

# ---------------------------------------------------------------------------
# Station display names and types (mapped by Météo France station_id)
# ---------------------------------------------------------------------------
METEO_STATION_META: dict[int, dict] = {
    7630: {"name": "Toulouse Blagnac",         "type": "outdoor"},
    7510: {"name": "Bordeaux Mérignac",        "type": "outdoor"},
    7621: {"name": "Tarbes-Lourdes Pyrénées",  "type": "outdoor"},
    7643: {"name": "Montpellier Aéroport",     "type": "outdoor"},
    7690: {"name": "Nice Côte d'Azur",         "type": "outdoor"},
}

# Synthetic indoor stations — no real Météo France sensor coverage
INDOOR_STATIONS: list[dict] = [
    {"name": "Neusta Office Indoor", "lat": 43.6085, "lon": 1.4465, "type": "indoor"},
    {"name": "Neusta Lab Indoor",    "lat": 43.6085, "lon": 1.4465, "type": "indoor"},
]

# 2025 monthly mean temperatures for Toulouse (°C) — used for indoor seasonal model
TOULOUSE_MONTHLY_MEAN_C = [7.2, 8.5, 11.8, 14.5, 18.7, 23.1, 26.4, 26.1, 21.8, 16.2, 10.3, 7.5]

# Indoor office parameters: A/C keeps interior ~4 °C cooler, humidity ~10 % lower
INDOOR_TEMP_OFFSET  = -4.0
INDOOR_HUMID_OFFSET = -10.0

random.seed(42)


# ---------------------------------------------------------------------------
# OHCOW 2022 humidex comfort classification
# ---------------------------------------------------------------------------

def humidex_comfort_class(humidex: float) -> str:
    """Assign OHCOW 2022 comfort class from humidex value."""
    if humidex < 30.0:
        return "Comfortable"
    elif humidex < 40.0:
        return "Caution"
    elif humidex < 45.0:
        return "Extreme Caution"
    else:
        return "Danger"


def risk_level_from_avg(avg_humidex: float) -> str:
    """Map average daily humidex to a categorical risk level for RiskPeriod."""
    if avg_humidex >= 45.0:
        return "HIGH"
    elif avg_humidex >= 40.0:
        return "MODERATE"
    elif avg_humidex >= 30.0:
        return "LOW"
    else:
        return "LOW"


def compute_humidex(temp_c: float, rel_humidity: float) -> float:
    """Masterton & Richardson 1979 humidex formula."""
    if temp_c < 0:
        return temp_c
    dew_approx = temp_c - ((100.0 - rel_humidity) / 5.0)
    e = 6.105 * math.exp(25.22 * (dew_approx - 273.16) / (dew_approx) if dew_approx > 0
                         else 17.67 * dew_approx / (dew_approx + 243.5))
    return temp_c + 0.5555 * (e - 10.0)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_seed_csv() -> pd.DataFrame:
    """Locate and load the Météo France seed CSV."""
    for path in DATA_CSV_CANDIDATES:
        if path.exists():
            print(f"  Loading seed CSV from: {path}")
            df = pd.read_csv(path)
            df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
            return df
    raise FileNotFoundError(
        "meteo_france_seed.csv.gz not found. Expected at:\n"
        + "\n".join(f"  {p}" for p in DATA_CSV_CANDIDATES)
    )


# ---------------------------------------------------------------------------
# Hourly slot classification (for RiskPeriod.favorable_hours / dangerous_hours)
# ---------------------------------------------------------------------------

def compute_hourly_slots(
    measurements_by_hour: dict[int, float],
) -> tuple[list[str], list[str]]:
    favorable, dangerous = [], []
    for hour, hx in sorted(measurements_by_hour.items()):
        label = f"{hour:02d}:00"
        if hx < 30.0:
            favorable.append(label)
        elif hx >= 45.0:
            dangerous.append(label)
    return favorable, dangerous


# ---------------------------------------------------------------------------
# Station seeding
# ---------------------------------------------------------------------------

def seed_stations(db: Session, df: pd.DataFrame) -> dict[int, Station]:
    """
    Create Station rows for each Météo France station in the CSV + indoor synthetic.
    Returns a dict mapping meteo station_id → ORM Station object.
    """
    existing = {s.name: s for s in db.query(Station).all()}
    id_map: dict[int, Station] = {}

    for mid, meta in METEO_STATION_META.items():
        subset = df[df["station_id"] == mid]
        if subset.empty:
            continue
        lat = float(subset["latitude"].iloc[0])
        lon = float(subset["longitude"].iloc[0])
        name = meta["name"]

        if name in existing:
            id_map[mid] = existing[name]
        else:
            s = Station(name=name, lat=lat, lon=lon, type=meta["type"], active=True)
            db.add(s)
            db.flush()
            id_map[mid] = s

    for indoor in INDOOR_STATIONS:
        name = indoor["name"]
        if name in existing:
            pass  # already in DB from a previous seed
        else:
            s = Station(
                name=name,
                lat=indoor["lat"],
                lon=indoor["lon"],
                type=indoor["type"],
                active=True,
            )
            db.add(s)
            db.flush()
            existing[name] = s

    db.commit()
    print(f"  Stations ready: {len(id_map) + len(INDOOR_STATIONS)}")
    return id_map


# ---------------------------------------------------------------------------
# Outdoor station seeding from CSV
# ---------------------------------------------------------------------------

def seed_outdoor_measurements(
    db: Session,
    df: pd.DataFrame,
    station_map: dict[int, Station],
) -> None:
    """Insert Measurement and RiskPeriod rows from real Météo France data."""
    total = 0

    for mid, station in station_map.items():
        subset = df[df["station_id"] == mid].copy()
        subset = subset.sort_values("timestamp_utc")
        print(f"  Seeding {station.name}: {len(subset)} hourly records")

        # Group by date for RiskPeriod aggregation
        subset["date_str"] = subset["timestamp_utc"].dt.date.astype(str)

        measurements: list[Measurement] = []

        for day_str, group in subset.groupby("date_str"):
            day_date = datetime.strptime(day_str, "%Y-%m-%d").date()
            daily_hx: dict[int, float] = {}
            daily_temps: list[float] = []
            daily_hx_vals: list[float] = []

            for _, row in group.iterrows():
                ts = row["timestamp_utc"].to_pydatetime()
                temp = float(row["temperature_c"]) if pd.notna(row["temperature_c"]) else 20.0
                humid = float(row["relative_humidity_pct"]) if pd.notna(row["relative_humidity_pct"]) else 60.0
                hx = float(row["humidex_c"]) if pd.notna(row["humidex_c"]) else compute_humidex(temp, humid)
                comfort = humidex_comfort_class(hx)
                hour = ts.hour

                m = Measurement(
                    station_id=station.id,
                    timestamp=ts,
                    temperature=round(temp, 2),
                    humidity=round(humid, 2),
                    humidex=round(hx, 2),
                    comfort_class=comfort,
                    source="meteo_france",
                )
                measurements.append(m)
                daily_hx[hour] = hx
                daily_temps.append(temp)
                daily_hx_vals.append(hx)
                total += 1

            # Flush in batches to avoid memory pressure
            if len(measurements) >= 500:
                db.bulk_save_objects(measurements)
                measurements = []

            # Daily aggregation → RiskPeriod
            avg_hx = round(sum(daily_hx_vals) / len(daily_hx_vals), 2)
            avg_tmp = round(sum(daily_temps) / len(daily_temps), 2)
            favorable, dangerous = compute_hourly_slots(daily_hx)

            existing_rp = (
                db.query(RiskPeriod)
                .filter(RiskPeriod.station_id == station.id, RiskPeriod.date == day_date)
                .first()
            )
            if not existing_rp:
                db.add(RiskPeriod(
                    station_id=station.id,
                    date=day_date,
                    risk_level=risk_level_from_avg(avg_hx),
                    favorable_hours=favorable,
                    dangerous_hours=dangerous,
                    avg_humidex=avg_hx,
                    avg_temp=avg_tmp,
                ))

        if measurements:
            db.bulk_save_objects(measurements)
        db.commit()

    print(f"  Total outdoor measurement rows inserted: {total}")


# ---------------------------------------------------------------------------
# Indoor synthetic seeding  (Neusta Office + Lab — full year 2025)
# ---------------------------------------------------------------------------

def _indoor_temp(month: int, hour: int, indoor_offset: float) -> float:
    """
    Seasonal + diurnal temperature model for an air-conditioned Toulouse office.
    Monthly mean follows Toulouse 2025 climate; diurnal range ±4 °C.
    """
    monthly_mean = TOULOUSE_MONTHLY_MEAN_C[month - 1] + indoor_offset
    diurnal = 4.0 * math.cos((hour - 14) * math.pi / 12.0)
    noise = random.gauss(0, 0.8)
    return round(max(15.0, min(35.0, monthly_mean + diurnal + noise)), 2)


def _indoor_humid(temp: float, outdoor_humid_mid: float = 55.0) -> float:
    """Indoor humidity: inverse of temp, constrained by A/C dehumidification."""
    base = outdoor_humid_mid + INDOOR_HUMID_OFFSET
    variation = (28.0 - temp) * 0.6
    noise = random.gauss(0, 2.0)
    return round(max(25.0, min(70.0, base + variation + noise)), 2)


def seed_indoor_measurements(db: Session) -> None:
    """Generate full-year 2025 synthetic data for the two indoor Neusta stations."""
    indoor_stations = (
        db.query(Station)
        .filter(Station.type == "indoor")
        .all()
    )
    if not indoor_stations:
        print("  No indoor stations found — skipping indoor seed.")
        return

    total = 0
    year = 2025

    for station in indoor_stations:
        print(f"  Seeding indoor station: {station.name}")
        measurements: list[Measurement] = []

        for month in range(1, 13):
            import calendar
            days_in_month = calendar.monthrange(year, month)[1]
            for day in range(1, days_in_month + 1):
                day_date = date(year, month, day)
                daily_hx: dict[int, float] = {}
                daily_temps: list[float] = []
                daily_hx_vals: list[float] = []

                for hour in range(0, 24, 3):  # 3-hourly to match outdoor resolution
                    ts = datetime(year, month, day, hour, 0, 0, tzinfo=timezone.utc)
                    temp = _indoor_temp(month, hour, INDOOR_TEMP_OFFSET)
                    humid = _indoor_humid(temp)
                    hx = compute_humidex(temp, humid)
                    comfort = humidex_comfort_class(hx)

                    m = Measurement(
                        station_id=station.id,
                        timestamp=ts,
                        temperature=temp,
                        humidity=humid,
                        humidex=round(hx, 2),
                        comfort_class=comfort,
                        source="neusta",
                    )
                    measurements.append(m)
                    daily_hx[hour] = hx
                    daily_temps.append(temp)
                    daily_hx_vals.append(hx)
                    total += 1

                if len(measurements) >= 500:
                    db.bulk_save_objects(measurements)
                    measurements = []

                avg_hx = round(sum(daily_hx_vals) / len(daily_hx_vals), 2)
                avg_tmp = round(sum(daily_temps) / len(daily_temps), 2)
                favorable, dangerous = compute_hourly_slots(daily_hx)

                existing_rp = (
                    db.query(RiskPeriod)
                    .filter(RiskPeriod.station_id == station.id, RiskPeriod.date == day_date)
                    .first()
                )
                if not existing_rp:
                    db.add(RiskPeriod(
                        station_id=station.id,
                        date=day_date,
                        risk_level=risk_level_from_avg(avg_hx),
                        favorable_hours=favorable,
                        dangerous_hours=dangerous,
                        avg_humidex=avg_hx,
                        avg_temp=avg_tmp,
                    ))

        if measurements:
            db.bulk_save_objects(measurements)
        db.commit()

    print(f"  Total indoor measurement rows inserted: {total}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_seed() -> None:
    """Creates tables then seeds all stations with real + synthetic data."""
    print("BioSense360 — Database Seed (Real Météo France 2025)")
    print("=" * 55)

    print("Creating tables if needed...")
    create_all_tables()

    db: Session = SessionLocal()
    try:
        existing_count = db.query(Measurement).count()
        if existing_count > 0:
            # Check if data already covers the full 2025 year with real data
            latest = db.query(RiskPeriod).order_by(RiskPeriod.date.desc()).first()
            if latest and str(latest.date) >= "2025-12-30":
                print(
                    f"Database already has {existing_count:,} measurements "
                    f"up to {latest.date} — skipping seed."
                )
                return
            print(
                f"Database exists ({existing_count:,} rows, up to "
                f"{latest.date if latest else '?'}) but not fully seeded. "
                "Re-seeding..."
            )
            db.query(Measurement).delete()
            db.query(RiskPeriod).delete()
            db.query(Station).delete()
            db.commit()

        # Load CSV
        print("Loading Météo France seed data...")
        df = load_seed_csv()
        print(f"  Loaded {len(df):,} rows across {df['station_id'].nunique()} stations")

        # Seed stations
        print("Seeding stations...")
        station_map = seed_stations(db, df)

        # Seed outdoor real measurements
        print("Seeding outdoor measurements (real Météo France 2025)...")
        seed_outdoor_measurements(db, df, station_map)

        # Seed indoor synthetic
        print("Seeding indoor measurements (synthetic Toulouse 2025 seasonal model)...")
        seed_indoor_measurements(db)

        print("=" * 55)
        total = db.query(Measurement).count()
        print(f"Seed complete — {total:,} total measurements in DB.")
        print("Default dashboard date: 2025-08-12 (peak heat — Toulouse Blagnac)")

    finally:
        db.close()


if __name__ == "__main__":
    run_seed()
