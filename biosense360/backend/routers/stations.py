"""
Stations router — GET /api/stations, GET /api/stations/{id}
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.models.station import Station
from backend.schemas.station import StationOut

router = APIRouter(prefix="/stations", tags=["stations"])


@router.get("", response_model=list[StationOut])
def list_stations(db: Session = Depends(get_db)) -> list[StationOut]:
    """Return all active stations."""
    try:
        stations = db.query(Station).filter(Station.active == True).all()  # noqa: E712
        return stations
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)}")


@router.get("/{station_id}", response_model=StationOut)
def get_station(station_id: int, db: Session = Depends(get_db)) -> StationOut:
    """Return a single station by ID."""
    try:
        station = db.query(Station).filter(Station.id == station_id).first()
        if station is None:
            raise HTTPException(status_code=404, detail=f"Station {station_id} not found")
        return station
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {str(exc)}")
