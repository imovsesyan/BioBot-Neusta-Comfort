"""RiskPeriod ORM model — daily aggregated humidex risk per station."""

from sqlalchemy import Column, Date, Float, Integer, String
from sqlalchemy.types import JSON

from backend.db.session import Base


class RiskPeriod(Base):
    """Daily risk summary derived from a station's measurements."""

    __tablename__ = "risk_periods"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    risk_level = Column(String, nullable=False)
    favorable_hours = Column(JSON, nullable=False, default=list)
    dangerous_hours = Column(JSON, nullable=False, default=list)
    avg_humidex = Column(Float, nullable=True)
    avg_temp = Column(Float, nullable=True)
