"""Measurement ORM model."""

from sqlalchemy import Column, DateTime, Float, Integer, String

from backend.db.session import Base


class Measurement(Base):
    """A single environmental reading (temperature/humidity/humidex) for a station."""

    __tablename__ = "measurements"

    id = Column(Integer, primary_key=True, index=True)
    station_id = Column(Integer, nullable=False, index=True)
    timestamp = Column(DateTime(timezone=True), nullable=False, index=True)
    temperature = Column(Float, nullable=False)
    humidity = Column(Float, nullable=False)
    humidex = Column(Float, nullable=False)
    comfort_class = Column(String, nullable=False)
    source = Column(String, nullable=False, index=True)
