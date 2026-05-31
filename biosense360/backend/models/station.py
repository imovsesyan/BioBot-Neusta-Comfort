"""Station ORM model."""

from sqlalchemy import Boolean, Column, Float, Integer, String

from backend.db.session import Base


class Station(Base):
    """A measurement station — real Météo France outdoor or synthetic indoor."""

    __tablename__ = "stations"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False, unique=True, index=True)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    type = Column(String, nullable=False, default="outdoor", index=True)
    active = Column(Boolean, nullable=False, default=True)
