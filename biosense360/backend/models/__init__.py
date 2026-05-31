"""
ORM model package.

Importing this package registers every model's table metadata on the shared
``Base`` so that ``Base.metadata.create_all()`` (see ``db.session``) can create
all tables. ``db.session.create_all_tables`` relies on this side-effect import.
"""

from backend.models.measurement import Measurement
from backend.models.risk_period import RiskPeriod
from backend.models.station import Station

__all__ = ["Measurement", "RiskPeriod", "Station"]
