"""
Database session management for BioSense360.

Uses SQLAlchemy 2.x with a connection pool appropriate for the FastAPI
async request lifecycle. The session is provided to route handlers via
FastAPI's dependency injection system.
"""

import os
from typing import Generator

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

load_dotenv()

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://user:pass@localhost:5432/biosense",
)

# Create synchronous engine — psycopg2 driver, connection pool size suitable
# for a development/demo deployment (5 connections + 10 overflow).
engine = create_engine(
    DATABASE_URL,
    pool_size=5,
    max_overflow=10,
    pool_pre_ping=True,  # validates connections before checkout
    echo=False,          # set True for SQL debug logging
)

SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Shared declarative base — all ORM models inherit from this."""
    pass


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that yields a database session and guarantees cleanup.

    Usage in a route:
        @router.get("/example")
        def example(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_all_tables() -> None:
    """
    Create all tables defined by ORM models.
    Called once at application startup if tables do not already exist.
    Prefer Alembic migrations for production; this is a dev convenience.
    """
    # Import all models so their metadata is registered before create_all.
    import backend.models  # noqa: F401 — side-effect import
    Base.metadata.create_all(bind=engine)
