"""
BioSense360 — FastAPI application entry point.

Registers all routers under the /api prefix and configures CORS,
startup event handlers, and API documentation metadata.

Run locally:
    uvicorn main:app --reload --port 8000

Or via Docker Compose:
    docker compose up backend
"""

import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------
app = FastAPI(
    title="BioSense360 API",
    description=(
        "Thermal Comfort Monitoring Platform for Toulouse. "
        "Humidex-based risk assessment, ML comfort prediction, "
        "and AI-powered safety briefings."
    ),
    version="3.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# ---------------------------------------------------------------------------
# CORS — open for all origins in development / Docker Compose
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: ensure database tables exist
# ---------------------------------------------------------------------------
@app.on_event("startup")
def on_startup() -> None:
    """
    Create all ORM-defined tables if they do not yet exist.
    This is a dev convenience; production should use Alembic migrations.
    """
    from backend.db.session import create_all_tables
    create_all_tables()


# ---------------------------------------------------------------------------
# Router registration — all prefixed under /api
# ---------------------------------------------------------------------------
from backend.routers import (  # noqa: E402
    ai_advisor,
    forecast,
    humidex,
    ml_info,
    plants,
    recommendations,
    risk,
    stations,
    trends,
)

app.include_router(stations.router,        prefix="/api")
app.include_router(humidex.router,         prefix="/api")
app.include_router(risk.router,            prefix="/api")
app.include_router(forecast.router,        prefix="/api")
app.include_router(trends.router,          prefix="/api")
app.include_router(recommendations.router, prefix="/api")
app.include_router(ai_advisor.router,      prefix="/api")
app.include_router(ml_info.router,         prefix="/api")
app.include_router(plants.router,          prefix="/api")


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Liveness probe — returns OK if the application is running."""
    return {"status": "ok", "service": "BioSense360", "version": "3.0.0"}


@app.get("/", tags=["root"])
def root() -> dict:
    """API root — redirects docs URL."""
    return {
        "message": "BioSense360 API",
        "docs": "/docs",
        "version": "3.0.0",
    }
