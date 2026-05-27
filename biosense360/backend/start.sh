#!/bin/sh
# Render startup script — seeds the database then starts the API server.
set -e

echo "=== BioSense360 — startup ==="
echo "Running database seed..."
python -m backend.db.seed

echo "Starting uvicorn on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
