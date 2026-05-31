#!/bin/sh
# Render startup script — seeds the database then starts the API server.
set -e

echo "=== BioSense360 — startup ==="

# Run from /project so that 'backend' is importable as a package
cd /project

echo "Running database seed..."
python -m backend.db.seed

echo "Starting uvicorn on port ${PORT:-8000}..."
cd /project/backend
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}"
