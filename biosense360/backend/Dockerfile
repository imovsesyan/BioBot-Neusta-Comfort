FROM python:3.11-slim

# libgomp1 is required by LightGBM; libpq-dev for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Content is copied into /project/backend so that `from backend.xxx` imports resolve
WORKDIR /project/backend

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Make `backend` importable as a top-level package
ENV PYTHONPATH=/project

EXPOSE 8000

RUN chmod +x start.sh
CMD ["./start.sh"]
