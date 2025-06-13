# ----------------------------
# 1) Builder Stage: compile wheels
# ----------------------------
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir=/wheels -r requirements.txt

# ----------------------------
# 2) Runtime Stage: minimal image
# ----------------------------
FROM python:3.10-slim

WORKDIR /app

# Install only the runtime libsndfile
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*

# Copy your application code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Default envs (override in docker-compose)
ENV REDIS_URL=redis://redis:6379/0 \
    CELERY_BROKER_URL=${REDIS_URL} \
    PYTHONPATH=/app/src:/app/workers

# Entrypoint: run the API (override to run the worker)
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]