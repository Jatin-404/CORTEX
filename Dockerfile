FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY cortex ./cortex

RUN pip install --no-cache-dir .

ENV PYTHONUNBUFFERED=1

CMD ["celery", "-A", "cortex.worker.celery_app", "worker", "--loglevel=info", "-Q", "ingestion"]
