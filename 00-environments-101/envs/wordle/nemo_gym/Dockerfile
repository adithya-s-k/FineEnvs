# syntax=docker/dockerfile:1
#
# Wordle NeMo Gym Resources Server
# Pure Python — no E2B or external sandbox needed.
#
# Build (from wordle/ directory):
#   docker build -f nemo_gym/Dockerfile -t wordle-nemo-gym .
#
# Run:
#   docker run -p 11000:11000 wordle-nemo-gym
#
# Dependencies come from uv.lock, which pins the NeMo Gym commit the tests run
# against. NeMo Gym's HEAD needs Python >= 3.13, so an unpinned install fails here.

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc git \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

WORKDIR /app
COPY nemo_gym/pyproject.toml nemo_gym/uv.lock ./
RUN uv sync --frozen --no-install-project --no-cache

FROM python:3.12-slim

WORKDIR /app

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Shared game logic + server
COPY core ./core
COPY nemo_gym/configs ./configs
COPY nemo_gym/server.py .

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11000/health')" || exit 1

EXPOSE 11000
CMD ["python", "server.py"]
