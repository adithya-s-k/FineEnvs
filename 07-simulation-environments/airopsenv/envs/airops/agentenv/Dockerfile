# AirOpsEnv v1 as an AgentEnv environment: airops-day, one hub desk's storm window, with the b6-v1 pack, on port 18765.
# Build context: envs/airops (core/, agentenv/ and tasks/ side by side), with the pack fetched first
# (tools/fetch_packs.py); Dockerfile.dockerignore keeps the rest out.
#   docker build -f agentenv/Dockerfile -t airops-day-agentenv .
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/adithya-s-k/FineEnvs" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.description="AirOpsEnv v1 as an AgentEnv environment. Built from public U.S. records: BTS on-time and T-100 data, the FAA registry, FAA Command Center advisories and NWS METARs, TAFs and forecast discussions via the Iowa Environmental Mesonet."
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 AIROPS_TASKS_DIR=/app/tasks/b6-v1
# The versions the published runs were recorded with. v1's day needs nothing beyond the standard library.
RUN pip install "agentenv-framework-protocol==0.1.338" "fastmcp==3.4.8" "mcp==1.30.0" "pydantic==2.13.5" "tzdata"
WORKDIR /app
COPY core /app/core
COPY agentenv/pyproject.toml /app/agentenv/pyproject.toml
COPY agentenv/airops_agentenv /app/agentenv/airops_agentenv
COPY tasks/b6-v1 /app/tasks/b6-v1
RUN pip install --no-deps /app/core /app/agentenv && useradd --create-home --uid 1000 airops
USER airops
EXPOSE 18765
CMD ["python", "-m", "airops_agentenv.server"]
