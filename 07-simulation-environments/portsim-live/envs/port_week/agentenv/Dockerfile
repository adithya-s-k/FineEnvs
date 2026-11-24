# PortSimEnv live as AgentEnv environments: portsim-live (the default), portsim-marine, portsim-wind and portsim-lines,
# picked by ENVIRONMENT_NAME, on port 18765. Build context: 07-simulation-environments, for berth_core and the dock-v1
# eval and train packs from portsim-v1 (Dockerfile.dockerignore keeps the rest out).
#   docker build -f portsim-live/envs/port_week/agentenv/Dockerfile -t portsim-live-agentenv .
# The image never holds the wind or lines weeks: portsim-wind needs them built (tools/build_wind.py) and mounted
# read-only at /app/port_week/tasks/dock-v1-wind, portsim-lines (tools/build_lines.py) at
# /app/port_week/tasks/dock-v1-lines.
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/adithya-s-k/FineEnvs" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-SA-4.0" \
      org.opencontainers.image.description="PortSimEnv live as AgentEnv environments. Contains data from the Port de Barcelona open data portal."
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BERTH_TASKS_DIR=/app/tasks/dock-v1-eval:/app/tasks/dock-v1-train \
    PORT_WEEK_TASKS_DIR=/app/port_week/tasks
# The versions the published runs were recorded with.
RUN pip install "agentenv-framework-protocol==0.1.290" "fastmcp==3.4.8" "mcp==1.30.0" "pydantic==2.13.5"
WORKDIR /app
COPY portsim-v1/envs/berth_planning/core /app/core
COPY portsim-v1/envs/berth_planning/tasks/dock-v1-eval /app/tasks/dock-v1-eval
COPY portsim-v1/envs/berth_planning/tasks/dock-v1-train /app/tasks/dock-v1-train
COPY portsim-live/envs/port_week/core /app/port_week/core
COPY portsim-live/envs/port_week/agentenv/pyproject.toml /app/port_week/agentenv/pyproject.toml
COPY portsim-live/envs/port_week/agentenv/port_week_agentenv /app/port_week/agentenv/port_week_agentenv
COPY portsim-live/envs/port_week/tasks /app/port_week/tasks
RUN pip install --no-deps /app/core /app/port_week/core /app/port_week/agentenv && useradd --create-home --uid 1000 portsim
USER portsim
EXPOSE 18765
CMD ["python", "-m", "port_week_agentenv.server"]
