# PortSimEnv v1 as an AgentEnv environment: the berth-planning MCP server with the dock-v1 eval and train packs, on
# port 18765. Build context: envs/berth_planning (core/, agentenv/ and tasks/ side by side), as for openenv/Dockerfile.
#   docker build -f agentenv/Dockerfile -t portsim-v1-agentenv .
FROM python:3.12-slim
LABEL org.opencontainers.image.source="https://github.com/adithya-s-k/FineEnvs" \
      org.opencontainers.image.licenses="Apache-2.0 AND CC-BY-SA-4.0" \
      org.opencontainers.image.description="PortSimEnv v1 as an AgentEnv environment. Contains data from the Port de Barcelona open data portal."
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    BERTH_TASKS_DIR=/app/tasks/dock-v1-eval:/app/tasks/dock-v1-train
# The versions the published runs were recorded with.
RUN pip install "agentenv-framework-protocol==0.1.290" "fastmcp==3.4.8" "mcp==1.30.0" "pydantic==2.13.5"
WORKDIR /app
COPY core /app/core
COPY agentenv/pyproject.toml /app/agentenv/pyproject.toml
COPY agentenv/berth_agentenv /app/agentenv/berth_agentenv
COPY tasks/dock-v1-eval /app/tasks/dock-v1-eval
COPY tasks/dock-v1-train /app/tasks/dock-v1-train
RUN pip install --no-deps /app/core /app/agentenv && useradd --create-home --uid 1000 portsim
USER portsim
EXPOSE 18765
CMD ["python", "-m", "berth_agentenv.server"]
