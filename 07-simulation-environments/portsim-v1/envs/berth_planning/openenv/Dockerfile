# Build context: envs/berth_planning (core/, openenv/ and tasks/ side by side) plus the data CSV is not needed at
# runtime - tasks are prebuilt in tasks/dock-v1-eval and tasks/dock-v1-train.
#   docker build -f openenv/Dockerfile -t berth-planning .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user PORT=8000 ENABLE_WEB_INTERFACE=true \
    BERTH_TASKS_DIR=/app/tasks/dock-v1-eval:/app/tasks/dock-v1-train BERTH_RUNS_DIR=/app/runs
WORKDIR /app
COPY core /app/core
COPY openenv /app/openenv
COPY tasks /app/tasks
RUN pip install /app/core /app/openenv \
    && useradd --create-home --uid 1000 user && mkdir -p /app/runs && chown -R user:user /app
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
CMD ["sh", "-c", "uvicorn berth_openenv.server:app --host 0.0.0.0 --port ${PORT:-8000} --ws-ping-timeout 600"]
