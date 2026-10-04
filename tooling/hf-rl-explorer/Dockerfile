# Public HF RL Explorer Docker Space. The admin has a separate private deployment.
#   /data         the private bucket FineEnvs/rl-explorer-data: indexes, packs, rollouts, admin settings; mounted by both
#   /tmp/...      dataset files read while indexing and the tasks rollouts run on: many small files, local disk
#   abcmidi       the MiMo release's music scorer renders a piece with abc2midi (app/mimo/vendor/music_scorer)
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends abcmidi ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/tmp/hf-home RLX_CACHE_DIR=/tmp/rlx-cache TASK_CACHE_DIR=/tmp/rlx-cache/mimo PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .

EXPOSE 7860
ENV RLX_APP=app.main:app
CMD ["sh", "-c", "exec uv run --frozen --no-dev uvicorn \"$RLX_APP\" --host 0.0.0.0 --port 7860 --proxy-headers --forwarded-allow-ips '*'"]
