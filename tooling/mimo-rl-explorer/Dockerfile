# MiMo RL Environments Explorer as a Docker Space.
#   /data         the private bucket (run records and traces), mounted by the Space
#   /tmp/...      the dataset cache: thousands of small task files, which a bucket mount is slow at
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends abcmidi ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir uv

# Spaces run the container as uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/tmp/hf-home TASK_CACHE_DIR=/tmp/task-cache PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user . .

EXPOSE 7860
CMD ["uv", "run", "--frozen", "--no-dev", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--proxy-headers", "--forwarded-allow-ips", "*"]
