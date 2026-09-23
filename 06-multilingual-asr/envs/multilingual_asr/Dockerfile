FROM python:3.12-slim
WORKDIR /app
# libsndfile backs soundfile, which reads the IEEE float WAV FLEURS publishes.
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
  && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev --no-editable
RUN useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
# Overridden by the deployment to the mounted snapshot; the bundled path is a fallback.
ENV ASR_SNAPSHOT=/app/snapshot
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "multilingual_asr.server.app:create_server", "--factory", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets", "--ws-ping-interval", "120", "--ws-ping-timeout", "600"]
