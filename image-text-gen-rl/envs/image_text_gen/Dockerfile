FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends fonts-dejavu-core && rm -rf /var/lib/apt/lists/*
COPY . /app
RUN pip install --no-cache-dir uv && uv sync --frozen --no-dev --no-editable
RUN useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
ENV PATH="/app/.venv/bin:$PATH" HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')"
CMD ["uvicorn", "image_text_gen.server.app:create_server", "--factory", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets", "--ws-max-size", "16777216", "--ws-ping-interval", "120", "--ws-ping-timeout", "600"]
