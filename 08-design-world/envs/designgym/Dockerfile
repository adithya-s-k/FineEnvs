# DesignGym: OpenEnv server + editor + headless Chromium for the GUI surface.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 UV_LINK_MODE=copy PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
WORKDIR /app/designgym
# skia-python links against EGL/GL and fontconfig even when rendering on the CPU.
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libfontconfig1 \
 && rm -rf /var/lib/apt/lists/* \
 && pip install --no-cache-dir uv

# Dependencies first, for layer caching.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --extra daytona --no-install-project

COPY . .
RUN uv sync --frozen --no-dev --extra daytona \
 && .venv/bin/python -m playwright install --with-deps chromium \
 && rm -rf /var/lib/apt/lists/*

# Tasks: the task pack from the Hub (DESIGNGYM_PACK_REPO), downloaded into /data/pack at the first start
# (docker-entrypoint.sh). DESIGNGYM_PACK_SPLITS (e.g. "test") fetches only those splits' designs.
ENV HF_HOME=/tmp/hf DESIGN_CANVAS_CACHE=/tmp/designgym-cache DESIGN_CANVAS_PACK=/data/pack
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
# an upload to the Hub does not keep the executable bit
RUN chmod 0755 /usr/local/bin/docker-entrypoint.sh

RUN useradd --create-home --uid 1000 user && mkdir -p /data && chown user /data
USER user
ENV PATH="/app/designgym/.venv/bin:$PATH" PORT=8000 MAX_CONCURRENT_ENVS=64
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=600s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
ENTRYPOINT ["docker-entrypoint.sh"]
# OpenEnv's entry point: one process, or DESIGN_CANVAS_WORKERS processes behind one URL (server/router.py)
CMD ["server"]
