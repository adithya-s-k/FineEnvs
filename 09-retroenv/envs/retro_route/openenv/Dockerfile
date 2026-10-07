# Build context: envs/retro_route (core/ and openenv/ side by side), which is
# also the layout deploy.py uploads to a Space.
#   docker build -f openenv/Dockerfile -t retroenv .
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user \
    PORT=8000 ENABLE_WEB_INTERFACE=true RETROENV_PREPARED_DIR=/app/prepared \
    RETROENV_TASKS_REPO=LiteFold/RetroEnv
# RDKit's molecule drawing (the playground) links against X11 render libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libexpat1 libfreetype6 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY core /app/core
COPY openenv /app/openenv
RUN pip install /app/core /app/openenv \
    && useradd --create-home --uid 1000 user \
    && mkdir -p /app/prepared && chown -R user:user /app
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
# Tasks are fetched at startup (prepare.py), never baked into the image; override RETROENV_TASKS_REPO to serve another bundle.
CMD ["bash", "/app/openenv/start.sh"]
