# Build context: envs/retro_route (core/ and openenv/ side by side), which is
# also the layout deploy.py uploads to a Space.
#   docker build -f openenv/Dockerfile -t retroenv .
FROM python:3.12-slim
# The serving index stores RDKit fingerprints and canonical SMILES; the same RDKit lets the
# server use them as built. Another version still works, but recomputes them at startup.
ARG RDKIT_VERSION=2026.3.6
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HOME=/home/user \
    PORT=8000 ENABLE_WEB_INTERFACE=true MAX_CONCURRENT_ENVS=64 \
    RETROENV_CORPUS_MANIFEST=/app/openenv/corpus-manifest.json \
    RETROENV_BUCKET_ROOT=/data \
    RETROENV_PREPARED_DIR=/home/user/prepared
# RDKit's molecule drawing (the playground) links against X11 render libraries.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libxrender1 libxext6 libexpat1 libfreetype6 libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY core /app/core
COPY openenv /app/openenv
RUN pip install "rdkit==${RDKIT_VERSION}" /app/core /app/openenv \
    && useradd --create-home --uid 1000 user \
    && chown -R user:user /app
USER user
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')"
# The image holds code and the pinned corpus-manifest.json only. At startup prepare.py takes
# the snapshot from the bucket mounted at /data, or downloads it when nothing is mounted.
CMD ["bash", "/app/openenv/start.sh"]
