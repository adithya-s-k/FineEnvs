# SPDX-License-Identifier: BSD-3-Clause

# This environment renders untrusted JavaScript in Chromium, so it builds on the
# Playwright image rather than on openenv-base: the browser, its system
# libraries and its fonts are already there and matched to the Playwright
# version. The same choice is made by envs/browsergym_env for the same reason.
FROM mcr.microsoft.com/playwright/python:v1.62.0-noble

WORKDIR /app/env

# Build context is envs/watercolour/ so both core/ and openenv/ are included.
COPY . /app/env

RUN pip install --no-cache-dir uv
RUN uv pip install --system --no-cache .

# The image ships browser builds already; this pins the one Playwright resolves
# at runtime to the same location and makes it readable by a non-root Space user.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN python -m playwright install chromium && chmod -R a+rX /ms-playwright

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app/env

# Hugging Face Spaces runs the container as a non-root user, and Chromium wants
# a writable home for its profile and crash dumps.
ENV HOME=/tmp
ENV XDG_CACHE_HOME=/tmp/.cache

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "watercolour_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
