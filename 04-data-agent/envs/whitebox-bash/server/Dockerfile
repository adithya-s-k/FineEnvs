FROM python:3.12-slim

WORKDIR /app

# Server extras only. The trainer-side install is deliberately lighter -- see pyproject.toml.
COPY pyproject.toml ./
RUN pip install --no-cache-dir \
      "openenv-core>=0.1" "fastmcp>=2.0" "fastapi>=0.110" "uvicorn>=0.27" \
      "e2b>=1.0" "datasets>=2.19"

COPY . /app/whitebox_bash/
ENV PYTHONPATH=/app

# E2B_API_KEY must be provided at run time. Without it the server starts and then fails every
# episode at sandbox creation, which is worse than refusing to start -- so app.py checks for it.
EXPOSE 8000
CMD ["uvicorn", "whitebox_bash.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
