FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

COPY pyproject.toml README.md openenv.yaml /app/
COPY src /app/src
COPY tasks /app/tasks
COPY stocks /app/stocks

RUN pip install --no-cache-dir .

ENV RETROENV_TASKS_DIR=/app/tasks \
    RETROENV_STOCKS_DIR=/app/stocks \
    RETROENV_DEFAULT_SPLIT=train

EXPOSE 8000
CMD ["uvicorn", "retroenv.server.app:app", "--host", "0.0.0.0", "--port", "8000"]

