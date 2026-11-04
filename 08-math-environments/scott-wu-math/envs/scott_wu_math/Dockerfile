# Scott Wu Math: OpenEnv server + play page. Pure Python, no GPU.
#   docker build -t scott-wu-math . && docker run -p 7860:7860 scott-wu-math
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=7860
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY --chown=user . ./scott_wu_math
RUN pip install --no-cache-dir ./scott_wu_math

USER user
EXPOSE 7860
CMD ["sh", "-c", "uvicorn scott_wu_math.server.app:app --host 0.0.0.0 --port ${PORT} --proxy-headers --forwarded-allow-ips='*'"]
