FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY . /app
RUN bash install.sh && useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1 PORT=7860 ENABLE_WEB_INTERFACE=true
EXPOSE 7860
CMD ["bash", "start.sh"]
