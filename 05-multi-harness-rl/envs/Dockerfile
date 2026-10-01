FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends git curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY envs/requirements.txt /app/envs/requirements.txt
RUN pip install --no-cache-dir -r envs/requirements.txt && \
    git clone --depth 1 https://github.com/huggingface/OpenEnv.git /opt/OpenEnv
ENV PYTHONPATH=/opt/OpenEnv/envs PYTHONUNBUFFERED=1 PORT=7860
COPY envs /app/envs
COPY data /app/data
COPY prepare.py /app/prepare.py
RUN useradd --create-home --uid 1000 user && chown -R user:user /app
USER user
ENV HOME=/home/user HF_HOME=/home/user/.cache/huggingface
EXPOSE 7860
CMD ["bash", "envs/start.sh"]
