# The agent's sandbox: Python for scratch work, and the hook that hands the proof to the verifier.
FROM python:3.12-slim
RUN pip install --no-cache-dir sympy==1.13.3 numpy==2.1.3
COPY collect_proof.py /opt/qed-nano/collect_proof.py
RUN mkdir -p /workspace /logs/artifacts
WORKDIR /workspace
