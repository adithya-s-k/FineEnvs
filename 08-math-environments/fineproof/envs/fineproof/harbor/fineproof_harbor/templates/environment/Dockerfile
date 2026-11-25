# The agent's sandbox: Python with sympy for scratch work. The proof goes to /workspace/proof.md, which task.toml
# declares as the one artifact the separate verifier receives.
# python:3.12-slim, pinned (multi-arch index digest, 2026-11-05) so every sandbox gets the same image.
FROM python:3.12-slim@sha256:b18b261e1b4974b4f321451ad5251ded85abb22c2b8b262da527dda1d031ab30
RUN pip install --no-cache-dir sympy==1.13.3 numpy==2.1.3
RUN mkdir -p /workspace
WORKDIR /workspace
