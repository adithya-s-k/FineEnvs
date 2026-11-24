# Shared by every openai/math task (agent and separate verifier). See lean_setup.sh.
FROM ubuntu:24.04
ENV ELAN_HOME=/opt/elan \
    PATH=/opt/elan/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    LANG=C.UTF-8
COPY lean_setup.sh fetch_mathlib.sh lakefile.lean lake-manifest.json pack_submission.py /opt/build/
RUN bash /opt/build/lean_setup.sh
WORKDIR /workspace
