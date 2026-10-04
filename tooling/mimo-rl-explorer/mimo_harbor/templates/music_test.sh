#!/bin/bash
# Music grading: Xiaomi's scorer (abc2midi + 18 human-likeness features, with its validity gate). abc2midi is
# installed only now, at the version the explorer uses, so the agent could not have run it while composing.
mkdir -p /logs/verifier
if ! command -v abc2midi >/dev/null 2>&1; then
  apt-get update -qq >/dev/null && apt-get install -y -qq --no-install-recommends @@ABCMIDI@@ >/dev/null \
    || { echo "could not install abcmidi (testbed problem, not scored)" >&2; exit 1; }
fi
cd /tests && python3 /tests/grade.py
