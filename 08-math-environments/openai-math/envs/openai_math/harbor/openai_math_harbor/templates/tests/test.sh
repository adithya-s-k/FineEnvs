#!/usr/bin/env bash
# Runs in the separate verifier sandbox, which never ran agent code. See grade.py.
exec python3 /tests/grade.py
