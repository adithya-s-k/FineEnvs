#!/bin/bash
# Webdev grading: a full-page render of dist/index.html, judged by a vision model against the brief.
mkdir -p /logs/verifier
cd /tests && python3 /tests/grade.py
