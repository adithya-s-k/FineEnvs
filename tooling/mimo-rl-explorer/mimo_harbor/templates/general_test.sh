#!/bin/bash
# General grading: verl general_agent/environment.py _do_calculate_reward, via the task's own run_verify.py.
mkdir -p /logs/verifier
python3 /tests/grade.py
