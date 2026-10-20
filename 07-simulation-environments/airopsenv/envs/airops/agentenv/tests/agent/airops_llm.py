"""Importing airops_llm imports agents/airops-llm/agent.py, a script in its image, in this module's place."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    __name__, Path(__file__).resolve().parents[2] / "agents" / "airops-llm" / "agent.py")
sys.modules[__name__] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules[__name__])
