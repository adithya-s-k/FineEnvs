"""Importing portsim_line imports portsim-v1's agentenv/line_agent/agent.py, a script in its image, in this module's
place."""

import importlib.util
import sys
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    __name__, Path(__file__).resolve().parents[5] / "portsim-v1" / "envs" / "berth_planning" / "agentenv" / "line_agent"
    / "agent.py")
sys.modules[__name__] = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sys.modules[__name__])
