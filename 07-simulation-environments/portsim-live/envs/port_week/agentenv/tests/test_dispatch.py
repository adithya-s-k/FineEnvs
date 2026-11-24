"""`python -m port_week_agentenv.server`, as the image runs it: the env ENVIRONMENT_NAME names, portsim-live when it is
unset, and a one-line exit for a name it doesn't serve or a wind or lines pack that isn't built."""

import os
import subprocess
import sys

import httpx
import pytest
from serving import served
from wind_fixture import tasks_dir

TOOLS = {"get_situation", "check_plan", "confirm_berths", "advance", "port_notice"}
LINE_TOOLS = {"ship_status", "agree_arrival"}


@pytest.mark.parametrize(("environment_name", "card_name"), [(None, "portsim-live"), ("portsim-live", "portsim-live"),
                                                             ("portsim-marine", "portsim-marine")])
def test_main_serves_the_env_agent_env_names(environment_name, card_name):
    with served(environment_name) as base:
        card = httpx.get(f"{base}/.well-known/agent-env.json").json()
    assert (card["name"], {t["name"] for t in card["capabilities"]["tools"]}) == (card_name, TOOLS)


def test_main_serves_the_lines_env_with_the_agents_two_tools(tmp_path):
    with served("portsim-lines", tasks_dir(tmp_path)) as base:
        card = httpx.get(f"{base}/.well-known/agent-env.json").json()
    assert (card["name"], {t["name"] for t in card["capabilities"]["tools"]}) == ("portsim-lines", TOOLS | LINE_TOOLS)


def exits(environ: dict) -> subprocess.CompletedProcess:
    environ = {k: v for k, v in os.environ.items() if k not in ("ENVIRONMENT_NAME", "PORT_WEEK_TASKS_DIR")} | environ
    return subprocess.run([sys.executable, "-m", "port_week_agentenv.server"], env=environ, capture_output=True,
                          text=True, timeout=60)


def test_an_env_it_doesnt_serve_exits_naming_the_four():
    out = exits({"ENVIRONMENT_NAME": "portsim", "MCP_HOST": "127.0.0.1", "MCP_PORT": "1"})
    assert (out.returncode, out.stderr) == (1, "ENVIRONMENT_NAME 'portsim' is not an env this server serves: "
                                               "portsim-live, portsim-marine, portsim-wind, portsim-lines\n")


def test_the_wind_env_without_its_built_pack_exits_naming_the_build(tmp_path):
    out = exits({"ENVIRONMENT_NAME": "portsim-wind", "PORT_WEEK_TASKS_DIR": str(tmp_path), "MCP_HOST": "127.0.0.1",
                 "MCP_PORT": "1"})
    assert (out.returncode, out.stderr) == (
        1, f"wind weeks not built ({tmp_path / 'dock-v1-wind'} is missing): run python tools/build_wind.py in "
           "07-simulation-environments/portsim-live/envs/port_week\n")


def test_the_lines_env_without_its_built_pack_exits_naming_the_build(tmp_path):
    out = exits({"ENVIRONMENT_NAME": "portsim-lines", "PORT_WEEK_TASKS_DIR": str(tmp_path), "MCP_HOST": "127.0.0.1",
                 "MCP_PORT": "1"})
    assert (out.returncode, out.stderr) == (
        1, f"lines weeks not built ({tmp_path / 'dock-v1-lines'} is missing): run python tools/build_lines.py in "
           "07-simulation-environments/portsim-live/envs/port_week, after tools/build_wind.py\n")
