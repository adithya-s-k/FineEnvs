# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The check that stops the whole thing silently breaking.

Three descriptions of one tool surface have to agree: `tools.py` (the registry), the client's public
methods (what TRL puts in the model's schema), and the server's registered FastMCP tools (what can
actually be invoked). If they drift, the model is offered a tool nobody implements -- it calls it,
gets an error, and the run reads as a policy that cannot use tools. Nothing else would report it.
"""

import pytest

# Absolute, not relative: the environment directory is hyphenated and therefore not a
# Python package. It is importable as `whitebox_bash` (see train/_pypath), which is also
# the name it installs under, so tests exercise the same import path users get.
from whitebox_bash import exposed_tool_names, tool_names


# Server-only plumbing: minted before the agent exists and called after it stops, so they are
# deliberately NOT in the model's schema.
SERVER_ONLY = {"start_episode", "grade"}


@pytest.mark.parametrize("selection", [None, "all", "bash", "bash,seta", "seta"])
def test_client_surface_matches_registry(selection):
    assert exposed_tool_names(selection) == tuple(sorted(tool_names(selection)))


@pytest.mark.parametrize("selection", [None, "all", "bash", "bash,seta"])
def test_server_implements_every_client_tool(selection):
    from whitebox_bash.server.environment import WhiteBoxBashEnvironment

    env = WhiteBoxBashEnvironment()
    served = set(env.get_callables()) - SERVER_ONLY
    exposed = set(exposed_tool_names(selection))
    missing = exposed - served
    assert not missing, f"client exposes tools the server does not implement: {sorted(missing)}"


def test_every_served_tool_is_reachable_from_some_selection():
    """The inverse: a server tool no selection exposes is dead code, and dead code rots."""
    from whitebox_bash.server.environment import WhiteBoxBashEnvironment

    env = WhiteBoxBashEnvironment()
    served = set(env.get_callables()) - SERVER_ONLY
    reachable = set(exposed_tool_names("all"))
    assert not (served - reachable), f"unreachable server tools: {sorted(served - reachable)}"
