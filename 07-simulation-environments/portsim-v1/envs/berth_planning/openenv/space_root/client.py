"""Client for PortSimEnv v1: an OpenEnv MCP client (tools get_situation, check_plan, submit_plan).

    from portsimenv import PortSimEnv, CallToolAction
    env = PortSimEnv("https://fineenvs-portsimenv.hf.space").sync()
    env.reset(task_id="dock-24B-w07x1-busy-0")
    step = env.step(CallToolAction(tool_name="submit_plan", arguments={"plan": plan}))
"""

from openenv.core.mcp_client import MCPToolClient


class PortSimEnv(MCPToolClient):
    """A WebSocket session on a PortSimEnv server; one episode at a time, graded once on submit_plan."""
