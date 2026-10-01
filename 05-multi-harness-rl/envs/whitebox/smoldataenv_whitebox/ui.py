"""Use OpenEnv's playground with connection examples for this environment."""

from openenv.core.env_server.gradio_ui import build_gradio_app

QUICK_START = """
Click **Reset**, then use Type `call_tool`. Start with Tool Name `start_task`
and Arguments `{"split": "test", "index": 0}`. The response contains your task.

Run `bash` with `{"command": "ls /home/user/input"}` to explore its data.
Use `submit_solution` with `{"answer": "your answer"}`, then `grade` with `{}`.
Grade releases the sandbox. Starting another task also closes the previous one.

From Python, use OpenEnv's public client:

```python
from openenv.core.mcp_client import MCPToolClient

with MCPToolClient("https://fineenvs-data-agent-seta-whitebox-env.hf.space").sync() as env:
    env.reset()
    task = env.call_tool("start_task", split="test", index=0)
    print(task)
    print(env.call_tool("bash", command="ls /home/user/input"))
```
"""


def build_ui(manager, fields, metadata, is_chat, title, quick_start):
    return build_gradio_app(
        manager, fields, metadata, is_chat, title=title, quick_start_md=QUICK_START
    )
