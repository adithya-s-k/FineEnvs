"""Use OpenEnv's playground with connection examples for this environment."""

from openenv.core.env_server.gradio_ui import build_gradio_app

QUICK_START = """
This environment runs native OpenCode in Daytona and returns its grade and
training trace. Configure `SANDBOX_VLLM_URL` and `SANDBOX_VLLM_KEY` on the server
before a rollout. The endpoint must return token IDs and logprobs for training.

Click **Reset**, set Type to `call_tool`, and use Tool Name `run_rollout`.
Arguments require `split`, `task_name`, `model` and a `sampling` object.
Use the Task API at `/docs` to find a task name.

From Python:

```python
import requests
from openenv.core.mcp_client import MCPToolClient

url = "https://fineenvs-smoldataenv-multi-harness-opencode.hf.space"
task = requests.post(url + "/smoldataenv_opencode/task",
                     json={"split": "test", "index": 0}).json()["task"]
with MCPToolClient(url, message_timeout_s=1800).sync() as env:
    env.reset()
    result = env.call_tool("run_rollout", split="test", task_name=task["name"],
                           model="YOUR_SERVED_MODEL", sampling={"temperature": 0.8})
```
"""


def build_ui(manager, fields, metadata, is_chat, title, quick_start):
    return build_gradio_app(
        manager, fields, metadata, is_chat, title=title, quick_start_md=QUICK_START
    )
