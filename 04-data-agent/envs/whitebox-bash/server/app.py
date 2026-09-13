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

"""FastAPI app.

    uv run uvicorn server.app:app --host 0.0.0.0 --port 8000

`E2B_API_KEY` must be present: without it every `start_episode` fails at sandbox creation, which
surfaces as an environment that accepts connections and then refuses every episode.
"""

import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

try:
    from .environment import WhiteBoxBashEnvironment
except ImportError:  # running as `server.app` rather than as a package
    from server.environment import WhiteBoxBashEnvironment


# MAX_CONCURRENT_ENVS IS NOT OPTIONAL FOR TRAINING.
#
# `create_app` defaults to ONE concurrent MCP session. TRL builds one environment instance per batch
# slot, so at `num_generations=4` three of four clients are refused. Over the WebSocket transport that
# refusal is SILENT -- the server accepts the connection and immediately closes it, and the client
# dies on its first call with `ConnectionClosedOK: received 1000 (OK)`, which names no cause. It
# killed the first two smoke runs at step 0. Over HTTP `/mcp` the same condition reports itself
# properly as `Server at capacity: 1/1 sessions`, which is how it was finally diagnosed.
#
# Safe to raise here because this environment does not set `REQUIRES_SINGLE_THREAD_EXECUTOR`: its
# per-episode state lives in a module-level registry guarded by a lock, not on the instance.
#
# Keep this >= the trainer's concurrent rollouts. The real ceiling is the sandbox provider, which the
# environment bounds separately with WHITE_BOX_BASH_MAX_SESSIONS.
MAX_CONCURRENT_ENVS = int(os.environ.get("WHITE_BOX_BASH_MAX_CONCURRENT_ENVS", "64"))

app = create_app(
    WhiteBoxBashEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="white_box_bash",
    max_concurrent_envs=MAX_CONCURRENT_ENVS,
)


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("E2B_API_KEY"):
        raise SystemExit("E2B_API_KEY is not set; every episode would fail at sandbox creation")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
