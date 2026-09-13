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


app = create_app(
    WhiteBoxBashEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="white_box_bash",
)


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("E2B_API_KEY"):
        raise SystemExit("E2B_API_KEY is not set; every episode would fail at sandbox creation")
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
