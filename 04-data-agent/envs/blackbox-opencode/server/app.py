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

"""ASGI entry point for a deployed `data_agent_env`.

Everything is read from the environment, so one image serves any split and any engine with no
rebuild:

    DATA_AGENT_SPLITS        comma-separated splits, e.g. `train:medium,test` (default `train`)
    DATA_AGENT_SANDBOX       default backend, `e2b` or `hf`
    DATA_AGENT_MAX_CONCURRENT  rollouts in flight; see the ceiling note below
    DATA_AGENT_CAPTURE_PORT  port the capture proxy binds (default 8300)
    CAPTURE_PUBLIC_URL       how the SANDBOX reaches that port, when it is not localhost
    OPENENV_LLM_URL          default engine; optional, since a rollout may name its own
    OPENENV_MODEL            default served model id
    HF_TOKEN                 reads the dataset and stages each task's tables
    E2B_API_KEY              required for the `e2b` backend

OPENENV_LLM_URL IS OPTIONAL, AND THAT IS THE USEFUL WAY ROUND. With no engine the server still comes
up serving its splits, and each rollout names the engine it wants -- probed once, then cached. The
dataset and its prebuilt sandbox templates are the expensive things to host; an engine restarts every
training run, and a train-tier engine and an eval-tier one are usually both wanted at once.

THE CAPTURE PROXY IS NOT STARTED HERE. It is started lazily by the first rollout, against that
rollout's engine (`capture.capture_server`). Starting it at import would mean either binding it to a
default engine that no rollout uses, or refusing to boot without one.
"""

from __future__ import annotations

import logging
import os

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation

from ..sandbox import describe
from .environment import DataAgentEnvironment


logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger(__name__)

_SPLITS = [
    s.strip()
    for s in os.environ.get("DATA_AGENT_SPLITS", "train").split(",")
    if s.strip()
]


def _hf_token() -> str | None:
    """Resolved ONCE, here, and passed down by value.

    Not re-resolved per rollout: it is used to stage every task's tables, and a per-rollout lookup
    would hit the token store thousands of times in a run. Passed as a value, never logged.
    """
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    try:
        from huggingface_hub import get_token

        return get_token()
    except ImportError:
        return None


DataAgentEnvironment.configure(
    splits=_SPLITS,
    llm_url=os.environ.get("OPENENV_LLM_URL", ""),
    model=os.environ.get("OPENENV_MODEL", ""),
    hf_token=_hf_token(),
    sandbox=os.environ.get("DATA_AGENT_SANDBOX", "e2b"),
)

# Which backends are actually usable is worth one line at startup: a missing E2B_API_KEY otherwise
# surfaces as a rollout that fails after paying for a sandbox that could never have started.
logger.info("data_agent_env: splits=%s sandboxes: %s", _SPLITS, describe())

# WARM THE CAPTURE PROXY when a default engine is configured.
#
# It is lazy by default, which is right for a deployment whose engine arrives per rollout. But when
# the engine is known at boot, starting it here moves two slow, failure-prone steps off the first
# rollout: binding the port and minting the public tunnel. Under a trainer that opens `num_generations`
# rollouts at once, all of them would otherwise queue on the one holding the lock -- and a tunnel that
# fails to mint would surface as a rollout timeout rather than as a boot error.
#
# Never fatal. A server that cannot publish its proxy can still serve the Task API, and saying so at
# boot beats refusing to start.
if os.environ.get("OPENENV_LLM_URL"):
    try:
        from .capture import agent_base_url, capture_server

        _srv = capture_server(
            os.environ["OPENENV_LLM_URL"], os.environ.get("OPENENV_MODEL", "")
        )
        logger.info("capture proxy ready; agents will be pointed at %s", agent_base_url(_srv))
    except Exception:  # noqa: BLE001
        logger.warning(
            "could not warm the capture proxy; the first rollout will try again", exc_info=True
        )

os.environ.setdefault("ENABLE_WEB_INTERFACE", "true")

# TWO DIFFERENT CONCURRENCY LIMITS, AND THEY ARE NOT THE SAME NUMBER.
#
# `max_concurrent_envs` caps WebSocket SESSIONS -- how many callers may hold this env open. A trainer
# holds one session per in-flight rollout, so it has to be at least `num_generations` or rollouts
# queue at the door.
#
# `DATA_AGENT_MAX_CONCURRENT` (in `rollout.py`) caps rollouts actually EXECUTING, and that is the one
# bounded by measurement: the capture proxy is a single uvicorn process which starved `/health` at
# ~200 concurrent and crashed outright at 320 (3,525 fds, 542 threads, 6.7 GB). E2B allows 500
# sandboxes per account, so capture gives out first.
#
# Session cap above execution cap on purpose: a rollout that arrives over the execution limit WAITS
# on a semaphore rather than being refused. An evaluation run and a training run share one deployment,
# and the eval must not be able to starve training out or take the proxy down with it.
app = create_app(
    DataAgentEnvironment,
    CallToolAction,
    CallToolObservation,
    env_name="data_agent_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "128")),
)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))


if __name__ == "__main__":
    main()
