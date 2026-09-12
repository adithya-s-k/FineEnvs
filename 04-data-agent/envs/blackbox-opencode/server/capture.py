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

"""The capture proxy this environment runs, and the control plane for one rollout's session.

WHY THE CONTROL PLANE IS IN-PROCESS AND THE DATA PLANE IS NOT
The proxy has to serve HTTP: the agent runs inside a sandbox, on another machine, and reaches the
engine only by URL. That part is unavoidable and stays HTTP.

Minting a session and reading its turns back is a different matter. Those happen in *this* process,
which owns the `SessionRegistry` the proxy is writing into. An earlier version of this file went out
over HTTP for both, and got two things wrong that the direct path cannot get wrong:

  * it had to re-parse `/trace_entries` JSON that was serialised from objects sitting in local memory,
    once per rollout, for every turn of every rollout;
  * it never deleted the session, because there was no obvious place to. Sessions release slowly and a
    leaked one collides with the next run's claim, which surfaces as a burst of `CAPACITY_REACHED`
    rather than as a leak.

`openenv.core.harness.capture.CaptureServer` was written for exactly this and says so in its own
docstring: "a thread rather than a subprocess because the rollout path needs the live
`SessionRegistry` -- it mints a session, then reads the graph back out of it directly."

ONE SERVER PER PROCESS, AT MODULE SCOPE
The Task API builds a throwaway environment instance per request and closes it in a `finally`
(`http_server.py:1082-1097`). Anything held on `self` dies with the request, so a per-instance proxy
would bind a port per rollout and then leak it. The server, the engine tier and the port live here,
at module level, guarded by a lock.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

from openenv.core.harness.capture import CaptureServer, to_trace_entries
from openenv.core.harness.capture.export import export_session
from openenv.core.harness.capture.sessions import Upstream

from ..models import DataAgentTurn


logger = logging.getLogger(__name__)

# The port the agent inside the sandbox will be pointed at. It has to be reachable from there, so a
# deployment behind a tunnel sets CAPTURE_PUBLIC_URL to the outside address of this same port.
CAPTURE_PORT = int(os.environ.get("DATA_AGENT_CAPTURE_PORT", "8300"))

_LOCK = threading.Lock()
_SERVER: CaptureServer | None = None
# Measured tier per (llm_url, model). Deciding it means sending real completions, so it is measured
# once per engine and shared, never per rollout.
_TIERS: dict[tuple[str, str], str] = {}


def capture_server(
    llm_url: str, model: str, *, port: int = CAPTURE_PORT
) -> CaptureServer:
    """The one proxy this process runs, started on first use.

    Args:
        llm_url (`str`):
            Default engine, used by any request that does not name its own.
        model (`str`):
            Default served model id.
        port (`int`, *optional*):
            Port to bind. `start()` verifies that the server answering on it is *this* one rather
            than merely that something answers -- reachability is not identity, and a stale process
            holding this port answers every probe while capturing nothing.

    Returns:
        `CaptureServer`: running, with a live `SessionRegistry`.
    """
    global _SERVER
    with _LOCK:
        if _SERVER is None:
            server = CaptureServer(llm_url=llm_url, model=model, port=port)
            server.start()
            logger.info(
                "capture proxy listening on :%d (engine %s, model %s)",
                port,
                llm_url,
                model,
            )
            _SERVER = server
        return _SERVER


def agent_base_url(server: CaptureServer) -> str:
    """The address the agent inside the sandbox should call.

    NOT necessarily where we bound. The agent runs on another machine, so a deployment behind a
    tunnel or a reverse proxy has to advertise its outside address -- `CAPTURE_PUBLIC_URL`. Getting
    this wrong is quiet: opencode starts, cannot reach the engine, makes zero model calls, and the
    rollout comes back with a flat zero that reads exactly like a policy that cannot do the task.
    """
    public = os.environ.get("CAPTURE_PUBLIC_URL")
    return public.rstrip("/") if public else f"http://127.0.0.1:{server.port}"


def engine_tier(llm_url: str, model: str, *, require_tokens: bool) -> str:
    """Measure what this engine can return, once, and remember it.

    An engine served without `--return-tokens-as-token-ids --logprobs-mode processed_logprobs`
    degrades to `text` capture SILENTLY. Every rollout then looks completely normal and carries
    nothing to train on; two jobs spent hours that way before anyone noticed. So a training caller
    passes `require_tokens=True` and this raises at the first rollout instead.

    Args:
        require_tokens (`bool`):
            Raise if the engine cannot return token ids. True for training, False for evaluation --
            a text-only endpoint is a perfectly good eval backend, and refusing it would rule out
            every hosted provider.

    Returns:
        `str`: `"tokens"` or `"text"`.
    """
    key = (llm_url, model)
    with _LOCK:
        hit = _TIERS.get(key)
    if hit is not None and not (require_tokens and hit != "tokens"):
        return hit

    from openenv.core.harness.capture.validate_llm import require_llm

    report = require_llm(llm_url, model, require_tokens=require_tokens)
    level = report.capture_level or "text"
    with _LOCK:
        _TIERS[key] = level
    logger.info("engine %s (%s) probed: capture_level=%s", llm_url, model, level)
    return level


def mint_session(
    server: CaptureServer,
    *,
    llm_url: str,
    model: str,
    rollout_id: str,
    capture_level: str,
    max_model_calls: int = 0,
    **metadata: Any,
) -> tuple[str, str]:
    """Create a capture session on the live registry.

    `upstream` names the engine for THIS rollout rather than for the deployment, which is what lets
    one server serve a training run and an evaluation run against different engines at the same time.

    Args:
        max_model_calls (`int`, *optional*, defaults to `0`):
            Ceiling on model calls for this rollout; `0` is unlimited. Enforced by the proxy, which
            is the only component that sees every call -- `agent.build.steps` caps nothing, measured.

    Returns:
        `tuple[str, str]`: the session id -- which is also the agent's API key -- and `"train"` or
        `"eval"`.
    """
    session = server.registry.create(
        session_id=None,
        upstream=Upstream(llm_url=llm_url, model=model),
        capture_level=capture_level,
        max_model_calls=max_model_calls,
        rollout_id=rollout_id,
        **metadata,
    )
    level = session.capture_level or capture_level
    return session.session_id, ("train" if level == "tokens" else "eval")


def fetch_turns(server: CaptureServer, session_id: str) -> list[DataAgentTurn]:
    """Read the session's turns back, with the engine's own tokenization.

    `to_trace_entries` puts `prompt_token_ids` and `loss_mask` on every entry, so nothing downstream
    re-renders a prompt. Before that field existed a consumer had to rebuild each prompt with
    `apply_chat_template`, which matched the engine on 0 of 28 measured turns on Qwen3.5-4B and
    collapsed a run at its first weight update.
    """
    from .rollout import turns_from_capture

    session = server.registry.get(session_id)
    if session is None:
        logger.warning("capture session %s is gone; no turns to read", session_id)
        return []
    level = session.capture_level or server.capture_level
    document = export_session(session, include_messages=True, capture_level=level)
    findings = [f for f in document.get("validation", []) if not f.startswith("[INFO]")]
    if findings:
        logger.warning(
            "capture findings for %s: %s", session_id, "; ".join(findings[:5])
        )
    return turns_from_capture(to_trace_entries(session.graph, document))


def release_session(server: CaptureServer, session_id: str | None) -> None:
    """Drop the session as soon as its turns have been read.

    Not optional bookkeeping. A session held past its rollout keeps the graph alive and its key valid,
    and leftovers collide with the next run's claim -- which presents as a burst of
    `CAPACITY_REACHED` on a server that looks idle, not as a leak.
    """
    if not session_id:
        return
    try:
        server.registry.delete(session_id)
    except Exception:
        logger.warning(
            "could not release capture session %s", session_id, exc_info=True
        )


def shutdown() -> None:
    """Stop the proxy. For tests and for a clean server exit; a rollout never calls this."""
    global _SERVER
    with _LOCK:
        if _SERVER is not None:
            _SERVER.stop()
            _SERVER = None
