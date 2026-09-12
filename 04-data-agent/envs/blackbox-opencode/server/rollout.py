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

"""Server-side execution of one rollout: mint capture, boot sandbox, run agent, grade.

CONCURRENCY IS A HARD CEILING HERE, NOT A TUNING KNOB
Three separate limits stack, and the tightest one is not the obvious one:

  * THE CAPTURE PROXY IS A SINGLE UVICORN PROCESS. It is the real ceiling. Measured: `/health`
    starved at ~200 concurrent sessions while rollouts still succeeded, and the process crashed
    outright at 320 (3,525 file descriptors, 542 threads, 6.7 GB). It survived 75+ minutes at 200.
  * E2B allows 500 concurrent sandboxes per account, but the practical limit is lower because the
    capture server gives out first.
  * Sessions release SLOWLY, not instantly, and killing a client leaks its sessions. Leftovers
    collide with the next run's claim and surface as a burst of CAPACITY_REACHED.

So the gate below is deliberately well under the crash point, and it is a SEMAPHORE rather than a
rejection: a rollout that arrives over the limit waits its turn instead of failing. A training run
and an evaluation run share one deployment, and the eval must not be able to starve training out or
take the proxy down with it.
"""

from __future__ import annotations

import logging
import os
import threading
import uuid
from typing import Any

from ..config import DataAgentConfig
from ..models import DataAgentRolloutResult, DataAgentTurn
from ..reward import data_agent_reward
from ..sandbox import build_backend
from ..task import DataAgentTask
from ..verifier import answer_paths_for, grade_rollout, metadata_for


logger = logging.getLogger(__name__)

# Well under the 320 that killed the capture process and the ~200 where /health starved. Raise it
# only alongside a measurement, and remember that a deployment serves training AND eval at once.
MAX_CONCURRENT_ROLLOUTS = int(os.environ.get("DATA_AGENT_MAX_CONCURRENT", "64"))

_SLOTS = threading.BoundedSemaphore(MAX_CONCURRENT_ROLLOUTS)


def concurrency_status() -> dict[str, Any]:
    """What `capabilities()` reports, so a caller can size its own inflight budget."""
    return {
        "max_concurrent_rollouts": MAX_CONCURRENT_ROLLOUTS,
        "note": (
            "the capture proxy is a single uvicorn process; it starved /health at ~200 concurrent "
            "and crashed at 320. Training and eval share this budget."
        ),
    }


def run_rollout(
    task: DataAgentTask,
    *,
    llm_url: str,
    model: str,
    hf_token: str | None,
    config: DataAgentConfig,
    require_tokens: bool = True,
) -> DataAgentRolloutResult:
    """Run one rollout end to end. Never raises.

    A rollout that fails to launch, or whose agent dies, comes back UNGRADED (`reward=None`) rather
    than as a zero. That distinction is load-bearing: the trainer drops an ungraded rollout from the
    group baseline, whereas a zero says the policy was wrong. One suite once emitted a null reward,
    the type rejected it, and 86 of 250 tasks vanished from scoring while the run printed clean
    numbers over a third of the data.

    Args:
        task (`DataAgentTask`):
            The task to run, carrying gold and the bucket to stage.
        llm_url (`str`):
            The engine behind the proxy.
        model (`str`):
            Served model id.
        hf_token (`str`, *optional*):
            Token for staging the task's tables. Resolved once at startup, not per rollout.
        config (`DataAgentConfig`):
            Sandbox choice, timeouts, step cap.
        require_tokens (`bool`, *optional*, defaults to `True`):
            Refuse an engine that cannot return token ids. True for training, where such a rollout is
            worthless; False for evaluation, where a text-only endpoint is a fine backend.

    Returns:
        `DataAgentRolloutResult`: turns with engine token ids, the grade, and how it was graded.
    """
    rollout_id = uuid.uuid4().hex
    acquired = _SLOTS.acquire(timeout=config.agent_timeout_s * 2)
    if not acquired:
        logger.warning(
            "rollout %s waited past its budget for a slot; returning ungraded",
            rollout_id,
        )
        return DataAgentRolloutResult(
            metadata={"error": "no capacity", "rollout_id": rollout_id}
        )
    server = None
    session_id = None
    sandbox = None
    try:
        from .capture import (
            agent_base_url,
            capture_server,
            engine_tier,
            fetch_turns,
            mint_session,
        )

        server = capture_server(llm_url, model)
        capture_url = agent_base_url(server)
        session_id, rollout_type = mint_session(
            server,
            llm_url=llm_url,
            model=model,
            rollout_id=rollout_id,
            capture_level=engine_tier(llm_url, model, require_tokens=require_tokens),
            max_model_calls=config.agent_step_limit,
            task=task.instruction_id,
            sandbox=config.sandbox,
        )
        backend = build_backend(config.sandbox, image=config.image)
        sandbox = backend.create(
            setup_shell=task.setup_shell(hf_token),
            env=task.env(hf_token),
            timeout_s=config.agent_timeout_s,
        )
        # The agent's API KEY is the capture session id. That is how one proxy serves many concurrent
        # rollouts without a port per rollout, and why the sandbox never sees a real credential.
        exit_code = _run_agent(
            sandbox, capture_url, session_id, model, config, task.instruction
        )
        timed_out = exit_code != 0

        turns, capture_findings = fetch_turns(server, session_id)
        n_tool_calls = sum(len(t.tool_calls) for t in turns)
        final = turns[-1].text if turns else None

        grade = grade_rollout(
            task, sandbox.read_text, answer_paths_for(config.home), final_message=final
        )
        reward = data_agent_reward(grade.correctness, n_tool_calls)
        return DataAgentRolloutResult(
            rollout_type=rollout_type,
            reward=reward,
            correctness=grade.correctness,
            answer=grade.answer,
            answer_source=grade.source,
            graded_by=grade.graded_by,
            turns=turns,
            n_tool_calls=n_tool_calls,
            timed_out=timed_out,
            metadata={
                **metadata_for(task, grade, n_tool_calls),
                "rollout_id": rollout_id,
                "session_id": session_id,
                "sandbox": config.sandbox,
                # Surfaced rather than swallowed: `per_turn_capture_only` means the turns are exact
                # but became one graph root each, so a consumer expecting multi-turn credit
                # assignment is not getting it. That is invisible in the turn list itself.
                "capture_findings": capture_findings,
            },
        )
    except Exception as exc:  # noqa: BLE001 -- a flaky sandbox must not take the server down
        logger.warning(
            "rollout %s failed; returning ungraded", rollout_id, exc_info=True
        )
        return DataAgentRolloutResult(
            metadata={"error": f"{type(exc).__name__}: {exc}", "rollout_id": rollout_id}
        )
    finally:
        # Order matters: kill the sandbox and drop the capture session before releasing the slot, or
        # the next rollout claims a slot while this one is still holding an E2B seat and a live
        # session. Leftover sessions collide with the next run's claim and surface as a burst of
        # CAPACITY_REACHED on a server that looks idle.
        if sandbox is not None:
            try:
                sandbox.kill()
            except Exception:
                logger.warning(
                    "sandbox cleanup failed for %s", rollout_id, exc_info=True
                )
        if server is not None and session_id is not None:
            from .capture import release_session

            release_session(server, session_id)
        _SLOTS.release()


def _run_agent(
    sandbox: Any,
    capture_url: str,
    session_id: str,
    model: str,
    config: DataAgentConfig,
    instruction: str,
) -> int:
    """Configure opencode inside the sandbox and run it to completion.

    Written into `{home}` rather than a fixed path: the home differs by backend, and a config the
    agent cannot read means it starts with no model configured and makes zero model calls.
    """
    import json

    settings = {
        "$schema": "https://opencode.ai/config.json",
        "provider": {
            "openai_compatible": {
                "options": {"baseURL": f"{capture_url}/v1", "apiKey": session_id},
                "models": {model: {}},
            }
        },
        "model": f"openai_compatible/{model}",
        **config.opencode_settings(),
    }
    sandbox.write_text(
        f"{config.home}/.config/opencode/opencode.json", json.dumps(settings, indent=2)
    )
    result = sandbox.exec(
        f"cd {config.home}/workdir && opencode run --print-logs {json.dumps(instruction)}",
        timeout_s=config.agent_timeout_s,
    )
    return int(getattr(result, "exit_code", 0) or 0)


def turns_from_capture(entries: list[dict[str, Any]]) -> list[DataAgentTurn]:
    """Capture trace entries -> `DataAgentTurn`s, keeping the engine's own tokenization."""
    out = []
    for i, e in enumerate(entries):
        msg = ((e.get("response") or {}).get("choices") or [{}])[0].get("message") or {}
        out.append(
            DataAgentTurn(
                turn=i,
                prompt_token_ids=list(e.get("prompt_token_ids") or []),
                completion_token_ids=list(e.get("completion_token_ids") or []),
                per_token_logps=list(e.get("per_token_logps") or []),
                trainable=bool(
                    e.get("prompt_token_ids") and e.get("completion_token_ids")
                ),
                request_messages=list((e.get("request") or {}).get("messages") or []),
                request_tools=(e.get("request") or {}).get("tools"),
                text=msg.get("content") or "",
                tool_calls=list(msg.get("tool_calls") or []),
                finish_reason=((e.get("response") or {}).get("choices") or [{}])[0].get(
                    "finish_reason"
                ),
            )
        )
    return out
