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
    api_key: str | None = None,
    sampling: dict[str, float | int] | None = None,
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
    rollout_type = "eval"
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
            capture_level=engine_tier(llm_url, model, require_tokens=require_tokens, api_key=api_key),
            api_key=api_key,
            sampling=sampling,
            max_output_tokens=config.max_output_tokens,
            max_model_calls=config.agent_step_limit,
            task=task.instruction_id,
            sandbox=config.sandbox,
        )
        backend = build_backend(config.sandbox, image=config.image)
        # `create` takes only timeout/envs/metadata -- there is no setup hook on it. Staging runs as a
        # separate `exec` below, which is also what `opencode_env`'s harness does.
        sandbox = backend.create(
            timeout_s=int(config.agent_timeout_s),
            envs=task.env(None),
            metadata={"rollout_id": rollout_id, "task": task.instruction_id},
        )
        _wait_ready(sandbox)
        _ensure_opencode(sandbox, config)
        _stage_inputs(sandbox, task, hf_token, config)
        # The agent's API KEY is the capture session id. That is how one proxy serves many concurrent
        # rollouts without a port per rollout, and why the sandbox never sees a real credential.
        exit_code = _run_agent(
            sandbox, capture_url, session_id, model, config, task.instruction
        )
        timed_out = exit_code != 0

        turns, capture_findings = fetch_turns(server, session_id)
        n_tool_calls = sum(len(t.tool_calls) for t in turns)
        final = turns[-1].text if turns else None

        # ZERO MODEL CALLS IS AN INFRASTRUCTURE FAILURE, NOT A WRONG ANSWER.
        #
        # If capture saw nothing, the agent never reached the proxy -- opencode missing from the
        # image, a base URL the sandbox cannot route to, auth rejected. Grading that produces
        # correctness 0.0, which says the POLICY was wrong, and the trainer then counts it in the
        # group baseline. An ungraded rollout is dropped from the baseline instead, which is the
        # honest treatment: nothing about the model was measured here.
        if not turns:
            logger.warning(
                "rollout %s produced no model calls; returning ungraded. findings: %s",
                rollout_id,
                "; ".join(capture_findings[:3]) or "none",
            )
            return DataAgentRolloutResult(
                rollout_type=rollout_type,
                turns=[],
                timed_out=timed_out,
                metadata={
                    "error": "the agent made no model calls",
                    "rollout_id": rollout_id,
                    "session_id": session_id,
                    "sandbox": config.sandbox,
                    "capture_findings": capture_findings,
                },
            )

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
                "implementation": "standalone-opencode",
                "opencode_version": OPENCODE_VERSION,
                "max_output_tokens": config.max_output_tokens,
                "task_id": task.task_id,
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
        # Preserve observed tokens for diagnosis even when infrastructure prevents
        # grading. A partial capture must never turn an ungraded attempt into zero.
        turns, capture_findings = [], []
        if server is not None and session_id is not None:
            try:
                turns, capture_findings = fetch_turns(server, session_id)
            except Exception as capture_exc:
                capture_findings = [f"capture export failed: {type(capture_exc).__name__}"]
        return DataAgentRolloutResult(
            rollout_type=rollout_type, turns=turns,
            metadata={"error": f"{type(exc).__name__}: {exc}", "rollout_id": rollout_id,
                      "task_id": task.task_id, "capture_findings": capture_findings}
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


def _stage_inputs(sandbox: Any, task: DataAgentTask, hf_token: str | None, config: DataAgentConfig) -> None:
    """Pull this task's tables into the sandbox before the agent starts.

    RETRIED ONCE, deliberately. The exec channel is the fragile part of a sandbox, not the registry:
    across 13,200 trials, 5 of the 6 hard install failures were
    `Request timed out: the stream didn't open within 'request_timeout' (60.0 s)`, with zero npm/nvm
    rate-limiting. A single transient exec failure would otherwise cost the whole rollout.

    A staging failure RAISES rather than continuing. An agent that starts with no data files cannot
    solve the task, and the resulting empty answer scores identically to a model that could not do
    it -- so this must surface as an ungraded rollout, not as a zero.
    """
    setup = task.setup_shell(hf_token)
    if not setup:
        return
    last = None
    for attempt in (1, 2):
        result = sandbox.exec(setup, timeout=config.setup_timeout_s, envs=task.env(hf_token))
        code = _exit_code(result, default=0)
        if code == 0:
            return
        last = getattr(result, "stderr", "") or getattr(result, "stdout", "")
        logger.warning("staging attempt %d failed (%s): %s", attempt, code, str(last)[:300])
    raise RuntimeError(f"staging this task's inputs failed: {str(last)[:400]}")


# opencode lands here when installed at runtime; the E2B template also puts it on PATH.
OPENCODE_BIN = "$HOME/.opencode/bin"
OPENCODE_VERSION = os.environ.get("DATA_AGENT_OPENCODE_VERSION", "1.18.31")


def _exit_code(result: Any, *, default: int = 1) -> int:
    """Exit code of an `ExecResult`, treating a MISSING code and a ZERO code as different things.

    `int(getattr(r, "exit_code", 1) or 1)` looks right and is not: `0 or 1` is 1, so every SUCCESSFUL
    command reads as a failure. That turned "is opencode installed?" into a permanent no, which made
    every rollout reinstall it, and the installer then exits non-zero on "already installed" -- so a
    perfectly good sandbox failed with a message saying the thing it needed was already there.
    """
    code = getattr(result, "exit_code", None)
    return default if code is None else int(code)


def _wait_ready(sandbox: Any, *, attempts: int = 15, delay_s: float = 1.0) -> None:
    """Probe until `echo ok` succeeds. A backend returns the handle before the guest is usable.

    Without this, the FIRST command run in the sandbox fails for a reason that has nothing to do with
    what it was trying to do. Here that surfaced as "opencode is not installed" on an image where it
    was installed all along -- the probe simply ran too early.
    """
    import time

    last = ""
    for _ in range(attempts):
        try:
            r = sandbox.exec("echo ok", timeout=5)
            if _exit_code(r) == 0 and "ok" in (getattr(r, "stdout", "") or ""):
                return
            last = (getattr(r, "stderr", "") or getattr(r, "stdout", "") or "").strip()
        except Exception as exc:  # noqa: BLE001 -- a not-yet-listening guest raises rather than returns
            last = f"{type(exc).__name__}: {exc}"
        time.sleep(delay_s)
    raise RuntimeError(f"sandbox never became ready: {last[:300]}")


def _opencode_present(sandbox: Any) -> bool:
    """Whether `opencode` runs in this sandbox. The PATH export is load-bearing on a fresh install."""
    try:
        r = sandbox.exec(f'export PATH="{OPENCODE_BIN}:$PATH"; opencode --version', timeout=20)
        return _exit_code(r) == 0 and r.stdout.strip() == OPENCODE_VERSION
    except Exception:  # noqa: BLE001
        return False


def _ensure_opencode(sandbox: Any, config: DataAgentConfig) -> None:
    """Install opencode if the image does not already ship it.

    THE TWO BACKENDS DIFFER HERE AND IT IS NOT COSMETIC. The E2B template bakes opencode in, so the
    probe short-circuits and a rollout starts immediately. The HF image carries the data-science stack
    only, so opencode is installed at runtime -- roughly 30-50 s of the rollout.

    Skipping it on HF produces no useful error: `opencode run` is simply not found, the agent makes
    ZERO model calls, and the rollout returns an empty answer. Capture's `no_turns` finding is the
    only clue.

    SUCCESS IS DECIDED BY RE-PROBING, NOT BY THE INSTALLER'S EXIT CODE. The upstream installer exits
    non-zero when it finds the version already present ("Version 1.18.30 already installed"), so
    trusting the code turns a working sandbox into a failed rollout.
    """
    if _opencode_present(sandbox):
        return
    install = (
        f"mkdir -p {config.home}/.config/opencode {config.home}/workdir && "
        "set -o pipefail; curl -fsSL https://opencode.ai/install | bash -s -- --version "
        + __import__("shlex").quote(OPENCODE_VERSION)
    )
    last = None
    for attempt in (1, 2, 3):
        try:
            result = sandbox.exec(install, timeout=config.install_timeout_s)
            last = getattr(result, "stderr", "") or getattr(result, "stdout", "")
        except Exception as exc:  # noqa: BLE001 -- curl | bash is flaky; retry rather than abort
            last = f"{type(exc).__name__}: {exc}"
        if _opencode_present(sandbox):
            logger.info("opencode available after install attempt %d", attempt)
            return
        logger.warning("opencode still absent after attempt %d: %s", attempt, str(last)[:300])
    raise RuntimeError(f"could not install opencode in the sandbox: {str(last)[:400]}")


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
                # `npm` names the SDK opencode loads for a custom provider; the reference sets it
                # explicitly rather than relying on the provider key resolving by name.
                "npm": "@ai-sdk/openai-compatible",
                "name": "Intercepted",
                "options": {
                    "baseURL": f"{capture_url}/v1",
                    "apiKey": session_id,
                    # 10 minutes, matching the reference. Qwen3.5 is hybrid linear-attention with NO
                    # prefix caching, so every turn reprocesses the whole conversation and late turns
                    # are slow. An undeclared client timeout turns that into a FAILED tool result
                    # rather than a slow one.
                    "timeout": 600_000,
                },
                "models": {model: {}},
            }
        },
        "model": f"openai_compatible/{model}",
        **config.opencode_settings(),
    }
    sandbox.write_text(
        f"{config.home}/.config/opencode/opencode.json", json.dumps(settings, indent=2)
    )
    # THE INSTRUCTION GOES THROUGH A FILE, NEVER ONTO THE COMMAND LINE.
    #
    # `json.dumps(instruction)` produces a DOUBLE-quoted shell word, and these instructions contain a
    # backticked example:
    #
    #     Write only that value to /workdir/answer.txt (e.g. `echo -n "<value>" > /workdir/answer.txt`)
    #
    # Inside double quotes the shell runs backticks as COMMAND SUBSTITUTION. So the shell itself wrote
    # `<value>` into answer.txt before the agent started, AND deleted the example from the text the
    # agent got, which arrived as "(e.g. ), then stop.". Measured: 498 of 575 eval rollouts filed the
    # literal string `<value>`, dragging pass@1 to 0.032 against 0.104 for the same model under an
    # invocation that used a file. It reads as a model that cannot follow instructions.
    #
    # json.dumps also escapes newlines to a literal two-character \n, so a multi-paragraph instruction
    # reached the agent as one line of backslash-n.
    #
    # A file has neither problem, and it is what the working eval harness did.
    sandbox.write_text(f"{config.home}/workdir/task.md", instruction)
    # The shared backend protocol already supports detached execution. In particular,
    # Daytona's synchronous exec holds one HTTP request for the whole command and can
    # lose its result at the command deadline. Poll a background process instead;
    # a genuine agent budget expiry still leaves the captured trajectory available.
    process = sandbox.start_bg(
        f'export PATH="{OPENCODE_BIN}:$PATH"; '
        f"cd {config.home}/workdir && "
        f'opencode run --print-logs "$(cat {config.home}/workdir/task.md)"',
    )
    try:
        return process.wait(timeout=config.agent_timeout_s)
    except TimeoutError:
        process.kill()
        return 124


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
                loss_mask=list(e.get("loss_mask") or []),
                capture_metadata=dict(e.get("metadata") or {}),
                trainable=bool(any(e.get("loss_mask") or [])),
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
