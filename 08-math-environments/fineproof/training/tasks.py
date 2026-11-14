"""Prepared FineProofEnvs tasks, and one Harbor session per rollout, all through our harness."""

from __future__ import annotations

import json
import tomllib
from pathlib import Path

# Our Harbor agent, by import path: OpenEnv's `--harness module:Class` points a host-side agent at the capture
# proxy through its `api_base`/`api_key` kwargs. Options the server cannot pass come from FINEPROOF_* env vars.
HARNESS = "fineproof_agent.agent:FineProofAgent"
# A 32k-token proof takes 15 minutes or more on a loaded engine; the agent, its model call (FINEPROOF_REQUEST_TIMEOUT_SEC
# in jobs/run.py) and the server connection all wait an hour.
AGENT_TIMEOUT_SEC = 3600
# Our harness makes 1 + rounds model calls (see harness/fineproof_agent/loop.py); the cap is a backstop.
AGENT_STEP_LIMIT = 6


def dataset_dir(data, split):
    """The folder `openenv harbor serve --dataset` is given for a split. The server names the split by it."""
    return str(Path(data).resolve() / "datasets" / split)


def load_tasks(dataset):
    """Task folders of one served dataset, in the order the Harbor server indexes them (sorted by name)."""
    root = Path(dataset) / "tasks"
    tasks = []
    for folder in sorted(root.iterdir()):
        spec = tomllib.loads((folder / "task.toml").read_text())
        tasks.append({
            "name": folder.name,
            "folder": str(folder),
            "instruction": (folder / "instruction.md").read_text(),
            "base_reward": spec["metadata"].get("reward_mean"),
        })
    if not tasks or len({t["instruction"] for t in tasks}) != len(tasks):
        raise ValueError("Tasks must have unique instructions; run prepare.py first")
    return tasks


def agent_record(result, trial_root) -> dict:
    """What the FineProof agent recorded about its episode: `context.metadata["fineproof"]`, which Harbor keeps in the
    trial's result.json (written before the trial returns). {} when there is none."""
    name = getattr(result, "trial_name", None)
    if trial_root is None or not name:
        return {}
    try:
        trial = json.loads((Path(trial_root) / name / "result.json").read_text())
    except (OSError, ValueError):
        return {}
    record = (((trial or {}).get("agent_result") or {}).get("metadata") or {}).get("fineproof")
    return record if isinstance(record, dict) else {}


def submitted_proof_truncated(result, record: dict) -> bool:
    """Whether the proof the harness submitted was cut off: the model call it came from stopped at the output cap.

    With rewrite rounds that need not be the last call: a rewrite without a '## Final proof' heading keeps the earlier
    proof, so a rewrite cut off before its heading leaves a complete proof, and a cut-off first proof followed by such
    a rewrite is still the one graded. The agent records which call it submitted (`submitted_call`) and each call's
    finish reason (`finish_reasons`). Without that record (an older harness), the last captured call stands in, which
    is right only when the harness made one call (`--rounds 0`).
    """
    reasons, index = record.get("finish_reasons"), record.get("submitted_call")
    if isinstance(reasons, list) and isinstance(index, int) and 0 <= index < len(reasons) and reasons[index]:
        return reasons[index] == "length"
    turns = list(getattr(result, "turns", None) or [])
    return bool(turns) and getattr(turns[-1], "finish_reason", None) == "length"


def _session_class():
    """HarborSession, except that a rollout the server did not finish cleanly has no grade, and in training a proof
    cut off at the output cap scores 0.

    When the agent times out (a slow engine, say), Harbor still runs the verifier, which scores the missing proof 0:
    a wrong answer as far as GRPO can tell. Such a rollout says nothing about the policy, so it is left ungraded and
    drops out of its group, like a judge failure; rollouts.jsonl records it with no reward too, so its metrics do not
    count it as graded.

    A proof cut off at the cap is unfinished, but the judge grades whatever was written and often gives it partial
    credit (0.1 to 0.25 on average in our first runs), so running into the cap cost the policy little and proofs grew
    until most were cut off. With `zero_truncated` (training) such a proof scores 0, as in QED-Nano, where the judge
    sees nothing of an answer cut off mid-thought. It stays in its group: masking cut-off proofs instead destabilised
    Skywork-OR1. The judge's own score is kept in rollouts.jsonl as `judge_reward`. Evaluation keeps the judge's score.
    """
    from harbor_env.harness import HarborSession

    class FineProofSession(HarborSession):
        def _log_rollout(self, **extra):
            result, trial_root = self.result, getattr(self, "trial_root", None)
            if result is not None and trial_root is not None and not getattr(self, "_logged", False):
                import rollout_log

                self._logged = True  # one record per rollout, however often it is reached
                rollout_log.append(Path(trial_root).parent / "rollouts.jsonl", result, trial_root, **extra)

        def fetch_training_trace(self):
            # OpenEnv marks a capture with a FATAL validation finding as unusable ("drop it") and raises ValueError
            # for it, which TRL's harness worker reads as a broken capture contract and stops training on. One
            # such rollout (seen when HF Sandbox starts were timing out) ended a smoke run. Drop it as a failed
            # rollout instead: it is never trained on, it is recorded in rollouts.jsonl, and the ungraded-rollout
            # alert still fires if many go this way.
            result = self.result
            fatal = [f for f in (result.findings or []) if "[FATAL" in f] if result is not None else []
            if fatal:
                self._log_rollout(reward=None)  # never trained on, so not counted as graded either
                raise RuntimeError(f"unusable capture, dropped: {fatal[0][:300]}")
            return super().fetch_training_trace()

        def verify(self, transcript, final_state=None):
            verdict = super().verify(transcript, final_state)
            result = self.result
            if result is not None and (not result.ok or result.exception_type == "AgentTimeoutError"):
                self._log_rollout(reward=None)  # not the verifier's 0 for a missing proof: GRPO sees no grade either
                return type(verdict)(env_reward=None, done=True)
            truncated = result is not None and submitted_proof_truncated(
                result, agent_record(result, getattr(self, "trial_root", None)))
            if truncated and getattr(self, "zero_truncated", False) and verdict.env_reward is not None:
                self._log_rollout(reward=0.0, judge_reward=verdict.env_reward, truncated=True)
                return type(verdict)(env_reward=0.0, done=True)
            self._log_rollout(truncated=truncated)
            return verdict

    return FineProofSession


def session_for(args, index, task, split, sampling):
    """`sampling=None` is an eval rollout on the server's own engine (a hosted provider is eval-only)."""
    from harbor_env import HarborEnv

    session = _session_class()(
        env=HarborEnv(args.server, message_timeout_s=AGENT_TIMEOUT_SEC),
        owns_env=True,
        split=split,
        task_index=index,
        instruction=task["instruction"],
        harness=HARNESS,
        sandbox=getattr(args, "sandbox", None) or "daytona",
        llm_url=getattr(args, "env_llm_url", None) or args.vllm_url or "",
        model=args.model,
        sampling=sampling,
        purpose="auto" if sampling is not None else "eval",
        reward_key="reward",
        agent_step_limit=AGENT_STEP_LIMIT,
        agent_timeout_sec=AGENT_TIMEOUT_SEC,
    )
    session.trial_root = Path(args.trials)
    session.zero_truncated = getattr(args, "truncated_reward", "judge") == "zero"
    return session


class TaskFactory:
    """What TRL's HarnessRolloutWorker calls: `factory(sampling=...)`, then `.create(prompt, ...)` per rollout."""

    def __init__(self, args, tasks, *, split="train", sampling):
        self.args, self.sampling = args, sampling
        self.split = dataset_dir(args.data, split)
        self.tasks = {task["instruction"]: (i, task) for i, task in enumerate(tasks)}

    def create(self, task, seed=None, episode_id=None):
        # Match by prompt, not by the worker's seed: seeds can change after a restart.
        index, row = self.tasks[task[-1]["content"]]
        return session_for(self.args, index, row, self.split, self.sampling)
