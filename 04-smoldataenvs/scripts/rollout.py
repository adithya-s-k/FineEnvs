# /// script
# dependencies = ["datasets>=3.0.0", "huggingface_hub>=1.22.0", "math-verify"]
# ///
"""The whole environment, in one file: prompt in, reward out.

No environment framework. A rollout here is one turn:

    prompt   the question plus the file names sitting in /home/user/input
    model    writes a Python program in a fenced block
    sandbox  runs it (Hugging Face Sandboxes, one VM reused for the whole run)
    reward   SmolDataEnvs' own grader compares the printed value to the gold answer

Both `train_grpo.py` and `eval_pass1.py` import from here, and it is runnable on
its own as a smoke test:

    uv run rollout.py            # 3 tasks, gold-answer code, expects reward 1.0
"""

from __future__ import annotations

import importlib.util
import os
import re
import sys

SYSTEM = (
    "You are a data analyst. You answer questions about CSV files by writing a short "
    "Python program and reading what it prints."
)

PROMPT = """{question}

The files are in /home/user/input and your program runs in that directory:
{files}

Write one Python program in a ```python block, then stop.

- Look at the data if you need to, then compute the answer.
- The LAST thing the program prints must be the answer on its own: a bare number
  (no commas or units), a short label, yes/no, or a comma-separated list.
- Keep it under 40 lines. pandas, numpy, scipy, sklearn and statsmodels are installed."""


def build_prompt(row: dict) -> list[dict]:
    """The one prompt shape, so training and eval cannot drift apart."""
    files = "\n".join(f"- {f}" for f in row["files"])
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": PROMPT.format(question=row["question"], files=files)},
    ]


CODE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)


def extract_code(completion: str | list[dict]) -> str:
    """Last fenced block wins: models often think in one block and answer in the next."""
    if isinstance(completion, list):  # conversational completions
        completion = completion[-1]["content"] if completion else ""
    blocks = CODE_RE.findall(completion or "")
    if blocks:
        return blocks[-1].strip()
    # no fence at all: treat the whole thing as code rather than scoring a zero for
    # punctuation. If it is prose it will fail to compile, which is its own signal.
    return (completion or "").strip()


class SandboxUnavailable(RuntimeError):
    """The sandbox died. That is not the policy's fault and must not be scored."""


class SandboxRunner:
    """One Hugging Face Sandbox, reused for every rollout in the process.

    A sandbox takes ~6-10s to start, which is the whole per-rollout budget if you
    create one per completion. So: start once, and per task swap the contents of
    /home/user/input. Data for a task is pulled once and cached, which matters
    because GRPO asks for N completions of the *same* task back to back.
    """

    IMAGE = os.environ.get("SANDBOX_IMAGE", "docker.io/savatar101/env-data-agent-train:base")
    FLAVOR = os.environ.get("SANDBOX_FLAVOR", "cpu-basic")

    def __init__(self, timeout: int = 120):
        self._sb = None
        self._prefix = None
        self.timeout = timeout

    @property
    def sandbox(self):
        if self._sb is None:
            from huggingface_hub import Sandbox

            # forward_hf_token: the data lives in a Hub bucket, and the image's
            # own puller authenticates with HF_TOKEN.
            self._sb = Sandbox.create(
                image=self.IMAGE,
                flavor=self.FLAVOR,
                idle_timeout=os.environ.get("SANDBOX_IDLE", "30m"),
                forward_hf_token=True,
            )
        return self._sb

    def _stage(self, bucket_prefix: str) -> None:
        """Put this task's tables in /home/user/input, replacing the last task's."""
        if bucket_prefix == self._prefix:
            return
        # The image ships /opt/pull_bucket.py and it refuses to overwrite a
        # non-empty input dir, so clear it first.
        r = self.sandbox.run(
            f"rm -rf /home/user/input/* && BUCKET_PREFIX={bucket_prefix} python3 /opt/pull_bucket.py",
            timeout=self.timeout,
            check=False,
        )
        if "staged" not in (r.stdout or ""):
            raise RuntimeError(f"data pull failed for {bucket_prefix}: {r.stderr or r.stdout}")
        self._prefix = bucket_prefix

    def run(self, bucket_prefix: str, code: str) -> tuple[str, str]:
        """Run one program against one task's data. Returns (stdout, stderr).

        Raises `SandboxUnavailable` if the sandbox itself failed, which the caller
        turns into an ungraded rollout rather than a zero.
        """
        try:
            self._stage(bucket_prefix)
        except Exception as exc:  # the sandbox, not the program
            self._recycle()
            try:
                self._stage(bucket_prefix)
            except Exception as exc2:
                raise SandboxUnavailable(str(exc2)) from exc
        payload = code.replace("'", "'\\''")
        # check=False: the model's program failing is the ordinary case here, not
        # an error in the harness. A traceback is a reward of 0 and a signal, so it
        # has to come back as text rather than as an exception.
        r = self.sandbox.run(
            f"cd /home/user/input && printf '%s' '{payload}' > /tmp/solve.py && "
            f"timeout 90 python3 /tmp/solve.py",
            timeout=self.timeout,
            check=False,
        )
        return (r.stdout or ""), (r.stderr or "")

    def _recycle(self) -> None:
        """Throw the sandbox away and start a fresh one on the next call."""
        try:
            self.close()
        except Exception:
            pass
        self._prefix = None

    def close(self) -> None:
        if self._sb is not None:
            self._sb.kill()
            self._sb = None


# Adapted from 04-data-agent/envs/whitebox-bash/grader.py, where 42% of partial
# credit once went to strings like `echo -n "2.14" > answer.txt`: the text contains
# the right number and grades as the right answer.
_COMMAND_SHAPED = re.compile(
    r"(^|\s)(echo|printf|cat|python3?|bash|sh|tee|awk|sed)\b|[>|]{1,2}\s*\S+|\$\(|`",
)


def looks_like_a_command(answer: str) -> bool:
    """True when the 'answer' is really the line that would produce it."""
    return bool(answer) and bool(_COMMAND_SHAPED.search(answer.strip()))


def last_line(stdout: str) -> str:
    """The answer is the last thing printed. Anything above it is the model thinking."""
    lines = [ln.strip() for ln in (stdout or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


_grader = None


def grader():
    """SmolDataEnvs' own grader, so training and the dataset agree on what correct means."""
    global _grader
    if _grader is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download("FineEnvs/SmolDataEnvs", "grader.py", repo_type="dataset")
        spec = importlib.util.spec_from_file_location("smoldataenvs_grader", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["smoldataenvs_grader"] = mod  # its dataclasses need this
        spec.loader.exec_module(mod)
        _grader = mod
    return _grader


def grade(row: dict, prediction: str) -> float:
    if not prediction:
        return 0.0
    r = grader().grade(
        row["answer"],
        prediction,
        reward_mode=row["reward_mode"],
        abs_tol=row["atol"],
        rel_tol=row["rtol"],
    )
    return float(r.reward)


def rollout(runner: SandboxRunner, row: dict, completion) -> dict:
    """One completion → one graded result. The only place rewards come from."""
    code = extract_code(completion)
    try:
        compile(code, "<rollout>", "exec")
    except SyntaxError as exc:
        # Not a program. Scoring it costs a sandbox round trip and the answer is
        # always zero, so decide here. Length is deliberately not the test: a
        # correct program can be one short line.
        return {"reward": 0.0, "ran": 0.0, "prediction": "", "stderr": f"syntax: {exc}"}

    try:
        stdout, stderr = runner.run(row["bucket_prefix"], code)
    except SandboxUnavailable as exc:
        # An ungraded rollout is None, never 0.0. A dead sandbox and a wrong
        # answer are different events, and collapsing them teaches the model that
        # the dead sandbox was its fault. TRL turns None into NaN and drops the
        # sample from its group baseline.
        return {"reward": None, "ran": 0.0, "prediction": "", "stderr": f"sandbox: {exc}"}

    prediction = last_line(stdout)
    if looks_like_a_command(prediction):
        return {"reward": 0.0, "ran": 1.0, "prediction": prediction,
                "stderr": "answer is a command, not a value"}
    return {
        "reward": grade(row, prediction),
        # A program only counts as having run if it PRINTED something. An earlier
        # version counted "no traceback" as success, which a program that does
        # nothing also satisfies -- and the policy found that: by step ~70 it was
        # emitting two tokens, collecting the shaping reward on every rollout, and
        # every completion in a group scored identically, so the advantage was
        # zero and training had quietly stopped. Requiring output closes it.
        "ran": 1.0 if prediction else 0.0,
        "prediction": prediction,
        "stderr": stderr[-400:],
    }


if __name__ == "__main__":
    # Smoke test: feed the grader the gold answer through a real sandbox run.
    from datasets import load_dataset

    ds = load_dataset("FineEnvs/SmolDataEnvs", split="eval")
    runner = SandboxRunner()
    try:
        for row in list(ds)[:3]:
            gold = row["answer"]
            fake = f"```python\nprint({gold!r})\n```"
            out = rollout(runner, row, fake)
            print(
                f"{row['task_id']:28} mode={row['reward_mode']:12} "
                f"gold={gold[:22]!r:26} got={out['prediction'][:22]!r:26} reward={out['reward']}"
            )
            wrong = rollout(runner, row, "```python\nprint('definitely not the answer')\n```")
            assert wrong["reward"] == 0.0, "a wrong answer scored above zero"
            assert out["reward"] == 1.0, "the gold answer did not score 1.0"
    finally:
        runner.close()
    print("\nok: gold scores 1.0, wrong scores 0.0, sandbox torn down")
