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

"""One data-agent task: what the agent is asked, where its data lives, and how it will be graded."""

from __future__ import annotations

import hashlib
import os
import shlex
from typing import Any

from pydantic import BaseModel, Field


# Absolute paths the instruction text itself promises the agent. They cannot be changed here without
# editing every instruction in the dataset, so they are constants rather than configuration.
INPUT_DIR = "/home/user/input"
ANSWER_PATH = "/workdir/answer.txt"

# The agent may write its answer relative to a working directory that is not `/`, so the same file
# legitimately appears under the sandbox home. Both are read back; see `verifier.py`.
ANSWER_PATHS = (ANSWER_PATH, "{home}/workdir/answer.txt", "/root/workdir/answer.txt")


def instruction_id(instruction: str) -> str:
    """Stable id for an instruction string.

    The loop-owning path forwards only the prompt, so this is how a rollout finds its way back to the
    task that produced it. Hashing the instruction rather than trusting a row index means the mapping
    survives reordering, filtering and curricula.
    """
    return hashlib.sha1(instruction.encode()).hexdigest()


class DataAgentTask(BaseModel):
    """A single task, parsed from one dataset row.

    Attributes:
        task_id (`str`):
            The dataset's own identifier, carried for reporting.
        instruction (`str`):
            The full prompt shown to the agent, including the submission protocol.
        answer (`str`):
            Gold value. Never sent to the sandbox.
        question (`str`):
            The question alone, without the surrounding protocol text.
        reward_mode (`str`):
            How `grader.py` should compare: exact, numeric, list.
        atol (`float`):
            Absolute tolerance for numeric comparison.
        rtol (`float`):
            Relative tolerance for numeric comparison.
        hf_bucket (`str`):
            Hugging Face bucket holding this task's tables.
        bucket_prefix (`str`):
            Prefix within the bucket.
        files (`list[str]`):
            File names staged into `INPUT_DIR`, used only to name them in the prompt.
        difficulty_tier (`str`):
            easy, medium or hard.
        difficulty_level (`int`, *optional*):
            The dataset's finer-grained level, carried through for analysis.
    """

    task_id: str = ""
    instruction: str
    answer: str
    question: str = ""
    reward_mode: str = ""
    atol: float = 0.0
    rtol: float = 0.0
    hf_bucket: str
    bucket_prefix: str
    files: list[str] = Field(default_factory=list)
    difficulty_tier: str | None = None
    # An INT in the dataset (1/2/3), not a string -- `difficulty_tier` is the word form
    # ('easy'/'medium'/'hard') and these two are easy to mix up. Declaring this `str` made
    # every `get_task` 500 with a pydantic error that named the field but not the dataset.
    difficulty_level: int | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> "DataAgentTask":
        """Build from one `HuggingEnvs/data-agent` row."""
        return cls(
            task_id=str(row.get("task_id") or ""),
            instruction=row["instruction"],
            answer=str(row["answer"]),
            question=row.get("question", "") or "",
            reward_mode=row.get("reward_mode") or "",
            atol=float(row.get("atol") or 0.0),
            rtol=float(row.get("rtol") or 0.0),
            hf_bucket=row["hf_bucket"],
            bucket_prefix=row["bucket_prefix"],
            files=list(row.get("files") or []),
            difficulty_tier=row.get("difficulty_tier"),
            difficulty_level=row.get("difficulty_level"),
        )

    @property
    def instruction_id(self) -> str:
        return instruction_id(self.instruction)

    def env(self, token: str | None) -> dict[str, str]:
        """Environment for the staging step.

        A missing token is not defaulted to empty: the bucket pull then succeeds while downloading
        nothing, `INPUT_DIR` is empty, and every rollout scores zero in a way indistinguishable from
        a model that cannot do data analysis. The caller resolves the token once at startup and fails
        there instead.
        """
        env = {
            "HF_BUCKET": self.hf_bucket,
            "BUCKET_PREFIX": self.bucket_prefix,
            "INPUT_DIR": INPUT_DIR,
        }
        if token:
            env["HF_TOKEN"] = token
        return env

    def setup_shell(self, token: str | None) -> str:
        """Shell that stages this task's tables into `INPUT_DIR` before the agent starts.

        HF BUCKETS, NOT A DATASET REPO. `hf_bucket` names a bucket (`hf://buckets/<owner>/<name>`),
        so it is read with `list_bucket_tree` / `download_bucket_files`. Reaching for
        `snapshot_download(repo_type="dataset")` instead returns a plain 404 that names the repo and
        reads exactly like a permissions problem.

        FAILS LOUDLY ON AN EMPTY DIRECTORY, with a distinct exit code per cause. A silent miss hands
        the agent a task whose data is absent; it then scores 0 for a reason indistinguishable from a
        wrong answer, which is the most expensive kind of failure to debug because the reward looks
        entirely plausible.

        Credentials travel by NAME through the environment and are never interpolated into the command
        text, so a token cannot reach a log line or a trace.
        """
        bucket = self.hf_bucket
        prefix = self.bucket_prefix.rstrip("/")
        return (
            "set -e; "
            f"mkdir -p {shlex.quote(INPUT_DIR)} /workdir; "
            "python3 - <<'PULL'\n"
            "import sys\n"
            "from pathlib import Path\n"
            "from huggingface_hub import download_bucket_files, list_bucket_tree\n"
            f"dest = Path({INPUT_DIR!r})\n"
            "dest.mkdir(parents=True, exist_ok=True)\n"
            "if [p for p in dest.iterdir() if p.is_file()]:\n"
            "    print('[pull] already staged'); sys.exit(0)\n"
            # Flattened AT DOWNLOAD TIME rather than moved afterwards: the instruction promises the
            # files directly in INPUT_DIR with no subfolders, and a post-hoc `find -exec mv` silently
            # collides when two prefixes contain the same basename.
            f"targets = [(it.path, str(dest / Path(it.path).name))\n"
            f"           for it in list_bucket_tree({bucket!r}, prefix={prefix + '/'!r}, recursive=True)\n"
            "           if getattr(it, 'type', None) == 'file']\n"
            "if not targets:\n"
            f"    print('[pull] FATAL: nothing at hf://buckets/{bucket}/{prefix}'); sys.exit(2)\n"
            f"download_bucket_files({bucket!r}, files=targets)\n"
            "print('[pull] staged', len(targets), 'file(s)')\n"
            "PULL\n"
            f'[ -n "$(ls -A {shlex.quote(INPUT_DIR)})" ] || {{ echo "[pull] FATAL: {INPUT_DIR} empty"; exit 3; }}'
        )


def resolve_hf_token() -> str | None:
    """The Hub token, from the environment or a cached login.

    Checked at server startup rather than per rollout: see `env()` for why an absent token is worse
    than an obvious failure.
    """
    for name in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    try:
        from huggingface_hub import get_token

        return get_token()
    except Exception:
        return None
