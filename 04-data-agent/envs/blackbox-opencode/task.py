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
        difficulty_level (`str`):
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
    difficulty_level: str | None = None

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

        Credentials are passed by NAME through the environment, never interpolated into the command
        text, so a token cannot leak into a log line or a trace.
        """
        parts = [
            "set -eu",
            f"mkdir -p {shlex.quote(INPUT_DIR)} /workdir",
            # hf_hub is already in the base image; the pull is a library call rather than a shelled-out
            # CLI so a failure surfaces as a traceback instead of a silent empty directory.
            "python3 - <<'PY'\n"
            "import os\n"
            "from huggingface_hub import snapshot_download\n"
            "dest = os.environ['INPUT_DIR']\n"
            "snapshot_download(\n"
            "    repo_id=os.environ['HF_BUCKET'],\n"
            "    repo_type='dataset',\n"
            "    allow_patterns=os.environ['BUCKET_PREFIX'].rstrip('/') + '/*',\n"
            "    local_dir=dest,\n"
            ")\n"
            "PY",
            # Flatten: the instruction promises files directly in INPUT_DIR with no subfolders.
            f"find {shlex.quote(INPUT_DIR)} -mindepth 2 -type f -exec mv -t {shlex.quote(INPUT_DIR)} {{}} + || true",
            f"find {shlex.quote(INPUT_DIR)} -mindepth 1 -type d -empty -delete || true",
        ]
        return "\n".join(parts)


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
