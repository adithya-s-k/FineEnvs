"""SETA tools, sandbox lifecycle and correctness grading for SmolDataEnvs."""

import math
import shlex
import time
from pathlib import Path

from .daytona import DaytonaSandboxBackend
from .tasks import grade_answer, stage_task

SYSTEM = "Use the sandbox tools to solve the task. Call submit_solution with the final answer itself."


class BashEnvironment:
    def __init__(self):
        self._sandbox = None
        self._calls = 0
        self._deadline = 0
        self._submitted = False

    def reset(self, folder, **kwargs):
        self._close()
        self._folder = Path(folder)
        self._calls, self._submitted = 0, False
        self._sandbox = DaytonaSandboxBackend(
            image="docker.io/savatar101/env-data-agent-train:base"
        ).create(timeout_s=900)
        try:
            stage_task(self._sandbox, self._folder)
        except BaseException:
            self._close()
            raise
        self._deadline = time.monotonic() + 600
        return (self._folder / "instruction.md").read_text()

    def _check(self):
        if self._submitted or time.monotonic() >= self._deadline:
            raise RuntimeError("Episode is finished")

    def _execute(self, command):
        self._check()
        self._calls += 1
        result = self._sandbox.exec(
            command,
            cwd="/workdir",
            timeout=min(60, max(1, self._deadline - time.monotonic())),
        )
        return (result.stdout + result.stderr)[-16000:]

    def bash(self, command: str) -> str:
        """Run a shell command in /workdir.

        Args:
            command: The shell command to execute.
        """
        return self._execute(command)

    def read(self, path: str) -> str:
        """Read a text file.

        Args:
            path: Absolute path of the file.
        """
        return self._execute("cat -- " + shlex.quote(path))

    def write(self, path: str, content: str) -> str:
        """Write a text file, creating parent directories.

        Args:
            path: Absolute path of the file.
            content: Text to write.
        """
        self._check()
        self._calls += 1
        self._sandbox.write_text(path, content)
        return "Saved."

    def edit(self, path: str, old: str, new: str) -> str:
        """Replace one exact occurrence in a file.

        Args:
            path: Absolute path of the file.
            old: Text that must occur exactly once.
            new: Replacement text.
        """
        self._check()
        self._calls += 1
        text = self._sandbox.read_text(path)
        if text.count(old) != 1:
            return "The old text must occur exactly once."
        self._sandbox.write_text(path, text.replace(old, new, 1))
        return "Saved."

    def ls(self, path: str = "/workdir") -> str:
        """List a directory.

        Args:
            path: Directory to list.
        """
        return self._execute("ls -la -- " + shlex.quote(path))

    def grep(self, pattern: str, path: str) -> str:
        """Search text files recursively.

        Args:
            pattern: Regular expression to search for.
            path: File or directory to search.
        """
        return self._execute(
            "grep -rn -- " + shlex.quote(pattern) + " " + shlex.quote(path)
        )

    def glob(self, pattern: str, path: str = "/workdir") -> str:
        """Find files by name.

        Args:
            pattern: Filename pattern, for example *.csv.
            path: Directory to search.
        """
        return self._execute(
            "find " + shlex.quote(path) + " -name " + shlex.quote(pattern)
        )

    def submit_solution(self, answer: str) -> str:
        """Save your final answer and finish the episode.

        Args:
            answer: The answer itself, not a command.
        """
        self._check()
        self._sandbox.write_text("/workdir/answer.txt", answer)
        self._submitted = True
        return "Answer saved. End your response now."

    # Submitting the answer does not count as a tool action.
    def get_reward(self):
        try:
            self._correctness = grade_answer(self._sandbox, self._folder)
            bonus = 0.1 * 15 / (15 + self._calls) if self._calls > 0 else 0.0
            return self._correctness * (1 + bonus)
        except (OSError, RuntimeError, ValueError, TimeoutError):
            return math.nan  # No grade is different from a wrong answer.
        finally:
            self._close()

    def _close(self):
        if self._sandbox is not None:
            self._sandbox.kill()
            self._sandbox = None
