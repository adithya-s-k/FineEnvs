"""TRL's tool interface backed by the local server or the same server on a Space."""

import math
from pathlib import Path

from openenv.core.mcp_client import MCPToolClient


class RemoteBashEnvironment:
    def __init__(self, server):
        self._client = MCPToolClient(server, message_timeout_s=900).sync()
        self._calls = 0
        self._submitted = False

    def reset(self, folder, **kwargs):
        folder = Path(folder)
        result = self._client.reset(
            split=folder.parent.parent.name, task_name=folder.name
        )
        self._calls, self._submitted = 0, False
        return result.observation.metadata["instruction"]

    def bash(self, command: str) -> str:
        """Run a shell command in /workdir.

        Args:
            command: The shell command to execute.
        """
        return self._client.call_tool("bash", command=command)

    def read(self, path: str) -> str:
        """Read a text file.

        Args:
            path: Absolute path of the file.
        """
        return self._client.call_tool("read", path=path)

    def write(self, path: str, content: str) -> str:
        """Write a text file, creating parent directories.

        Args:
            path: Absolute path of the file.
            content: Text to write.
        """
        return self._client.call_tool("write", path=path, content=content)

    def edit(self, path: str, old: str, new: str) -> str:
        """Replace one exact occurrence in a file.

        Args:
            path: Absolute path of the file.
            old: Text that must occur exactly once.
            new: Replacement text.
        """
        return self._client.call_tool("edit", path=path, old=old, new=new)

    def ls(self, path: str = "/workdir") -> str:
        """List a directory.

        Args:
            path: Directory to list.
        """
        return self._client.call_tool("ls", path=path)

    def grep(self, pattern: str, path: str) -> str:
        """Search text files recursively.

        Args:
            pattern: Regular expression to search for.
            path: File or directory to search.
        """
        return self._client.call_tool("grep", pattern=pattern, path=path)

    def glob(self, pattern: str, path: str = "/workdir") -> str:
        """Find files by name.

        Args:
            pattern: Filename pattern, for example *.csv.
            path: Directory to search.
        """
        return self._client.call_tool("glob", pattern=pattern, path=path)

    def submit_solution(self, answer: str) -> str:
        """Save your final answer and finish the episode.

        Args:
            answer: The answer itself, not a command.
        """
        result = self._client.call_tool("submit_solution", answer=answer)
        self._submitted = True
        return result

    def get_reward(self):
        result = self._client.call_tool("grade")
        self._calls = result["tool_calls"]
        self._correctness = result["correctness"]
        return result["reward"] if result["reward"] is not None else math.nan

    def _close(self):
        self._client.close()
