"""Daytona implementation of the standalone OpenCode sandbox protocol."""
from __future__ import annotations

from functools import lru_cache
import math
import os
import shlex
import time
import uuid

from .base import ExecResult


@lru_cache(maxsize=1)
def client():
    from daytona import Daytona
    return Daytona()


class DaytonaBgJob:
    def __init__(self, handle, directory, pid):
        self.handle, self.directory, self._pid = handle, directory, pid

    @property
    def pid(self):
        return self._pid

    def wait(self, timeout=None):
        deadline = time.monotonic() + timeout if timeout is not None else float("inf")
        while time.monotonic() < deadline:
            if self.handle.exists(self.directory + "/exit"):
                return int(self.handle.read_text(self.directory + "/exit").strip())
            time.sleep(min(0.5, max(0, deadline-time.monotonic())))
        raise TimeoutError("Background command exceeded its deadline")

    def kill(self):
        self.handle.exec(f"kill -TERM -- -{self.pid} 2>/dev/null || true", timeout=10)


class DaytonaSandboxHandle:
    def __init__(self, sandbox):
        self._sandbox = sandbox
        self._deleted = False

    @property
    def sandbox_id(self):
        return self._sandbox.id

    def exec(self, cmd, *, envs=None, cwd=None, timeout=60):
        # The SDK's exec is not a shell. Quote the complete script once and let bash
        # interpret it; neither user text nor environment values are interpolated.
        result = self._sandbox.process.exec("bash -lc " + shlex.quote(cmd), cwd=cwd,
            env=envs, timeout=max(1, math.ceil(timeout)) if timeout is not None else None)
        code = result.exit_code
        return ExecResult(124 if code is None else int(code), result.result or "", "")

    def write_text(self, path, content):
        parent = path.rsplit("/", 1)[0] or "."
        result = self.exec("mkdir -p " + shlex.quote(parent))
        if result.exit_code:
            raise RuntimeError("Could not create sandbox directory")
        self._sandbox.fs.upload_file(content.encode(), path)

    def read_text(self, path):
        return self._sandbox.fs.download_file(path).decode()

    def exists(self, path):
        return self.exec("test -e " + shlex.quote(path), timeout=15).exit_code == 0

    def start_bg(self, cmd, *, envs=None, cwd=None):
        directory = "/tmp/openenv-process-" + uuid.uuid4().hex
        self.write_text(directory + "/run.sh", "#!/bin/bash\n(\n" + cmd + "\n)" +
                        "\ncode=$?\nprintf '%s' \"$code\" > " + shlex.quote(directory + "/exit") + "\n")
        result = self.exec("setsid bash " + shlex.quote(directory + "/run.sh") +
            " > " + shlex.quote(directory + "/output") + " 2>&1 < /dev/null & echo $!",
            envs=envs, cwd=cwd, timeout=15)
        if result.exit_code:
            raise RuntimeError("Could not launch background command")
        return DaytonaBgJob(self, directory, int(result.stdout.strip().splitlines()[-1]))

    def kill(self):
        if not self._deleted:
            client().delete(self._sandbox, timeout=60, wait=True)
            self._deleted = True


class DaytonaSandboxBackend:
    def __init__(self, *, image, snapshot=None):
        self.image, self.snapshot = image, snapshot or os.environ.get("DATA_AGENT_DAYTONA_SNAPSHOT")

    def create(self, *, timeout_s=900, envs=None, metadata=None):
        from daytona import CreateSandboxFromImageParams, CreateSandboxFromSnapshotParams, Resources
        options = dict(language="python", os_user="root", env_vars=envs,
            labels={**(metadata or {}), "openenv_component": "blackbox-opencode",
                    "openenv_owner": os.environ.get("RUN_OWNER", "local")},
            public=False, auto_stop_interval=max(30, math.ceil(timeout_s/60)),
            auto_delete_interval=0)
        if self.snapshot:
            params = CreateSandboxFromSnapshotParams(snapshot=self.snapshot, **options)
        else:
            params = CreateSandboxFromImageParams(image=self.image,
                resources=Resources(cpu=1, memory=4, disk=5), **options)
        return DaytonaSandboxHandle(client().create(params, timeout=300))
