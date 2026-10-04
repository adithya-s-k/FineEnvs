"""OpenCode for MiMo tasks: Harbor's OpenCode agent plus two steps of Xiaomi's harness that a task can't express.

1. The answer-leak blocklist. mimoagent appends it to /etc/hosts after the agent is installed (the install itself
   needs github.com). Task setup leaves the list in /var/lib/mimo/blocklist; this agent applies it right after
   install and fails closed, as mimoagent does.
2. The unprivileged agent user. Cyber and General run the agent as `agent`, so the verify server and the systems'
   data stay out of its reach. Tasks say so in [agent].user; on environments that cannot switch users (HF Sandbox
   runs everything as root), this agent drops to that user itself with runuser.

3. The output ceiling. OpenCode caps a reply at 32,000 tokens unless OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX says
   otherwise; it is set here from the configured model limit (65,536, mimoagent's opencode.yaml) rather than in the
   job's env, because Harbor treats any env var named *TOKEN* as a secret and blanks its value in every output.

Use it with `import_path: mimo_opencode:MimoOpenCode` (see jobs/*.yaml).
"""

from __future__ import annotations

import shlex
from typing import Any

from harbor.agents.installed.opencode import OpenCode
from harbor.environments.base import BaseEnvironment

MIMO = "/var/lib/mimo"


class MimoOpenCode(OpenCode):
    @staticmethod
    def name() -> str:
        return "mimo-opencode"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._drop_to: str | None | bool = False   # False: not checked yet
        limits = [m.get("limit", {}).get("output") for p in (self._opencode_config.get("provider") or {}).values()
                  for m in (p.get("models") or {}).values()]
        top = max([x for x in limits if isinstance(x, int)] or [0])
        self._mimo_env = {"OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": str(top)} if top > 32000 else {}

    async def _target_user(self, environment: BaseEnvironment) -> str | None:
        """The user the task wants the agent to run as, if the environment would otherwise run it as root."""
        if self._drop_to is False:
            want = await environment.exec(f"cat {MIMO}/agent_user 2>/dev/null", user="root")
            user = (want.stdout or "").strip() or None
            if user:
                who = await super().exec_as_agent(environment, "id -un")
                user = user if (who.stdout or "").strip() == "root" else None
            self._drop_to = user
        return self._drop_to

    async def exec_as_agent(self, environment: BaseEnvironment, command: str, env: dict[str, str] | None = None,
                            cwd: str | None = None, timeout_sec: int | None = None) -> Any:
        env = {**(env or {}), **self._mimo_env}
        user = await self._target_user(environment)
        if user:
            home = f"/home/{user}"
            exports = " ".join(f"{k}={shlex.quote(v)}" for k, v in (env or {}).items())
            command = (f"runuser -u {user} -- env HOME={home} USER={user} {exports} "
                       f"bash -c {shlex.quote(('cd ' + shlex.quote(cwd) + ' && ' if cwd else '') + command)}")
        return await super().exec_as_agent(environment, command, env=env, cwd=cwd, timeout_sec=timeout_sec)

    async def install(self, environment: BaseEnvironment) -> None:
        await super().install(environment)
        res = await environment.exec(
            f"if [ -f {MIMO}/blocklist ]; then cat {MIMO}/blocklist >> /etc/hosts && grep -c '^0.0.0.0' /etc/hosts; "
            f"else echo none; fi", user="root")
        if res.return_code != 0:
            raise RuntimeError("could not install the answer-leak blocklist in /etc/hosts: " + (res.stderr or res.stdout or "")[-300:])
        out = (res.stdout or "").strip()
        self.logger.info("answer-leak blocklist: " + ("none for this task" if out == "none" else f"{out} hosts blocked in /etc/hosts"))
