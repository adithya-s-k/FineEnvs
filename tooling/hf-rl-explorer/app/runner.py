"""Rollouts, run with OpenEnv's Harbor runner, the same code as OpenEnv's Harbor UI.

`openenv.harbor.rollout.run_rollout` starts the task's sandbox, runs the agent in it and grades the result with the
task's own tests. Its capture proxy sits between the agent and the model: the sandbox holds only a session id, never
a key, and every model call is recorded, which is what the run page shows while the rollout runs.

Whose account pays:
- the sandbox: Harbor's HF Sandbox environment creates sandboxes with the process's own HF token, so `Sandbox.create`
  is wrapped to use the token of whoever started the rollout. It travels in a context variable, which the rollout's
  thread and every asyncio task and worker thread it starts carry along.
- the model: Inference Providers through the router with the visitor's token, or an endpoint they bring, with its key.

The proxy must be reachable from the sandbox for agents installed in it: on a Space it is mounted into this app at
/capture; locally it runs on its own port behind a tunnel, as OpenEnv's UI does.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import json
import logging
import re
import secrets
import shlex
import threading
import time
from pathlib import Path
from typing import Any

from . import catalog, config, dockerfile, judge, store

log = logging.getLogger("rlx")

# Agents offered on the run panel: validated with OpenEnv's harness qualification. Host-side agents run in this
# process and only need the proxy locally; installed agents run in the sandbox and need it public.
AGENTS = [
    {"id": "opencode", "name": "OpenCode", "where": "sandbox"},
    {"id": "terminus-2", "name": "Terminus 2", "where": "host"},
    {"id": "mini-swe-agent", "name": "mini-SWE-agent", "where": "sandbox"},
    {"id": "pi", "name": "Pi", "where": "sandbox"},
]
AGENT_IDS = {a["id"] for a in AGENTS}
FLAVOR = "cpu-basic"
JUDGE_MAX_CALLS = 400   # a grader's model calls in one rollout (one per rubric check, a few retries)
TRIALS_DIR = config.CACHE_DIR.parent / "trials"   # agent logs: local disk, not the bucket

_SANDBOX_TOKEN: contextvars.ContextVar[str | None] = contextvars.ContextVar("rlx_sandbox_token", default=None)
# A task's `${VAR}`s (in [environment.env], [verifier.env], steps): filled only from what this rollout was given
# (a judge's relay, say), never from this server's own environment, which holds its secrets.
_TASK_ENV: contextvars.ContextVar[dict[str, str] | None] = contextvars.ContextVar("rlx_task_env", default=None)
_service: Any = None
_service_lock = threading.Lock()
_live: dict[str, "Rollout"] = {}
_live_lock = threading.Lock()
# A rollout's place is reserved under this one lock, Harbor's and MiMo's alike, from the capacity check until it is
# registered as live: two submits at the same moment can't both take the last place.
_capacity = threading.Lock()
_reserved: dict[str, str] = {}   # reservation -> user


# ── a task's variables: never this server's ──────────────────────────────────
_TEMPLATE = re.compile(r"\$\{([^}:]+)(?::-(.*))?\}")
# names that could carry a secret of this server's (its OAuth app, session key, tokens) or its operator's
SECRET_NAME = re.compile(r"KEY|SECRET|TOKEN|PASSW|CREDENTIAL|AUTH|COOKIE|SESSION|PRIVATE|^HF_|^HUGGING|^OAUTH|^RLX_|^OPENENV_|^AWS_|^GOOGLE_|^AZURE_",
                         re.IGNORECASE)


class MissingTaskEnv(ValueError):
    """A task asks for a variable this explorer doesn't provide (Harbor would have read it from the host)."""


def _resolver(phase: str):
    """Harbor's `resolve_env_vars` for one phase of a rollout ("environment": setup and the agent; "verifier": the
    grader; "any": both, for scrubbing): `${VAR}` comes from this rollout's values for that phase (app/judge.py's
    bindings, by key, then by variable), else the task's default; never from os.environ."""
    def resolve(env: dict[str, str]) -> dict[str, str]:
        given = _TASK_ENV.get() or {}
        vals = {**given.get("environment", {}), **given.get("verifier", {})} if phase == "any" else given.get(phase, {})
        out: dict[str, str] = {}
        for key, value in (env or {}).items():
            m = _TEMPLATE.fullmatch(str(value))
            if not m:
                out[key] = value
            elif key in vals:
                out[key] = vals[key]
            elif m.group(1) in vals:
                out[key] = vals[m.group(1)]
            elif m.group(2) is not None:
                out[key] = m.group(2)
            else:
                raise MissingTaskEnv(f"this task needs {m.group(1)}, which rollouts here don't provide")
        return out
    resolve._rlx_phase = phase
    return resolve


resolve_task_env = _resolver("environment")


class _HostEnv(dict):
    """What an agent may read of this server's environment when it looks for a setting: nothing secret-shaped."""

    def __init__(self) -> None:
        import os

        super().__init__({k: v for k, v in os.environ.items() if not SECRET_NAME.search(k)})


def _patch_harbor_env() -> None:
    """Harbor fills a task's `${VAR}` from the host's environment, and its agents fall back to the host's environment for
    their settings: here neither can reach this server's secrets."""
    import sys

    import harbor.environments.base  # noqa: F401 - imported so their references to resolve_env_vars are swapped too
    import harbor.trial.trial  # noqa: F401
    import harbor.verifier.verifier  # noqa: F401
    from harbor.agents import base as agent_base
    from harbor.agents.installed import base as installed_base
    from harbor.utils import env as harbor_env

    if getattr(harbor_env, "_rlx", False):
        return
    real = harbor_env.resolve_env_vars
    phases = {"harbor.verifier.verifier": "verifier", "harbor.trial.trial": "any"}
    for name, mod in list(sys.modules.items()):
        if mod is not None and getattr(mod, "resolve_env_vars", None) is real:
            mod.resolve_env_vars = _resolver(phases.get(name, "environment"))
    agent_base.BaseAgent._env_sources = lambda self: (self._extra_env, _HostEnv())
    installed_base.BaseInstalledAgent._env_sources = lambda self: (self._resolved_env_vars, self._extra_env, _HostEnv())
    harbor_env._rlx = True


def _patch_opencode_install() -> None:
    """Harbor installs OpenCode with apt (curl, nodejs, npm), nvm and `npm i -g`: on a cpu-basic sandbox with a big task
    image that alone ran past Harbor's 6 minute setup limit. The release binary (what the MiMo harness installs, one
    download) goes first; Harbor's own install stays the fallback, for an image that can't fetch it."""
    from harbor.agents.installed import opencode

    from .mimo.runner.opencode import INSTALL

    real = opencode.OpenCode.install
    if getattr(real, "_rlx", False):
        return

    async def install(self, environment) -> None:
        try:
            res = await environment.exec(command=INSTALL, user="root", timeout_sec=240)
            why = "" if res.return_code == 0 else (res.stderr or res.stdout or "")[-200:].strip()
        except Exception as e:  # noqa: BLE001 - a timeout or a dropped sandbox call: Harbor's install decides
            why = f"{type(e).__name__}: {e}"[:200]
        if not why:
            await self.ensure_system_dependencies(environment, ("coreutils",))   # run() pipes through stdbuf
            return
        log.warning("OpenCode release binary didn't install (%s); falling back to Harbor's npm install", why)
        await real(self, environment)

    install._rlx = True
    opencode.OpenCode.install = install


# ── the sandbox, billed to the visitor ───────────────────────────────────────
def _patch_hf_sandbox() -> None:
    """Make Harbor's HF Sandbox environment create each sandbox with the visitor's token, and run tasks that only
    have a Dockerfile by replaying it on its FROM image (see app/dockerfile.py)."""
    from harbor.environments import hf_sandbox

    if getattr(hf_sandbox.Sandbox, "_rlx", False):
        return
    real = hf_sandbox.Sandbox

    class VisitorSandbox:
        _rlx = True

        @staticmethod
        def create(*args, **kw):
            token = _SANDBOX_TOKEN.get()
            if not token:
                raise RuntimeError("no account to run this sandbox on: sign in again")
            kw["token"] = token
            kw.setdefault("start_timeout", 900)   # big task images take minutes to pull
            kw["labels"] = {**(kw.get("labels") or {}), "app": "hf-rl-explorer"}
            return real.create(*args, **kw)

        def __getattr__(self, name):   # anything else is the real class's
            return getattr(real, name)

    hf_sandbox.Sandbox = VisitorSandbox
    env_cls = hf_sandbox.HFSandboxEnvironment
    validate, start = env_cls._validate_definition, env_cls.start

    def _validate_definition(self) -> None:
        if not self.task_env_config.docker_image:
            df = self.environment_dir / "Dockerfile"
            if df.is_file():
                self._rlx_plan = dockerfile.plan(df.read_text(errors="replace"))   # raises with the reason
                self.task_env_config.docker_image = self._rlx_plan.base
        validate(self)

    async def _start(self, force_build: bool) -> None:
        await start(self, force_build)
        plan = getattr(self, "_rlx_plan", None)
        if plan:
            await _replay(self, plan)

    env_cls._validate_definition = _validate_definition
    env_cls.start = _start


async def _replay(env: Any, plan: Any) -> None:
    """A Dockerfile's steps, in the sandbox started from its FROM image."""
    import posixpath

    workdir = "/"
    timeout = int(env.task_env_config.build_timeout_sec or 600)
    for op, arg in plan.steps:
        if op == "workdir":
            workdir = posixpath.normpath(posixpath.join(workdir, str(arg)))
            await env.exec(f"mkdir -p {shlex.quote(workdir)}", cwd="/")
        elif op == "env":
            env._persistent_env.update({k: str(v) for k, v in dict(arg).items()})   # the agent's and grader's commands too
        elif op == "run":
            r = await env.exec(str(arg), cwd=workdir, timeout_sec=timeout)
            if r.return_code != 0:
                raise RuntimeError(f"replaying the task's Dockerfile failed at: RUN {str(arg)[:200]}\n"
                                   f"{(r.stderr or r.stdout or '')[-800:]}")
        elif op == "copy":
            srcs, dst = arg
            dst = dst if dst.startswith("/") else posixpath.join(workdir, dst)
            many = len(srcs) > 1 or dst.endswith("/")
            for src in srcs:
                local = (env.environment_dir / src.lstrip("./")) if src not in (".", "./") else env.environment_dir
                for path in (sorted(local.parent.glob(local.name)) if any(c in src for c in "*?[") else [local]):
                    if path.is_dir():
                        await env.upload_dir(path, dst.rstrip("/") or "/")
                    elif path.is_file():
                        target = posixpath.join(dst, path.name) if many else dst
                        await env.exec(f"mkdir -p {shlex.quote(posixpath.dirname(target) or '/')}", cwd="/")
                        await env.upload_file(path, target)
    if not env.task_env_config.workdir and workdir != "/":
        env.task_env_config.workdir = workdir   # where the agent starts, as the image would have had it


# ── the capture proxy ────────────────────────────────────────────────────────
def service() -> Any:
    """OpenEnv's Harbor service: the capture proxy, and the URL the sandbox reaches it at."""
    global _service
    with _service_lock:
        if _service is None:
            from openenv.harbor.serving import HarborService

            # No default engine: every rollout brings its own (the visitor's token or endpoint).
            _service = HarborService(llm_url=config.ROUTER, model="", datasets=[], provider="hf", capture_level="text",
                                     expose=config.CAPTURE_EXPOSE)
            _patch_hf_sandbox()
            _patch_harbor_env()
            _patch_opencode_install()
        return _service


def capture_app() -> Any:
    """The proxy's ASGI app, mounted at /capture on a Space (where this app is the one public port)."""
    return service().capture.app


def _proxy_url() -> str:
    svc = service()
    with _service_lock:
        if not svc.public_url:
            svc.start()   # on a Space: <space>/capture, already mounted; locally: its own port and a tunnel
    return svc.public_url


# ── one rollout ──────────────────────────────────────────────────────────────
class Cancelled(Exception):
    pass


class Rollout:
    def __init__(self, run: dict[str, Any], token: str, endpoint_key: str | None):
        self.run = run
        self.id = run["id"]
        self._token = token
        self._key = endpoint_key
        self.plan: dict[str, Any] = {"vars": [], "judge": None, "missing": [], "agent_keys": []}   # its variables (app/judge.py)
        self.judge_session: str | None = None
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task | None = None
        self.session_id: str | None = None
        self.cancelled = threading.Event()
        self.done = threading.Event()
        self.t0 = time.time()
        self._last_msgs = 0

    def update(self, **fields) -> None:
        self.run = store.update(self.id, **fields)

    # the model this rollout calls, through the proxy
    def upstream(self) -> Any:
        from openenv.core.harness.capture.sessions import Upstream

        ep = self.run.get("endpoint")
        if ep:
            return Upstream(llm_url=ep["base_url"], model=ep["model"], api_key=self._key or None, provider="openai")
        model = self.run["model"] + (f":{self.run['provider']}" if self.run.get("provider") else "")
        return Upstream(llm_url=config.ROUTER, model=model, api_key=self._token, provider="hf")

    def _heartbeat(self) -> None:
        """Touch the record every 20 s while this rollout runs, so a silent one is known to have lost its worker."""
        while not self.done.wait(20):
            try:
                store.update(self.id)
            except Exception:  # noqa: BLE001
                pass

    def execute(self) -> None:
        threading.Thread(target=self._heartbeat, daemon=True, name=f"beat-{self.id}").start()
        try:
            self.update(status="starting", phase="task", started_at=time.time())
            from .envs import registry

            # whatever the environment's format, its adapter writes the task out as a Harbor task folder
            task_dir = registry.materialize(self.run["dataset"], self.run["path"], self._token)
            if self.cancelled.is_set():
                raise Cancelled()
            toml = task_dir / "task.toml"
            self.plan = judge.plan(toml.read_text(errors="replace") if toml.is_file() else "")
            if self.plan["missing"]:
                raise MissingTaskEnv(f"this task needs {', '.join(self.plan['missing'])}, which rollouts here don't provide")
            self.update(status="setup", phase="sandbox")
            _SANDBOX_TOKEN.set(self._token)
            _TASK_ENV.set(judge.bindings(self.plan))   # no judge yet: keys empty, defaults kept (the judge is set in _rollout)
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.task = self.loop.create_task(self._rollout(task_dir))
            watcher = threading.Thread(target=self._watch, daemon=True, name=f"watch-{self.id}")
            watcher.start()
            result = self.loop.run_until_complete(self.task)
            self._finish(result)
        except (Cancelled, asyncio.CancelledError):
            self.update(status="cancelled", phase="done", finished_at=time.time(), wall_s=round(time.time() - self.t0, 1))
        except Exception as exc:  # noqa: BLE001 - reported on the run page
            self.update(status="failed", phase="done", error=_friendly(exc), finished_at=time.time(),
                        wall_s=round(time.time() - self.t0, 1))
        finally:
            self.done.set()
            self._key = None
            if self.loop:
                self.loop.close()
            with _live_lock:
                _live.pop(self.id, None)

    async def _rollout(self, task_dir: Path) -> Any:
        from openenv.harbor.rollout import run_rollout

        url = await asyncio.to_thread(_proxy_url)
        pool = service().capture.app.state.upstreams
        if self.run.get("endpoint"):   # checked when the rollout was asked for, and again now: a name can resolve elsewhere since
            from . import endpoints

            await asyncio.to_thread(endpoints.check_url, self.run["endpoint"]["base_url"])
        upstream = self.upstream()
        jup = None
        try:
            client, level = await pool.resolve(upstream)   # probes the endpoint once: tools, logprobs
            if self.plan.get("judge") and self.run.get("judge"):
                jup = await self._judge_session(url, pool)
            params = self.run.get("params") or {}
            return await run_rollout(
                task_dir=task_dir, harness=self.run["harness"], sandbox="hf-sandbox", registry=service().capture.registry,
                intercept_url=url, model=client.served_model or upstream.model, trials_dir=TRIALS_DIR / self.id,
                dataset=self.run["dataset"], require_reward=False, capture_level=level, purpose="eval",
                upstream=upstream, inference=client, agent_timeout_sec=params.get("timeout_min", 30) * 60,
                agent_step_limit=params.get("steps"), on_session_created=self._session,
            )
        finally:
            pool.forget(upstream)   # the visitor's key leaves the pool with the rollout
            if self.judge_session:
                sess = service().capture.registry.get(self.judge_session)
                calls = sess.graph.stats()["n_turns"] if sess is not None else None
                service().capture.registry.delete(self.judge_session)   # the capability dies with the rollout
                self.update(judge_calls=calls)
                self.judge_session = None
            if jup is not None:
                pool.forget(jup)

    async def _judge_session(self, url: str, pool: Any) -> Any:
        """The relay a model-graded task's grader calls: a session of the capture proxy of its own, serving the judge
        the visitor picked on HF Inference Providers with their token, reached with a per-rollout key that only the
        grading phase gets, capped, and deleted when the rollout ends."""
        from dataclasses import replace

        from openenv.core.harness.capture.sessions import Upstream

        jup = Upstream(llm_url=config.ROUTER, model=self.run["judge"], api_key=self._token, provider="hf")
        jclient, jlevel = await pool.resolve(jup)
        sid = "j" + secrets.token_hex(16)
        service().capture.registry.create(sid, upstream=replace(jup, model=jclient.served_model or jup.model), capture_level=jlevel,
                                          purpose="eval", max_model_calls=JUDGE_MAX_CALLS, role="judge", run=self.id)
        self.judge_session = sid
        _TASK_ENV.set(judge.bindings(self.plan, relay=url, capability=sid, judge=jclient.served_model or self.run["judge"]))
        return jup

    def _session(self, session_id: str) -> None:
        self.session_id = session_id

    def _watch(self) -> None:
        """While the agent works: its trajectory so far, from the proxy's record of its newest call."""
        while self.task and not self.task.done():
            try:
                self._snapshot()
            except Exception:  # noqa: BLE001 - the next tick tries again
                pass
            time.sleep(2)

    def _snapshot(self) -> None:
        if not self.session_id:
            return
        session = service().capture.registry.get(self.session_id)
        if session is None:
            return
        nodes = sorted(session.graph.nodes(), key=lambda n: n.index)
        if not nodes:
            return
        if self.run.get("status") == "setup":
            self.update(status="running", phase="agent")
        working = [n for n in nodes if n.n_tools]
        latest = (working or nodes)[-1]
        msgs = list(latest.request_messages or [])
        if latest.response_message:
            msgs.append({**latest.response_message, "role": "assistant"})
        if len(msgs) != self._last_msgs:
            self._last_msgs = len(msgs)
            store.write_artifact(self.id, "trajectory.json", json.dumps(_trim(msgs)))
            self.update(n_turns=len(nodes))

    def _finish(self, result: Any) -> None:
        agent = next((c for c in result.conversations if c.role == "agent"), None)
        if agent and agent.messages:
            store.write_artifact(self.id, "trajectory.json", json.dumps(_trim(agent.messages)))
        keep = result.model_dump(mode="json", exclude={"turns", "conversations"})
        store.write_artifact(self.id, "result.json", json.dumps(keep))
        graded = result.reward is not None
        # graded counts as done even when the agent ran out of time: benchmarks score a timeout like any other end
        self.update(status="done" if result.ok or graded else "failed", phase="done", finished_at=time.time(),
                    reward=result.reward, rewards=result.rewards or {}, reward_key=result.reward_key,
                    error=None if result.ok or graded else _friendly_text(result.error or result.exception_type or "failed"),
                    note=None if result.ok or not graded else _friendly_text(result.error or ""),
                    n_turns=result.n_turns, wall_s=round(result.wall_s or time.time() - self.t0, 1),
                    phase_timings=result.phase_timings, graded=graded, findings=result.findings[:20],
                    cost=_cost(self.run, result.wall_s))

    def cancel(self) -> None:
        self.cancelled.set()
        if self.loop and self.task and not self.task.done():
            self.loop.call_soon_threadsafe(self.task.cancel)


def _trim(messages: list[dict[str, Any]], limit: int = 8000) -> list[dict[str, Any]]:
    """The trajectory for the page: long tool outputs and file dumps cut, at most 600 messages."""
    out = []
    for m in messages[-600:]:
        m = dict(m)
        c = m.get("content")
        if isinstance(c, str) and len(c) > limit:
            m["content"] = c[:limit] + f"\n… ({len(c) - limit:,} more characters)"
        elif isinstance(c, list):
            m["content"] = [({**p, "text": p["text"][:limit]} if isinstance(p, dict) and isinstance(p.get("text"), str) else p)
                            for p in c if not (isinstance(p, dict) and p.get("type") in ("image_url", "image"))]
        out.append(m)
    return out


def _cost(run: dict[str, Any], wall_s: float | None) -> dict[str, float]:
    sandbox = round((wall_s or 0) / 3600 * config.FLAVOR_PRICE_PER_HOUR.get(FLAVOR, 0.01), 4)
    return {"sandbox": sandbox}


_CMD_FAILED = re.compile(r"Command failed \(exit (-?\d+)\):.*?(?=\nstdout:|\nstderr:|$)", re.S)


def _agent_error(t: str) -> str | None:
    """The last error an agent CLI printed as a JSON event (OpenCode's `{"type":"error",...}`), in a sentence."""
    for line in reversed(t.splitlines()):
        line = line.strip().removeprefix("stdout:").strip()
        if not (line.startswith("{") and '"error"' in line):
            continue
        try:
            err = (json.loads(line).get("error") or {})
        except ValueError:
            continue
        data = err.get("data") or {}
        msg, code, url = data.get("message") or err.get("name") or "an error", data.get("statusCode"), ((data.get("metadata") or {}).get("url") or "")
        where = ("at the gradio.live tunnel to this server's model proxy" if "gradio.live" in url
                 else "at the model proxy" if "/v1/" in url else "from the model")
        return f"The agent's model calls failed with {msg}{f' ({code})' if code else ''} {where}, so it stopped without an answer."
    return None


def _friendly_text(text: str) -> str:
    t = str(text)
    agent = _agent_error(t) if "Command failed" in t else None
    if agent:
        return agent
    if re.search(r"\b402\b|Payment Required", t):
        return "Your Hugging Face account has no credit for this: add a payment method or credits in your billing settings."
    if re.search(r"(sandbox|jobs?)\b.{0,80}\b(401|403)\b|\b(401|403)\b.{0,80}(sandbox|jobs?)\b", t, re.I | re.S):
        return "Hugging Face refused to start a sandbox on your account. Sign in again (with the Jobs permission)."
    if "requires a prebuilt Docker image" in t:
        return "This task builds its environment from a Dockerfile this explorer can't replay."
    # a failed command's text repeats the command, which holds the whole task prompt: keep its exit code and output only
    t = _CMD_FAILED.sub(lambda m: f"A command in the sandbox failed (exit {m.group(1)}).", t)
    return t[:600]


def _friendly(exc: Exception) -> str:
    return _friendly_text(f"{type(exc).__name__}: {exc}")


# ── starting, stopping, reading ──────────────────────────────────────────────
def submit(user: dict[str, Any], dataset: str, path: str, harness: str, model: str | None, provider: str | None,
           endpoint: dict[str, Any] | None, endpoint_key: str | None, params: dict[str, Any], visibility: str,
           runner: str = "harbor", fields: dict[str, Any] | None = None) -> dict:
    """Start a rollout of task `path` (the environment's ref) of `dataset`, with agent `harness`, by `runner`; `fields`
    are the run option's own inputs (contract.run_option fields), checked against it."""
    from . import settings

    token = user.get("token")
    if not token:
        raise PermissionError("sign in to run a rollout")
    if not settings.get("rollouts_enabled", True):
        raise RuntimeError("Rollouts are paused for maintenance. Try again later.")
    if harness not in settings.get("agents", list(AGENT_IDS)):
        raise ValueError("this agent is turned off here")
    max_user, max_all = settings.get("max_per_user", config.MAX_ACTIVE_PER_USER), settings.get("max_active", config.MAX_ACTIVE_ROLLOUTS)
    from .envs import registry

    env, row, options = registry.run_task(dataset, path, token)   # may this visitor read it, and how can it run
    opt = next((o for o in options if o["runner"] == runner), None)
    if opt is None:
        raise ValueError(f"this task doesn't run with the {runner} runner here")
    if not opt["ok"]:
        raise ValueError(f"This task can't run here: {opt['why']}.")
    if opt.get("harnesses") and harness not in opt["harnesses"]:
        raise ValueError(f"the {opt['label']} runner can't drive that agent")
    if endpoint and not opt.get("endpoint", True):
        raise ValueError(f"the {opt['label']} runner can't use your own endpoint")
    fields = _check_fields(opt.get("fields") or [], fields or {})
    if runner == "mimo":
        return _submit_mimo(user, env, row, model, provider, endpoint, endpoint_key, fields, visibility, max_user, max_all)
    params = {**params, **{k: fields[k] for k in ("steps", "timeout_min") if k in fields}}   # the Harbor runner's limits
    if (row.get("bytes") or 0) > config.MAX_TASK_BYTES:
        raise ValueError(f"This task's files are {row['bytes'] / 1024**3:.1f} GB, more than rollouts here download.")
    with _slot(user["name"], max_user, max_all):
        run_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)
        restricted = bool(row.get("restricted"))
        run = store.create({
            "id": run_id, "user": user["name"], "avatar": user.get("avatar"), "dataset": dataset, "path": path, "task_id": f"{dataset}:{path}",
            "title": row["title"], "collection": row.get("collection"), "env": env.key, "adapter": env.adapter.id, "runner": runner,
            "difficulty": row.get("difficulty"), "category": row.get("category"),
            "model": endpoint["model"] if endpoint else model, "provider": None if endpoint else provider,
            "endpoint": endpoint, "harness": harness, "sandbox": "hf-sandbox", "flavor": FLAVOR,
            "image": row.get("image"), "replayed": (row.get("runnable") or {}).get("how") == "dockerfile",
            "params": params, "fields": fields, "judge": fields.get("judge"), "status": "queued", "phase": "queued",
            # a private dataset's rollouts stay private: their trajectories quote its tasks
            "visibility": "private" if restricted else visibility, "restricted": restricted,
            "sha": row["sha"],
        })
        r = Rollout(run, token, endpoint_key)
        with _live_lock:
            _live[run_id] = r
        threading.Thread(target=r.execute, daemon=True, name=f"rollout-{run_id}").start()
    return run


@contextlib.contextmanager
def _slot(user: str, max_user: int, max_all: int):
    """Hold one of the explorer's rollout places for `user` while the rollout is created and started (both runners)."""
    from .mimo.runner import core as mimo

    key = secrets.token_hex(6)
    with _capacity:
        with _live_lock:
            mine = sum(1 for r in _live.values() if r.run.get("user") == user)
            total = len(_live)
        with mimo._live_lock:
            mine += sum(1 for r in mimo._live.values() if r.run.get("user") == user and not r.cancelled.is_set())
            total += len(mimo._live)
        mine += sum(1 for u in _reserved.values() if u == user)
        total += len(_reserved)
        if mine >= max_user:
            raise RuntimeError(f"You have {mine} rollouts running; wait for one to finish.")
        if total >= max_all:
            raise RuntimeError("The explorer is running as many rollouts as it can; try again in a few minutes.")
        _reserved[key] = user
    try:
        yield
    finally:
        with _capacity:
            _reserved.pop(key, None)


def _check_fields(spec: list[dict[str, Any]], given: dict[str, Any]) -> dict[str, Any]:
    """A run option's inputs: only the ones it declares, each of its type (one of its choices, for a select; a model of
    its pool, for a model), within its bounds."""
    known = {f["key"]: f for f in spec}
    extra = set(given) - set(known)
    if extra:
        raise ValueError(f"this runner takes no {', '.join(sorted(extra))}")
    out = {}
    for key, f in known.items():
        v = given.get(key, f.get("default"))
        if v is None:
            if f.get("required"):
                raise ValueError(f"pick a {f.get('label', key).lower()}")
            continue
        kind = f.get("type", "text")
        if kind == "select" and v not in [o if not isinstance(o, dict) else o.get("value") for o in f.get("options") or []]:
            raise ValueError(f"{f.get('label', key)}: pick one of its choices")
        if kind == "model":
            from . import models

            if not isinstance(v, str) or v not in {m["id"] for m in models.catalog().get(f.get("pool") or "", [])}:
                raise ValueError(f"{f.get('label', key)}: pick one of the models offered")
        if kind == "number" and (not isinstance(v, (int, float)) or isinstance(v, bool)):
            raise ValueError(f"{f.get('label', key)} should be a number")
        if kind == "number" and ((f.get("min") is not None and v < f["min"]) or (f.get("max") is not None and v > f["max"])):
            raise ValueError(f"{f.get('label', key)} should be between {f.get('min')} and {f.get('max')}")
        if kind == "bool" and not isinstance(v, bool):
            raise ValueError(f"{f.get('label', key)} should be yes or no")
        if kind == "text" and (not isinstance(v, str) or len(v) > 500):
            raise ValueError(f"{f.get('label', key)} should be a short text")
        out[key] = v
    return out


def _submit_mimo(user: dict[str, Any], env: Any, row: dict[str, Any], model: str | None, provider: str | None, endpoint: dict[str, Any] | None,
                 endpoint_key: str | None, fields: dict[str, Any], visibility: str, max_user: int, max_all: int) -> dict:
    """A MiMo task on the release's own harness (app/mimo/runner): its record sits in the same store, with the
    environment's dataset and ref, so it is listed with every other rollout of the task."""
    from . import models
    from .mimo.runner import core as mimo

    m = row["mimo"]
    judge = fields.get("judge") if m.get("needs_judge") else None
    if m.get("needs_judge") and not judge:
        raise ValueError("this task is graded by a model: pick a judge")
    params = {k: fields[k] for k in ("thinking", "temperature", "max_tokens", "steps", "timeout_min") if k in fields}
    if endpoint:
        ep = {"base_url": endpoint["base_url"], "host": endpoint["host"], "price_in": endpoint.get("price_in"), "price_out": endpoint.get("price_out")}
        model, provider = endpoint["model"], None
    else:
        ep = None
        provider = (models.get(model or "") or {}).get("provider")   # the cheapest that calls tools, as the MiMo explorer runs them
    with _slot(user["name"], max_user, max_all):
        return mimo.submit(user["name"], user["token"], {"id": m["id"], "domain": m["domain"], "title": m["title"], "facets": m.get("facets")},
                           model, provider, judge, endpoint=ep, agent_key=endpoint_key, params=params, visibility=visibility,
                           extra={"dataset": env.id, "path": m["id"], "env": env.key, "adapter": env.adapter.id, "runner": "mimo",
                                  "restricted": False, "sha": row.get("sha")}, reserved=True)


def mimo_live(run_id: str):
    from .mimo.runner import core as mimo

    return mimo.live(run_id)


def cancel(run_id: str) -> bool:
    with _live_lock:
        r = _live.get(run_id)
    if not r:
        m = mimo_live(run_id)
        if m is not None:
            from .mimo.runner import core as mimo

            return mimo.cancel(run_id)
        return False
    r.cancel()
    return True


def watch_cancel_requests() -> None:
    """Every few seconds: stop the rollouts the admin Space asked to stop (it shares the settings file, not this
    process), and mark rollouts whose worker is gone as interrupted (every 30 s)."""
    from . import settings

    tick = 0
    while True:
        try:
            for run_id, at in (settings.get("cancel_requests", {}) or {}).items():
                if time.time() - float(at) < 3600 and is_live(run_id):
                    cancel(run_id)
            if tick % 10 == 0:
                from .mimo.runner import core as mimo

                with _live_lock:
                    mine = set(_live)
                with mimo._live_lock:
                    mine |= set(mimo._live)
                store.mark_interrupted(mine)
            store.flush_due()
        except Exception:  # noqa: BLE001 - try again next tick
            pass
        tick += 1
        time.sleep(3)


def live() -> list[dict[str, Any]]:
    from .mimo.runner import core as mimo

    with _live_lock:
        out = [r.run for r in _live.values()]
    with mimo._live_lock:
        return out + [r.run for r in mimo._live.values()]


def is_live(run_id: str) -> bool:
    with _live_lock:
        if run_id in _live:
            return True
    return mimo_live(run_id) is not None
