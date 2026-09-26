"""One rollout, start to finish: sandbox up, environment set, agent runs, task's own verifier grades.

Everything a rollout does becomes an *event*, kept in memory for live viewers and appended to
`events.jsonl` in batches. Event kinds the UI understands:

    phase    {name, status: "start"|"done"|"error", detail}
    text     {text}                                 an assistant message
    tool     {tool, title, input, output, status, ms}
    step     {tokens: {...}, cost}                  one model call finished
    log      {text}                                 setup / verifier output worth showing
    checks   {checks: [{id, passed, score, method, tier, weight, message}], reward, summary}
    error    {text}

The user's token is held here for the rollout's lifetime only. It never reaches an event or a file.
"""

from __future__ import annotations

import re
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from .. import config, models, store

_pool = ThreadPoolExecutor(max_workers=config.MAX_ACTIVE_ROLLOUTS, thread_name_prefix="rollout")
_live: dict[str, "Rollout"] = {}
_live_lock = threading.Lock()


# key-shaped: a known prefix, then a run with an uppercase letter or a digit (code names like hf_raise_for_status don't match)
SECRET_RE = re.compile(r"\b(hf_|sk-|sk_|api_key=|Bearer )(?=[A-Za-z0-9\-]*[A-Z0-9])[A-Za-z0-9\-]{4,}(?![a-z_])")


class Cancelled(Exception):
    pass


class Rollout:
    def __init__(self, run: dict, token: str, agent_key: str | None = None):
        self.run = run
        self.id = run["id"]
        self.token = token
        self.agent_key = agent_key   # a bring-your-own endpoint's key: memory only, like the HF token
        self.events: list[dict] = store.read_events(self.id)
        self._pending: list[dict] = []
        self._last_flush = time.time()
        self._lock = threading.Lock()
        self.t0 = time.time()
        self.sandbox = None
        self.cancelled = threading.Event()
        self.tokens = {"input": 0, "output": 0, "reasoning": 0, "cache_read": 0, "cache_write": 0}
        self.model_cost = 0.0

    # ── events ───────────────────────────────────────────────────────────────
    def _redact(self, v):
        """Nothing secret reaches a trace: the user's HF token, an endpoint key, or anything shaped like one
        (the General verifier, for one, prints the first characters of its judge key)."""
        if isinstance(v, str):
            for secret in (self.token, self.agent_key):
                if secret:
                    v = v.replace(secret, "[redacted]")
            return SECRET_RE.sub(lambda m: m.group(1) + "[redacted]", v)
        if isinstance(v, dict):
            return {k: self._redact(x) for k, x in v.items()}
        if isinstance(v, list):
            return [self._redact(x) for x in v]
        return v

    def emit(self, kind: str, **data) -> None:
        ev = {"i": len(self.events), "t": round(time.time() - self.t0, 2), "kind": kind, **self._redact(data)}
        with self._lock:
            self.events.append(ev)
            self._pending.append(ev)
            if time.time() - self._last_flush > 1.0 or kind in ("phase", "checks", "error"):
                self._flush()

    def _flush(self) -> None:
        pending, self._pending = self._pending, []
        self._last_flush = time.time()
        store.append_events(self.id, pending)

    def phase(self, name: str, status: str = "start", detail: str = "") -> None:
        self.emit("phase", name=name, status=status, detail=detail)
        if status == "start":
            self.update(status={"sandbox": "starting", "setup": "setup", "agent": "running",
                                "verify": "verifying"}.get(name, self.run.get("status")), phase=name)

    def log(self, text: str) -> None:
        if text and text.strip():
            self.emit("log", text=text.strip()[-6000:])

    def add_tokens(self, t: dict) -> None:
        for k in self.tokens:
            self.tokens[k] += int(t.get(k) or 0)
        ep = self.run.get("endpoint")
        if ep:   # your own endpoint: metered only if you told us its prices
            t = self.tokens
            self.model_cost = ((t["input"] + t["cache_read"] + t["cache_write"]) * (ep.get("price_in") or 0)
                               + (t["output"] + t["reasoning"]) * (ep.get("price_out") or 0)) / 1e6
        else:
            self.model_cost = models.cost(self.run["model"], self.run.get("provider"), self.tokens)
        self.update(tokens=self.tokens, cost=self.cost_now())

    def cost_now(self) -> dict:
        hours = (time.time() - self.t0) / 3600
        sandbox = hours * config.FLAVOR_PRICE_PER_HOUR.get(self.run.get("flavor") or "", 0.0)
        judge = self.run.get("judge_cost") or 0.0
        return {"model": round(self.model_cost, 5), "judge": round(judge, 5), "sandbox": round(sandbox, 5),
                "total": round(self.model_cost + judge + sandbox, 5)}

    def update(self, **fields) -> None:
        fields = self._redact(fields)
        self.run.update(fields)
        store.update(self.id, **fields)

    def check_cancel(self) -> None:
        if self.cancelled.is_set():
            raise Cancelled()

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start_sandbox(self, image: str, flavor: str) -> "object":
        from huggingface_hub import Sandbox

        self.phase("sandbox", detail=f"{image.rsplit(':', 1)[-1]} on {flavor}")
        self.update(flavor=flavor)
        t = time.time()
        self.sandbox = Sandbox.create(image=image, flavor=flavor, idle_timeout=config.SANDBOX_IDLE_TIMEOUT,
                                      forward_hf_token=True, token=self.token, start_timeout=900,
                                      labels={"app": "mimo-explorer", "run": self.id})
        self.update(sandbox_id=self.sandbox.id)
        self.phase("sandbox", "done", f"ready in {time.time() - t:.0f}s")
        return self.sandbox

    def agent_api(self) -> tuple[str, str | None, str]:
        """(base URL, key, model string) the agent's model calls go to."""
        ep = self.run.get("endpoint")
        if ep:
            return ep["base_url"], self.agent_key, self.run["model"]
        mid = f"{self.run['model']}:{self.run['provider']}" if self.run.get("provider") else self.run["model"]
        return config.ROUTER, self.token, mid

    def sh(self, cmd: str, timeout: float = 600, **kw):
        """Run a shell command in the sandbox; never raises on a non-zero exit."""
        self.check_cancel()
        return self.sandbox.run(cmd, shell=True, check=False, timeout=timeout, **kw)

    def execute(self) -> None:
        from .domains import ADAPTERS

        adapter = ADAPTERS[self.run["domain"]]
        try:
            self.update(status="starting", started_at=time.time())
            result = adapter.run(self)
            self.update(status="done", reward=result.get("reward"), reward_error=result.get("error"),
                        finished_at=time.time(), cost=self.cost_now())
            self.phase("done", "done", f"reward {result.get('reward')}" if result.get("reward") is not None
                       else f"not scored: {result.get('error')}")
        except Cancelled:
            self.update(status="cancelled", finished_at=time.time(), cost=self.cost_now())
            self.phase("done", "error", "cancelled")
        except Exception as e:  # noqa: BLE001
            if self.cancelled.is_set():   # killing the sandbox makes the blocked call fail: that is the cancel, not a crash
                self.update(status="cancelled", finished_at=time.time(), cost=self.cost_now())
            else:
                msg = friendly(e) or f"{type(e).__name__}: {str(e)[:300]}"
                self.emit("error", text=msg, trace=traceback.format_exc()[-3000:])
                self.update(status="failed", error=msg, finished_at=time.time(), cost=self.cost_now())
        finally:
            with self._lock:
                self._flush()
            self.stop_sandbox()
            self.agent_key = None
            with _live_lock:
                _live.pop(self.id, None)

    def stop_sandbox(self) -> None:
        if self.sandbox is None:
            return
        try:
            from huggingface_hub import Sandbox

            Sandbox.kill(self.sandbox.id, token=self.token)
        except Exception:  # it idles out on its own anyway
            pass
        self.sandbox = None


def friendly(e: Exception) -> str | None:
    """The Hub errors people actually hit, said so they know what to do."""
    status = getattr(getattr(e, "response", None), "status_code", None)
    text = str(e)
    if status == 402 or "402 Payment Required" in text:
        return ("Your Hugging Face account has no prepaid credit for sandboxes, so none could start. Add credit at "
                "https://huggingface.co/settings/billing and run again. Nothing was charged.")
    if status in (401, 403) and "/api/jobs" in text:
        return ("Your sign-in isn't allowed to start sandboxes. Sign in again and allow Jobs, or use a write token.")
    if status == 429:
        return "Hugging Face is rate-limiting sandbox starts for your account. Wait a minute and run again."
    return None


# ── public API ───────────────────────────────────────────────────────────────
def active_for(user: str) -> int:
    with _live_lock:
        return sum(1 for r in _live.values() if r.run["user"] == user and not r.cancelled.is_set())


def submit(user: str, token: str, task: dict, model: str, provider: str | None, judge: str | None,
           endpoint: dict | None = None, agent_key: str | None = None, params: dict | None = None) -> dict:
    if active_for(user) >= config.MAX_ACTIVE_PER_USER:
        raise RuntimeError(f"You already have {config.MAX_ACTIVE_PER_USER} rollouts running. Wait for one to finish.")
    with _live_lock:
        if len(_live) >= config.MAX_ACTIVE_ROLLOUTS:
            raise RuntimeError("The explorer is at capacity right now. Try again in a few minutes.")
    run = store.create({
        "id": time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6],
        "user": user, "task_id": task["id"], "domain": task["domain"], "title": task["title"],
        "facets": task.get("facets"), "model": model, "provider": provider, "judge": judge,
        "status": "queued", "harness": "opencode", "endpoint": endpoint, "params": params or {},
    })
    r = Rollout(run, token, agent_key)
    with _live_lock:
        _live[r.id] = r
    _pool.submit(r.execute)
    return run


def live(run_id: str) -> Rollout | None:
    with _live_lock:
        return _live.get(run_id)


def events(run_id: str, after: int = 0) -> list[dict]:
    r = live(run_id)
    if r is not None:
        with r._lock:
            return r.events[after:]
    return store.read_events(run_id, after)


def cancel(run_id: str) -> bool:
    r = live(run_id)
    if r is None:
        return False
    r.cancelled.set()
    # Say so at once: the worker may sit in a blocking sandbox call for a while after the kill.
    r.update(status="cancelled", finished_at=time.time(), cost=r.cost_now())
    r.phase("done", "error", "stopped by you")
    threading.Thread(target=r.stop_sandbox, daemon=True).start()
    return True
