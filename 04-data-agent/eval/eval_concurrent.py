#!/usr/bin/env python
"""Concurrent Harbor eval across several engines at once, with every trace kept.

Supersedes `eval_pass_at_k.py`. Three things changed, each for a measured reason:

1. ASYNCIO WITH LAYERED SEMAPHORES, not `ThreadPoolExecutor.map`. `pool.map` yields results in
   SUBMISSION order, so one slow rollout stalls the result stream and every partial write queued
   behind it. Here results are consumed as they complete.

2. A CLIENT-SIDE CLOCK. `HarborRolloutResult.wall_s` starts inside the server's tool body, i.e.
   AFTER the server admits the rollout, so any queueing upstream of that is invisible to it. That is
   how a server that admitted only 40 concurrent rollouts looked like it was running 150: the
   throughput figure was `concurrency / mean_duration`, an identity, not a measurement. We record
   `submit_s` ourselves and report `queue_s = submit_s - wall_s`.

3. CRASH-SAFE TRACES. One JSONL line per rollout, flushed and fsync'd the moment it returns, instead
   of one JSON array at the end. A crash at task 240 of 250 used to lose all 239.

The engine is named PER ROLLOUT, so one engineless `openenv harbor serve` can host every arm at once
and the task tree and sandbox templates are paid for once. Verified: `_resolve_and_run` takes the
`if llm_url:` branch and never consults the server's default engine.

    python tools/eval_concurrent.py --server http://HOST:8200 \
        --arms tools/arms.example.json --indices @tools/indices_250_shuffled.txt \
        --concurrency 64 --trace-dir logs/traces/
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import random
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_pass_at_k as epk  # the brake + breaker live there; one source of truth

logger = logging.getLogger("eval_concurrent")

# Base URLs are deployment knowledge, not an abstraction — which is why this table lives in the
# driver and NOT in OpenEnv. The library's primitive is (llm_url, model, api_key, auth_header); a
# `--provider` enum there would immediately need a `--provider-url` escape hatch for vLLM and
# self-hosted routers and would have bought nothing.
PROVIDERS = {
    "openai":    ("https://api.openai.com/v1",        "OPENAI_API_KEY",    "Authorization"),
    "anthropic": ("https://api.anthropic.com/v1",     "ANTHROPIC_API_KEY", "Authorization"),
    "hf":        ("https://router.huggingface.co/v1", "HF_TOKEN",          "Authorization"),
}

# Account ceilings, per sandbox backend. ARGUMENTS, not constants: they are a property of whoever is
# running this, not of the code. Nothing in OpenEnv knows or should know these numbers.
SANDBOX_CAPS = {"e2b": 500, "modal": 100, "docker": 32, "daytona": 100}

# A capacity rejection is the server saying "not now", answered BEFORE any sandbox is created. It is
# backpressure, not a failure: it must not touch the circuit breaker (which counts unscorable
# rollouts) nor the connection brake (which counts refusals). But it must be COUNTED — a sweep that
# reports 44 rollouts/min while silently retrying 300 rejections is the same headline-number trap
# this whole driver exists to avoid.
CAPACITY_MARKERS = ("CAPACITY_REACHED", "capacity", "SessionCapacityError")

# SANDBOX-side failures. These look like connection errors and match the global brake's markers
# ("timed out", "stream"), but they are a property of ONE rollout's sandbox, not of the harbor
# server — and treating them as systemic halts a sweep whose server is answering /health in 2 ms.
# Measured at 281 concurrent: 36 of 50 failures were the E2B exec channel
# ("the stream didn't open within 'request_timeout' (60.0 seconds)"), the dominant hard-failure mode
# across 13,200 trials. They are retried per rollout and never counted against the brake.
SANDBOX_MARKERS = (
    "the stream didn't open",
    "connection to sandbox",
    "ended before the stream completed",
    "Agent install failed",
    "sandbox timeout",
    "exit 137",          # the installer was OOM-killed inside the sandbox
)


def _is_sandbox_failure(err: str) -> bool:
    return any(m in err for m in SANDBOX_MARKERS)


class Arm:
    """One (engine, model, credential) triple being evaluated."""

    def __init__(self, name, base_url, model, api_key_env="", auth_header="Authorization",
                 concurrency=0, harness=""):
        self.name, self.base_url, self.model = name, base_url, model
        self.auth_header = auth_header or "Authorization"
        self.api_key_env = api_key_env
        # BY NAME, resolved here, never echoed. The key reaches the sandbox as nothing at all — the
        # agent's credential is the capture session id.
        self.api_key = os.environ.get(api_key_env, "") if api_key_env else ""
        self.concurrency = concurrency
        self.harness = harness
        self.sem: asyncio.Semaphore | None = None

    @property
    def ready(self) -> tuple[bool, str]:
        if self.api_key_env and not self.api_key:
            return False, f"${self.api_key_env} is unset"
        return True, ""


def load_arms(path: str) -> list[Arm]:
    spec = json.loads(Path(path).read_text())
    arms = []
    for a in spec:
        prov = a.get("provider", "")
        base, keyenv, auth = PROVIDERS.get(prov, ("", "", "Authorization"))
        arms.append(Arm(
            name=a["name"],
            base_url=a.get("base_url") or base,
            model=a["model"],
            api_key_env=a.get("api_key_env", keyenv),
            auth_header=a.get("auth_header", auth),
            concurrency=int(a.get("concurrency", 0)),
            harness=a.get("harness", ""),
        ))
    return arms


def selected_arms(args) -> list[Arm]:
    """Expand a single endpoint into harness arms under the same global semaphore."""
    if args.arms:
        engines = load_arms(args.arms)
    else:
        if not args.model:
            raise ValueError("--model is required with --vllm-url")
        engines = [Arm("model", args.vllm_url.rstrip("/"), args.model)]
    requested = [h.strip() for h in (args.harnesses or "").replace("+", ",").split(",") if h.strip()]
    if len(requested) != len(set(requested)):
        raise ValueError("--harnesses contains duplicates")
    out = []
    for engine in engines:
        for harness in requested or [engine.harness or args.harness]:
            name = f"{engine.name}--{harness}" if requested else engine.name
            if not name or Path(name).name != name or name in (".", ".."):
                raise ValueError("arm names must be plain filenames")
            out.append(Arm(name, engine.base_url, engine.model, engine.api_key_env,
                           engine.auth_header, engine.concurrency, harness))
    if len({a.name for a in out}) != len(out):
        raise ValueError("arm names must be unique")
    return out


def parse_indices(spec: str) -> list[int]:
    """Order-preserving. NEVER sorted.

    The dispatch order IS the sampling order, and on an easy->hard ordered suite a sorted partial run
    is an easy-prefix score: one model read 0.830 over indices 0-146 and 0.553 over 147-249, so
    "140/250 measured" hid which half. A fixed shuffle makes partial coverage an unbiased sample.
    """
    if spec.startswith("@"):
        spec = Path(spec[1:]).read_text()
    out, seen = [], set()
    for tok in spec.replace("\n", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


class TraceWriter:
    """One JSONL per arm, durable per line.

    `flush()` alone only moves bytes into the page cache; a node that dies still loses them. fsync is
    ~0.1 ms against rollouts that take 26-612 s, so there is no reason to batch.
    """

    def __init__(self, root: Path):
        root.mkdir(parents=True, exist_ok=True)
        self.root, self._fh = root, {}

    def write(self, arm: str, row: dict) -> None:
        fh = self._fh.get(arm)
        if fh is None:
            fh = self._fh[arm] = open(self.root / f"{arm}.jsonl", "a", buffering=1)
        fh.write(json.dumps(row, default=str) + "\n")
        fh.flush()
        os.fsync(fh.fileno())

    def close(self):
        for fh in self._fh.values():
            try:
                fh.close()
            except Exception:
                pass




# ── client pooling ──────────────────────────────────────────────────────────────────────────────────
# WHY. Measured: queue_s ~ 1.81 * C. That linear growth is a fixed-rate stage in front of the rollout,
# and it is the single thing capping 1000 rollouts at ~45 min no matter how high --concurrency goes.
# The cause is that `HarborSessionFactory.create()` calls `new_client()` per rollout and hands the
# session `owns_env=True`, so every rollout opens its own websocket and claims an env session, then
# closes it. Against a single-process uvicorn those connects serialise.
#
# `new_client()` is documented as overridable "so a caller can substitute a transport", which is the
# supported seam. One client per WORKER THREAD, reused across that thread's rollouts: a thread runs
# one rollout at a time, so this never trips `ConcurrencyError: cannot call recv while another
# coroutine is already running recv`, which is what a client shared across CONCURRENT rollouts causes.
#
# THE TRADE, stated plainly: a live client holds an env session for its whole life, so a pool of N
# clients pins N sessions on the server. That is bounded and predictable. The status quo is worse --
# a killed client leaks its session and they accumulate until CAPACITY_REACHED, which is exactly how
# a 1000-rollout run fell from 82% graded to 18% today.
_TLS = threading.local()

# Every pooled client, so they can be closed at the end of the run. Without this the websockets stay
# open until GC and each one holds an env session on the server -- the same leak that filled a server
# to 400/400 today and dropped a 1000-rollout run from 82% graded to 18%. A pool is only better than
# per-rollout clients if it is actually released.
_POOL_LOCK = threading.Lock()
_ALL_POOLED: list = []


def close_pooled_clients() -> int:
    """Close every pooled client. Returns how many were closed."""
    with _POOL_LOCK:
        clients, _ALL_POOLED[:] = list(_ALL_POOLED), []
    n = 0
    for c in clients:
        try:
            c.really_close(); n += 1
        except Exception:  # noqa: BLE001 - teardown must not mask the run's result
            pass
    return n


class _PooledClient:
    """A HarborEnv whose `close()` is a no-op, so the session cannot dispose of a pooled client."""

    def __init__(self, inner):
        object.__setattr__(self, "_inner", inner)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_inner"), name)

    def close(self):  # the session calls this after every rollout; pooling means ignoring it
        return None

    def really_close(self):
        object.__getattribute__(self, "_inner").close()


def _thread_factory(args, arm):
    """One factory + one client per worker thread, built once.

    `indices=None` matters: with an explicit index list `tasks()` issues one `get_task` per index,
    but with none it takes the `get_task_range` branch and fetches the whole split in ONE call. That
    turns 2 HTTP round-trips per rollout into 2 per thread.
    """
    from harbor_env.harness import HarborSessionFactory

    harness = arm.harness or args.harness
    key = (arm.name, args.split, harness, args.sandbox)
    if getattr(_TLS, "key", None) == key:
        return _TLS.factory

    class _Pooled(HarborSessionFactory):
        def new_client(self):
            c = getattr(_TLS, "client", None)
            if c is None:
                from openenv.harbor.client import HarborEnv
                c = _TLS.client = _PooledClient(HarborEnv(
                    base_url=self.server_url,max_message_size_mb=self._max_message_size_mb,
                    websocket_ping_interval_s=None,websocket_ping_timeout_s=None,
                ))
                with _POOL_LOCK:
                    _ALL_POOLED.append(c)
            return c

    f = _Pooled(
        args.server, split=args.split, harness=harness, sandbox=args.sandbox,
        llm_url=arm.base_url, model=arm.model, api_key=arm.api_key,
        auth_header=arm.auth_header,
        # NO reward_key here. It is a per-TASK choice on a heterogeneous suite, and the factory is
        # now shared across a thread's rollouts -- baking one in sends the whole preference string
        # ("correctness,reward") as a literal key name, which every task refuses, and the retry then
        # spins forever because a learned key can never reach a factory built once.
        agent_timeout_sec=args.agent_timeout, agent_step_limit=args.agent_step_limit,
        # An explicit policy also selects the same native harness settings as training.
        sampling=({"temperature": args.temperature, "top_p": 1.0, "top_k": 0}
                  if getattr(args, "temperature", None) is not None else None),
    )
    f.tasks()                      # one bulk fetch, on this thread, once
    _TLS.key, _TLS.factory = key, f
    _TLS.rows_by_index = {int(r["task_index"]): r for r in f.prompt_rows()}
    return f


def _agent_conversations(result) -> list[dict]:
    """The readable transcript, at EITHER tier.

    Not `fetch_proxy_trace()`: that is deliberately empty for an eval-tier rollout (there are no
    token fields to train on), and reading it here is why an earlier sweep recorded "no trace" for
    every cell. `result.conversations` is present either way.

    Agent-role only, and the LONGEST one — several harnesses emit one agent conversation PER STEP,
    each carrying the accumulated history, so they are strictly nested prefixes and taking `[0]`
    yields the SHORTEST. Measured: terminus-2 at n_turns=5 produced conversations of 2, 4, 6, 8 and
    10 messages, every pair a prefix of the next.
    """
    convs = list(getattr(result, "conversations", None) or [])
    agent = [c for c in convs if (getattr(c, "role", "") or (c.get("role") if isinstance(c, dict) else "")) == "agent"]
    pool = agent or convs
    if not pool:
        return []

    def msgs(c):
        m = getattr(c, "messages", None)
        if m is None and isinstance(c, dict):
            m = c.get("messages")
        return list(m or [])

    best = max(pool, key=lambda c: len(msgs(c)))
    out = []
    for m in msgs(best):
        if hasattr(m, "model_dump"):
            out.append(m.model_dump())
        elif isinstance(m, dict):
            out.append(m)
        else:
            out.append({"role": "unknown", "content": str(m)})
    return out


def _blocking_rollout(args, arm, index: int, rep: int) -> dict:
    """One rollout, synchronous. Runs on a worker thread; the semaphores are held by the caller."""
    f = _thread_factory(args, arm)
    row = _TLS.rows_by_index.get(int(index))
    if row is None:
        raise KeyError(f"task index {index} is not in split {args.split!r}")
    # Let OpenEnv select from the SAME verifier result. Discovering a key by
    # rerunning the entire agent turns reward routing into an unintended resample.
    f.reward_key = args.reward_key
    session = f.create(row["prompt"])
    try:
        session.wait_for_completion()
    finally:
        session.close()
    result = session.result
    if result is None:
        # A failed transport cannot remain the thread's pooled connection.
        client=getattr(_TLS,'client',None)
        _TLS.client=None
        if client is not None:
            try: client.really_close()
            except Exception: pass
        # HarborSession returns a status code when its transport fails. Treating that as an
        # empty successful result bypassed every infrastructure retry in the old driver.
        raise RuntimeError("No rollout result returned by Harbor transport")
    if getattr(result, "wall_s", None) is None and not getattr(result, "n_turns", 0):
        client=getattr(_TLS,'client',None)
        _TLS.client=None
        if client is not None:
            try: client.really_close()
            except Exception: pass
        raise RuntimeError(getattr(result, "error", "") or "No rollout result returned by Harbor transport")
    reward = session.verify([]).env_reward
    n_turns = getattr(result, "n_turns", 0) or 0

    # A graded 0.0 from a rollout that made ZERO model calls is the verifier's missing-answer
    # default, not a measurement of the policy. Recording it as 0.0 would drag every mean down with
    # infrastructure noise, so it becomes None (excluded) and is reported separately.
    infra_zero = ""
    if n_turns == 0:
        infra_zero = "0 turns: the agent never reached the proxy"
        reward = None

    capture_file = ""
    if getattr(args, "capture_dir", ""):
        import uuid
        root = Path(args.capture_dir) / arm.name
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{index}-{rep}-{uuid.uuid4().hex}.json"
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result.model_dump(mode="json")))
        temporary.replace(path)
        capture_file = str(path)

    return {
        "arm": arm.name, "model": arm.model, "index": index, "rep": rep,
        "harness": arm.harness or args.harness, "capture_file": capture_file,
        "reward": None if reward is None else float(reward),
        "infra_zero": infra_zero,
        "n_turns": n_turns,
        "ok": bool(getattr(result, "ok", False)),
        "task_id": getattr(result, "task_id", "") or "",
        "trial_name": getattr(result, "trial_name", "") or "",
        # SERVER-side duration. Starts after admission, so it cannot see queueing — which is exactly
        # why `submit_s` is recorded alongside it by the caller.
        "wall_s": getattr(result, "wall_s", None),
        # ALL keys, not just the chosen one: a headline built on one key should still let a
        # reader see what else the verifier measured.
        "rewards": dict(getattr(result, "rewards", None) or {}),
        "reward_key": getattr(result, "reward_key", "") or "",
        "findings": list(getattr(result, "findings", None) or [])[:5],
        "rollout_type": getattr(result, "rollout_type", ""),
        "capture_level": getattr(result, "capture_level", ""),
        "n_trainable_tokens": getattr(result, "n_trainable_tokens", 0) or 0,
        # What the provider made us change to be accepted at all. An arm that silently dropped
        # `temperature` did not evaluate what was asked for, and a score without this is not
        # reproducible.
        "param_fixes": [str(p) for p in (getattr(result, "param_fixes", None) or [])],
        "conversations": _agent_conversations(result),
        "error": (getattr(result, "error", "") or "")[:400],
    }


async def run_one(args, arm: Arm, index: int, rep: int, sems, counters) -> dict:
    """Acquire OUTERMOST-FIRST, then run. Retry infrastructure, never a score."""
    attempt = 0
    while True:
        attempt += 1
        halted = epk.SERVER_DOWN
        if halted:
            return {"arm": arm.name, "index": index, "rep": rep, "reward": None,
                    "skipped": True, "skip_reason": halted, "n_turns": 0, "ok": False}

        t0 = time.monotonic()
        admitted_at = None
        # Order matters: global -> server -> sandbox -> provider. Taking the SCARCEST (the provider's
        # rate limit) last means a rollout never sits on a server session while it waits for one.
        async with sems["global"]:
            async with sems["server"]:
                async with sems["sandbox"]:
                    async with arm.sem:
                        admitted_at = time.monotonic()
                        try:
                            row = await asyncio.to_thread(_blocking_rollout, args, arm, index, rep)
                            err = row.get("error") or ""
                        except Exception as exc:  # noqa: BLE001
                            row, err = None, str(exc)

        submit_s = time.monotonic() - t0

        if row is not None and not err:
            epk._note_connectivity(None, args.halt_after_conn_fails)
            epk._note_result(arm.name, row.get("reward") is not None, args.pause_after)
            row["submit_s"] = round(submit_s, 2)
            row["admission_wait_s"] = round(admitted_at - t0, 2)
            row["service_submit_s"] = round(time.monotonic() - admitted_at, 2)
            row["service_overhead_s"] = round(max(0.0, row["service_submit_s"] - (row.get("wall_s") or 0.0)), 2)
            row["queue_s"] = round(max(0.0, submit_s - (row.get("wall_s") or 0.0)), 2)
            row["attempt"] = attempt
            return row

        if any(m in err for m in CAPACITY_MARKERS):
            counters["capacity_rejections"] += 1
            if attempt <= args.max_retries:
                # Jittered, and requeued rather than spun on, so arm round-robin is preserved.
                await asyncio.sleep(min(30.0, 2.0 * attempt) * (0.5 + random.random()))
                continue

        # A sandbox failure is infrastructure for THIS rollout: retry it, and explicitly clear the
        # brake's streak so it is never mistaken for the server going away.
        if _is_sandbox_failure(err):
            counters["sandbox_failures"] += 1
            epk._note_connectivity(None, args.halt_after_conn_fails)
            if attempt <= args.max_retries:
                await asyncio.sleep(min(30.0, 3.0 * attempt) * (0.5 + random.random()))
                continue

        epk._note_connectivity(err, args.halt_after_conn_fails)
        if attempt <= args.max_retries and (
            any(m in err for m in epk._CONN_MARKERS)
            or "No rollout result returned by Harbor transport" in err
        ):
            await asyncio.sleep(min(30.0, 2.0 * attempt) * (0.5 + random.random()))
            continue

        epk._note_result(arm.name, False, args.pause_after)
        logger.warning("%s task %d rep %d failed: %s", arm.name, index, rep, err[:160])
        return {**(row or {}), "arm": arm.name, "model": arm.model,
                "harness": arm.harness or args.harness, "index": index, "rep": rep,
                "verifier_reward_on_failure": (row or {}).get("reward"), "reward": None,
                "n_turns": (row or {}).get("n_turns", 0), "ok": False, "error": err[:400], "attempt": attempt,
                "submit_s": round(submit_s, 2)}


def summarise(rows: list[dict], arms: list[Arm], indices: list[int], counters: dict) -> dict:
    """Scores on the COMMON item set, with the drop matrix that explains what is missing.

    A headline score over "whatever each arm managed to grade" is not comparable across arms: they
    rate-limit differently and therefore drop different tasks. And a common-set score with an
    invisible drop matrix is exactly the number that hides its own cost.
    """
    by_arm = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)

    graded_idx = {a.name: {r["index"] for r in by_arm[a.name] if r.get("reward") is not None}
                  for a in arms}
    common = set.intersection(*graded_idx.values()) if graded_idx and all(graded_idx.values()) else set()

    out = {"n_requested": len(indices), "n_common": len(common),
           "common_indices": sorted(common), "arms": {}, **counters}
    for a in arms:
        rs = by_arm[a.name]
        graded = [r for r in rs if r.get("reward") is not None]
        in_common = [r for r in graded if r["index"] in common]
        walls = [r["wall_s"] for r in rs if r.get("wall_s")]
        queues = [r["queue_s"] for r in rs if r.get("queue_s") is not None]
        walls.sort()
        out["arms"][a.name] = {
            "model": a.model,
            "harness": a.harness,
            "attempted": len(rs),
            "graded": len(graded),
            # Never averaged as 0.0 — an ungraded rollout is an exclusion, not a wrong answer.
            "ungraded": len(rs) - len(graded),
            "infra_zero": sum(1 for r in rs if r.get("infra_zero")),
            "score_common": (sum(r["reward"] for r in in_common) / len(in_common)) if in_common else None,
            "score_all_graded": (sum(r["reward"] for r in graded) / len(graded)) if graded else None,
            "mean_turns": (sum(r.get("n_turns", 0) for r in rs) / len(rs)) if rs else 0,
            "rollout_types": dict(_tally(r.get("rollout_type", "") for r in rs)),
            "wall_p50": walls[len(walls) // 2] if walls else None,
            "wall_p90": walls[int(len(walls) * 0.9)] if walls else None,
            "wall_max": walls[-1] if walls else None,
            # If this is large the client believed a concurrency the server never granted.
            "queue_p50": sorted(queues)[len(queues) // 2] if queues else None,
            "param_fixes": sorted({p for r in rs for p in (r.get("param_fixes") or [])}),
            # WHICH indices, not how many.
            "graded_indices": sorted(graded_idx[a.name]),
            "missing_from_common": sorted(set(indices) - graded_idx[a.name]),
        }
    return out


def _tally(it):
    d = defaultdict(int)
    for x in it:
        d[x] += 1
    return d


async def amain(args) -> int:
    arms = selected_arms(args)
    for a in arms:
        ok, why = a.ready
        if not ok:
            raise SystemExit(f"arm {a.name}: {why}")
    indices = parse_indices(args.indices)
    if not indices or min(indices) < 0:
        raise ValueError("--indices must select non-negative task indices")
    if args.concurrency < 1 or args.repeat < 1:
        raise ValueError("--concurrency and --repeat must be positive")
    if args.temperature is not None and (not math.isfinite(args.temperature) or args.temperature <= 0):
        raise ValueError("explicit TiTO sampling requires a finite positive temperature")
    manifest = {
        "server": args.server, "split": args.split, "indices": indices, "repeat": args.repeat,
        "arms": [{"name": a.name, "model": a.model, "base_url": a.base_url,
                  "harness": a.harness, "api_key_env": a.api_key_env} for a in arms],
        "temperature": args.temperature, "agent_step_limit": args.agent_step_limit,
        "agent_timeout": args.agent_timeout, "sandbox": args.sandbox,
        "reward_key": args.reward_key,
    }
    if args.dry_run:
        print(json.dumps({**manifest, "rollouts": len(arms) * len(indices) * args.repeat,
                          "global_concurrency": args.concurrency}, indent=2))
        return 0
    trace_root = Path(args.trace_dir)
    trace_root.mkdir(parents=True, exist_ok=True)
    manifest_path = trace_root / "eval_config.json"
    if args.resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("--resume requires the same saved evaluation configuration")
    elif manifest_path.exists() or any(trace_root.glob("*.jsonl")):
        raise ValueError("trace directory already contains an evaluation; use --resume or a new directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    cap = args.sandbox_concurrency or SANDBOX_CAPS.get(args.sandbox, 64)
    sems = {
        "global":  asyncio.Semaphore(args.concurrency),
        "server":  asyncio.Semaphore(args.server_concurrency or args.concurrency),
        "sandbox": asyncio.Semaphore(cap),
    }
    for a in arms:
        a.sem = asyncio.Semaphore(a.concurrency or args.concurrency)

    # ROUND-ROBIN across arms, not arm-major. Every arm then has partial coverage at any instant, so
    # an interrupted sweep is salvageable and no arm's data comes systematically from a different
    # period of server health.
    jobs = [(a, i, rep) for rep in range(args.repeat) for i in indices for a in arms]
    resumed = {}
    if args.resume:
        allowed = {(a.name, i, rep) for a, i, rep in jobs}
        for arm in arms:
            path = trace_root / f"{arm.name}.jsonl"
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    # Do not append to a truncated line: the existing valid results remain intact.
                    raise ValueError(f"incomplete JSONL record in {path}; repair the trailing record before resuming")
                key = (row.get("arm"), row.get("index"), row.get("rep"))
                reward = row.get("reward")
                if key in allowed and isinstance(reward, (int, float)) and math.isfinite(reward) and row.get("n_turns", 0) > 0:
                    # Keep the first graded attempt, including reward=0. Never select best-of-retries.
                    resumed.setdefault(key, row)
        jobs = [(a, i, rep) for a, i, rep in jobs if (a.name, i, rep) not in resumed]
        for row in resumed.values():
            epk._note_result(row["arm"], True, args.pause_after)
    limit = getattr(args, "max_new_rollouts", 0)
    if limit < 0:
        raise ValueError("--max-new-rollouts must be non-negative")
    if limit:
        jobs = jobs[:limit]

    print(f"arms        {', '.join(f'{a.name}({a.model})' for a in arms)}")
    print(f"tasks       {len(indices)}  x repeat {args.repeat}  = {len(jobs)} rollouts")
    print(f"concurrency global={args.concurrency} server={sems['server']._value} "
          f"sandbox[{args.sandbox}]={cap} per-arm={[a.sem._value for a in arms]}")
    print(f"traces      {args.trace_dir}\n", flush=True)

    # `asyncio.to_thread` runs on the loop's DEFAULT executor, which Python sizes at
    # min(32, cpu_count + 4). That is a hard ceiling of 32 concurrent rollouts no matter what
    # --concurrency says, and it is invisible: the semaphores all admit, the tasks all start, and
    # they queue inside the executor where nothing reports it. Measured: 32 live capture sessions
    # against --concurrency 320. Same shape as the anyio CapacityLimiter(40) that capped the server.
    import concurrent.futures
    pool = concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency, thread_name_prefix="rollout"
    )
    asyncio.get_running_loop().set_default_executor(pool)

    counters = defaultdict(int)
    writer = TraceWriter(Path(args.trace_dir))
    rows: list[dict] = list(resumed.values())
    t0 = time.monotonic()
    try:
        tasks = [asyncio.create_task(run_one(args, a, i, rep, sems, counters))
                 for (a, i, rep) in jobs]
        done_n = 0
        for fut in asyncio.as_completed(tasks):       # COMPLETION order, not submission order
            row = await fut
            rows.append(row)
            writer.write(row["arm"], row)             # durable before anything else happens
            done_n += 1
            if done_n % max(1, args.progress_every) == 0:
                el = time.monotonic() - t0
                print(f"  {done_n}/{len(jobs)}  {done_n/el*60:.1f}/min  "
                      f"capacity_rejections={counters['capacity_rejections']}", flush=True)
    finally:
        writer.close()
        pool.shutdown(wait=False, cancel_futures=True)
        freed = close_pooled_clients()
        print(f"released {freed} pooled client(s) / env session(s)")

    el = time.monotonic() - t0
    summary = summarise(rows, arms, indices, dict(counters))
    # MEASURED from real completions, never `concurrency / mean_duration`.
    summary["elapsed_s"] = round(el, 1)
    summary["throughput_per_min"] = round((len(rows) - len(resumed)) / max(el, 1e-9) * 60, 2)
    summary["resumed_rollouts"] = len(resumed)
    summary["repeat"] = args.repeat
    summary["expected_rollouts"] = len(indices) * len(arms) * args.repeat
    summary["coverage_complete"] = all(
        s["graded"] == len(indices) * args.repeat for s in summary["arms"].values()
    )
    if args.repeat == 1:
        for s in summary["arms"].values():
            s["pass_at_1"] = s["score_all_graded"]
            s["pass_at_1_full_set"] = s["pass_at_1"] if s["graded"] == len(indices) else None
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"summary": summary}, indent=2, default=str))

    print(f"\n{'='*78}\nelapsed {el/60:.1f} min   measured throughput {summary['throughput_per_min']}/min")
    print(f"common item set: {summary['n_common']}/{summary['n_requested']}")
    for name, s in summary["arms"].items():
        sc = "None" if s["score_common"] is None else f"{s['score_common']:.4f}"
        print(f"  {name:12s} common={sc} graded={s['graded']}/{s['attempted']} "
              f"ungraded={s['ungraded']} turns={s['mean_turns']:.1f} "
              f"wall p50/p90/max={s['wall_p50']}/{s['wall_p90']}/{s['wall_max']} "
              f"queue_p50={s['queue_p50']} types={s['rollout_types']}")
        if s["param_fixes"]:
            print(f"               param_fixes={s['param_fixes']}")
    if summary["n_common"] < 0.9 * summary["n_requested"]:
        print("\n!! common set < 90% of requested — these scores are NOT publishable as a comparison")
    print(f"summary -> {args.out}")
    return 2 if args.require_complete and not summary["coverage_complete"] else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--server", required=True, help="a running `openenv harbor serve` (may be engineless)")
    engine = p.add_mutually_exclusive_group(required=True)
    engine.add_argument("--arms", help="JSON file of engines to evaluate")
    engine.add_argument("--vllm-url", help="OpenAI-compatible /v1 endpoint; use with --model")
    p.add_argument("--model", default="", help="exact model ID advertised by the endpoint")
    p.add_argument("--split", default="AdithyaSK/data_agent_rl_environment_eval")
    p.add_argument("--indices", default="@tools/indices_250_shuffled.txt",
                   help="comma-separated, or @file. ORDER IS PRESERVED and must not be sorted.")
    p.add_argument("--repeat", type=int, default=1, help="samples per (arm, task)")
    p.add_argument("--harness", default="mini-swe-agent")
    p.add_argument("--harnesses", default="", help="comma- or plus-separated harness matrix; shares one global concurrency limit")
    p.add_argument("--temperature", type=float, default=None,
                   help="explicit full-vocabulary TiTO policy, also selecting training harness settings")
    p.add_argument("--reward-key", default="",  # comma-separated preference order
                   help="which verifier key is the score. REQUIRED for a multi-reward suite; "
                        "without it every rollout is refused rather than silently combined.")
    p.add_argument("--sandbox", default="e2b")
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--server-concurrency", type=int, default=0, help="0 = same as --concurrency")
    p.add_argument("--sandbox-concurrency", type=int, default=0,
                   help=f"0 = the account default for --sandbox {SANDBOX_CAPS}")
    p.add_argument("--agent-timeout", type=float, default=600.0)
    p.add_argument("--agent-step-limit", type=int, default=20)
    p.add_argument("--max-retries", type=int, default=3, help="infrastructure only; never a score")
    p.add_argument("--halt-after-conn-fails", type=int, default=15)
    p.add_argument("--pause-after", type=int, default=8)
    p.add_argument("--trace-dir", default="logs/traces")
    p.add_argument("--capture-dir", default="", help="retain full Harbor results, including exact token/logprob capture")
    p.add_argument("--resume", action="store_true", help="reuse the first graded result for each existing cell")
    p.add_argument("--require-complete", action="store_true", help="exit nonzero unless every requested cell is graded")
    p.add_argument("--dry-run", action="store_true", help="print the evaluation matrix without connecting or creating sandboxes")
    p.add_argument("--max-new-rollouts", type=int, default=0,
                   help="run only this many new cells, preserving the full manifest for --resume; 0 = all")
    p.add_argument("--out", default="logs/eval_concurrent.json")
    p.add_argument("--progress-every", type=int, default=10)
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.repeat > 1:
        # A k-sample number at temperature 0 is k copies of one trajectory. The engine must be
        # sampling, or pass@k is overstated while looking healthy.
        logger.warning("--repeat %d assumes the arm samples (temperature > 0); pass@k is "
                       "meaningless against a greedy endpoint", args.repeat)
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
