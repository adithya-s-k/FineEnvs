"""Which adapter reads which environment, and the one entry point the routes, the runner and the MCP bridge use.

    resolve(kind, id, token)   the environment, read by the adapter that fits it best (contract.Adapter.detect/confirm)
    summary / tasks / task / file / folder / random      what the pages show, for any environment
    materialize / run_task                               what the runner needs, for any environment that can run

Adding a format: write an adapter (contract.py, CUSTOM_ENVS.md) and put an instance in ADAPTERS. Order doesn't matter:
each says how sure it is, the surest that confirms wins. Linkers add links between tasks of different environments
(a converted dataset back to its source) without either adapter knowing the other.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from .. import catalog
from .contract import Adapter, Env
from .harbor import HarborAdapter
from .mimo import MiMoAdapter, harbor_aliases, harbor_to_mimo
from .rows import RowsAdapter

ADAPTERS: list[Adapter] = [HarborAdapter(), MiMoAdapter(), RowsAdapter()]
# Relations no single adapter knows, so neither has to know the other: a converted dataset's task and its source.
#   LINKERS   (env, ref, view) -> links shown on the task's page
#   ALIASERS  (env, ref) -> [(environment key, ref)] that are this very task (their rollouts are listed with it)
LINKERS: list[Callable[[Env, str, dict[str, Any]], list[dict[str, Any]]]] = [harbor_to_mimo]
ALIASERS: list[Callable[[Env, str], list[tuple[str, str]]]] = [harbor_aliases]
_choice: dict[tuple[str, str, str], tuple[str, float]] = {}


def register(adapter: Adapter) -> Adapter:
    """Add an adapter (a new format). Returns it, so it can be used as `register(MyAdapter())`."""
    ADAPTERS.append(adapter)
    return adapter


def adapter_by_id(aid: str) -> Adapter:
    return next(a for a in ADAPTERS if a.id == aid)


def _meta(kind: str, spec: str, token: str | None) -> dict[str, Any]:
    if kind != "dataset":
        raise LookupError("only datasets are read through adapters for now; Spaces have their own page")
    meta = dict(catalog.info(spec, token))   # checks the visitor may read it
    meta["id"] = spec
    return meta


def resolve(kind: str, spec: str, token: str | None = None) -> Env:
    """The environment, with the adapter that reads it. The choice is kept a minute (an index finishing can change it:
    a Harbor-tagged dataset whose tasks turn out to be packed into rows goes to the rows reader)."""
    spec = catalog.check_spec(spec)
    meta = _meta(kind, spec, token)
    key = (kind, spec, meta.get("sha") or "")
    hit = _choice.get(key)
    if hit and hit[1] > time.time():
        return Env(kind, spec, meta, adapter_by_id(hit[0]), token)
    for a in sorted(ADAPTERS, key=lambda a: -a.detect(kind, meta)):
        score = a.detect(kind, meta)
        if score <= 0:
            continue
        env = Env(kind, spec, meta, a, token)
        if score >= 1 or a.confirm(env):
            if a.settled(env):   # an adapter still finding out (an index running) isn't kept
                _choice[key] = (a.id, time.time() + 60)
            return env
    raise LookupError("no reader for this environment's format yet")


def _public(view: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in view.items() if not k.startswith("_")}


def describe_env(env: Env) -> dict[str, Any]:
    a = env.adapter
    return {"key": env.key, "kind": env.kind, "id": env.id, "adapter": a.id, "framework": a.framework, "name": a.name, "about": a.about}


# ── what pages show ──────────────────────────────────────────────────────────
def summary(spec: str, token: str | None = None, subset: str | None = None) -> dict[str, Any]:
    env = resolve("dataset", spec, token)
    s = env.adapter.summary(env, subset)
    if s.get("state") == "handoff":   # it found out it doesn't read this one after all
        _choice.pop(("dataset", env.id, env.meta.get("sha") or ""), None)
        env = resolve("dataset", spec, token)
        s = env.adapter.summary(env, subset)
    how = s.get("how") or {}
    return {"env": {**describe_env(env), "framework": how.get("framework") or env.adapter.framework},
            "info": {k: v for k, v in env.meta.items() if k != "all_tags"}, "collection": catalog.collection(spec), **s}


def tasks(spec: str, token: str | None = None, **kw: Any) -> dict[str, Any]:
    env = resolve("dataset", spec, token)
    return env.adapter.tasks(env, **kw)


def task(spec: str, ref: str, token: str | None = None) -> dict[str, Any]:
    env = resolve("dataset", spec, token)
    view = env.adapter.task(env, ref)
    options = env.adapter.run_options(env, ref, view)
    for linker in LINKERS:
        try:
            view.setdefault("links", []).extend(linker(env, ref, view))
        except Exception:  # noqa: BLE001 - a link that can't be worked out isn't shown
            continue
    out = _public(view)
    out.setdefault("collection", catalog.collection(spec))
    out["env"] = {**describe_env(env), "framework": view.get("framework") or env.adapter.framework}
    out["run"] = {"options": options, "note": "" if options else env.adapter.run_note}
    out["aliases"] = [{"env": k, "ref": r} for k, r in aliases(env, ref)]
    return out


def file(spec: str, ref: str, path: str, token: str | None = None) -> dict[str, Any]:
    env = resolve("dataset", spec, token)
    return env.adapter.file(env, ref, path)


def folder(spec: str, ref: str, path: str, token: str | None = None) -> dict[str, Any]:
    env = resolve("dataset", spec, token)
    return env.adapter.folder(env, ref, path)


def random(spec: str, subset: str | None = None, token: str | None = None) -> str:
    env = resolve("dataset", spec, token)
    return env.adapter.random(env, subset)


def data(spec: str, ref: str, name: str, params: dict[str, str], token: str | None = None) -> Any:
    env = resolve("dataset", spec, token)
    return env.adapter.data(env, ref, name, params)


def raw(spec: str, ref: str, path: str, token: str | None = None) -> tuple[bytes, str]:
    env = resolve("dataset", spec, token)
    return env.adapter.raw(env, ref, path)


def aliases(env: Env, ref: str) -> list[tuple[str, str]]:
    """The same task elsewhere: the adapter's own, and the aliasers'."""
    out = list(env.adapter.aliases(env, ref))
    for fn in ALIASERS:
        try:
            out += fn(env, ref)
        except Exception:  # noqa: BLE001 - an alias that can't be worked out isn't listed
            continue
    seen, uniq = {(env.key, ref)}, []
    for a in out:
        if a not in seen:
            seen.add(a)
            uniq.append(a)
    return uniq


# ── what the runner needs ────────────────────────────────────────────────────
def run_task(spec: str, ref: str, token: str | None = None) -> tuple[Env, dict[str, Any], list[dict[str, Any]]]:
    """The environment, what a run record says about the task, and its run options (checked again at run time)."""
    env = resolve("dataset", spec, token)
    view = env.adapter.task(env, ref)
    return env, env.adapter.run_task(env, ref), env.adapter.run_options(env, ref, view)


def materialize(spec: str, ref: str, token: str | None = None):
    env = resolve("dataset", spec, token)
    return env.adapter.materialize(env, ref)
