"""Swappable sandbox backends. Pick one per rollout; nothing else in this env knows which.

VENDORED, ON PURPOSE. `base.py`, `e2b.py` and `hf.py` are copies of
`OpenEnv/envs/opencode_env/sandbox/` at the revision pinned in this project's README. They are not
imported from there, for two reasons that both matter:

  * `openenv-opencode-env` is NOT published to PyPI -- only `openenv` is -- so `opencode_env.sandbox`
    is not reachable from an installed environment at all;
  * an environment in this repo has to stand on its own. Someone should be able to copy this
    directory, `uv sync`, and get a working env without also cloning OpenEnv.

The cost is a copy that can drift, and the honest answer to that is that this is a snapshot, not a
dependency. The three files have no upward imports, so they lift cleanly.

THE SANDBOX HOME IS THE WHOLE REASON `sandbox_home` EXISTS
E2B runs the agent as `user` with a home of `/home/user`; Hugging Face sandboxes run as root with
`/root`. Get it wrong and opencode writes its provider config where it cannot read it back, so the
agent starts with NO MODEL CONFIGURED and makes zero model calls -- which arrives as a flat-zero
reward that looks exactly like a policy that cannot do the task. This is the single place that knows;
every other module asks rather than assuming.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import BgJob, ExecResult, SandboxBackend, SandboxHandle  # noqa: F401


logger = logging.getLogger(__name__)

BACKENDS = ("e2b", "hf")

# Per-backend home directory. See the module docstring for what a wrong value costs.
_HOMES = {"e2b": "/home/user", "hf": "/root"}

# pandas / numpy / scipy and friends. A task's instruction names them, so a bare image turns every
# rollout into a package-install exercise the reward does not pay for.
DEFAULT_IMAGE = "docker.io/savatar101/env-data-agent-train:base"


def sandbox_home(backend: str) -> str:
    """Home directory the agent runs under, for `backend`."""
    try:
        return _HOMES[backend]
    except KeyError:
        raise ValueError(f"unknown sandbox backend {backend!r}; expected one of {BACKENDS}") from None


def build_backend(backend: str, *, image: str = DEFAULT_IMAGE, **kwargs: Any) -> SandboxBackend:
    """Construct a sandbox backend by name.

    Imported lazily so a machine with only one provider's SDK installed can still serve the other --
    an unconditional import at module scope would make a missing `e2b` package break the HF path too.
    """
    if backend == "e2b":
        from .e2b import E2BSandboxBackend

        return E2BSandboxBackend(image=image, **kwargs)
    if backend == "hf":
        from .hf import HFSandboxBackend

        return HFSandboxBackend(image=image, **kwargs)
    raise ValueError(f"unknown sandbox backend {backend!r}; expected one of {BACKENDS}")


def available() -> dict[str, bool]:
    """Which backends this machine can run: SDK importable AND credentials present.

    Reported by `capabilities()` so a caller learns before dispatching rollouts, rather than after
    paying for a sandbox that could never have started.
    """
    out: dict[str, bool] = {}
    try:
        import e2b  # noqa: F401

        out["e2b"] = bool(os.environ.get("E2B_API_KEY"))
    except ImportError:
        out["e2b"] = False
    try:
        from huggingface_hub import get_token

        out["hf"] = bool(os.environ.get("HF_TOKEN") or get_token())
    except ImportError:
        out["hf"] = False
    return out


def describe() -> str:
    """One line for the startup log: which backends are usable here."""
    return ", ".join(f"{name}={'ready' if ok else 'unavailable'}" for name, ok in available().items())


__all__ = [
    "BACKENDS",
    "DEFAULT_IMAGE",
    "SandboxBackend",
    "available",
    "build_backend",
    "describe",
    "sandbox_home",
]
