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

BACKENDS = ("e2b", "hf", "daytona")

# Per-backend home directory. See the module docstring for what a wrong value costs.
_HOMES = {"e2b": "/home/user", "hf": "/root", "daytona": "/root"}

# THE TWO PROVIDERS TAKE DIFFERENT THINGS, AND THAT IS NOT A DETAIL TO PAPER OVER.
#
# E2B takes a prebuilt TEMPLATE, with the data-science stack and opencode already baked in -- which is
# also what lets the harness's "is opencode already installed" check short-circuit instead of curling
# an installer into every sandbox. It has no `image` parameter at all.
#
# HF takes an IMAGE, and installs opencode at runtime.
#
# Sizing is baked into the E2B template at BUILD time and cannot be set per sandbox: `Sandbox.create`
# has no cpu/memory parameters. That matters here because the defaults are too small -- pandas wants
# roughly 3-5x a CSV's size in RAM and these tables reach 2.7 GB, so a 1 GB sandbox is OOM-killed, and
# an OOM-killed rollout files no answer, which is identical in the reward to a model that could not do
# the task. The template is built at cpu=2, mem=4096; to change it, rebuild the template.
E2B_TEMPLATE = os.environ.get("E2B_TEMPLATE", "data-agent-opencode")
HF_FLAVOR = os.environ.get("HF_SANDBOX_FLAVOR", "cpu-basic")

# Used by the HF backend only; E2B carries the equivalent inside its template.
DEFAULT_IMAGE = os.environ.get(
    "DATA_AGENT_IMAGE", "docker.io/savatar101/env-data-agent-train:base"
)


def sandbox_home(backend: str) -> str:
    """Home directory the agent runs under, for `backend`."""
    try:
        return _HOMES[backend]
    except KeyError:
        raise ValueError(f"unknown sandbox backend {backend!r}; expected one of {BACKENDS}") from None


def build_backend(backend: str, *, image: str = DEFAULT_IMAGE, **kwargs: Any) -> SandboxBackend:
    """Construct a sandbox backend by name.

    Args:
        backend (`str`):
            `"e2b"` or `"hf"`.
        image (`str`, *optional*):
            Container image. Used by the HF backend ONLY -- E2B carries the equivalent inside
            `E2B_TEMPLATE`, and its constructor has no `image` parameter.

    Returns:
        A sandbox backend.
    """
    if backend == "e2b":
        from .e2b import E2BSandboxBackend

        return E2BSandboxBackend(
            template=E2B_TEMPLATE,
            # The agent pulls its task's tables from a Hugging Face bucket, so it needs the network.
            sandbox_kwargs={"allow_internet_access": True},
            **kwargs,
        )
    if backend == "hf":
        from .hf import HFSandboxBackend

        return HFSandboxBackend(image=image, flavor=HF_FLAVOR, **kwargs)
    if backend == "daytona":
        from .daytona import DaytonaSandboxBackend
        return DaytonaSandboxBackend(image=image, **kwargs)
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
    try:
        import daytona  # noqa: F401
        out["daytona"] = bool(os.environ.get("DAYTONA_API_KEY"))
    except ImportError:
        out["daytona"] = False
    return out


def describe() -> str:
    """One line for the startup log: which backends are usable, and how each is configured."""
    got = available()
    return (
        f"e2b={'ready' if got.get('e2b') else 'unavailable'} (template={E2B_TEMPLATE}), "
        f"hf={'ready' if got.get('hf') else 'unavailable'} (flavor={HF_FLAVOR}), "
        f"daytona={'ready' if got.get('daytona') else 'unavailable'}"
    )


__all__ = [
    "BACKENDS",
    "DEFAULT_IMAGE",
    "E2B_TEMPLATE",
    "HF_FLAVOR",
    "SandboxBackend",
    "available",
    "build_backend",
    "describe",
    "sandbox_home",
]
