"""OpenEnv's Harbor server (`uvicorn harbor_env.server.app:app ...`), with HF Sandboxes given longer to start.

Harbor's hf-sandbox backend creates each task sandbox with huggingface_hub's `Sandbox.create`, which gives up after 120 s
(`start_timeout`) and Harbor does not pass another value. Sandboxes are HF Jobs, and when many start at once (the start
of a run, or a restart, with 128 rollouts in flight) some take longer: in one smoke all 8 rollouts started after a
restart failed that way. A failed start costs the rollout, so this process raises the default to
FINEPROOF_SANDBOX_START_TIMEOUT_SEC (600 s) before serving. Other arguments go to uvicorn unchanged.

    python jobs/harbor_server.py --host 127.0.0.1 --port 8200 --log-config harbor_logging.json
"""

import inspect
import os
import sys


def patch_sandbox_start_timeout(seconds: float) -> bool:
    """Make `Sandbox.create` wait `seconds` for a sandbox unless the caller says otherwise. False if not applicable."""
    try:
        from huggingface_hub import Sandbox
    except ImportError:  # an older huggingface_hub: no HF Sandbox, nothing to patch
        return False
    original = Sandbox.create
    if "start_timeout" not in inspect.signature(original).parameters:
        return False

    def create(cls, *args, **kwargs):
        kwargs.setdefault("start_timeout", seconds)
        return original(*args, **kwargs)

    create.__doc__ = original.__doc__
    Sandbox.create = classmethod(create)
    return True


def main() -> None:
    seconds = float(os.environ.get("FINEPROOF_SANDBOX_START_TIMEOUT_SEC", "600"))
    patched = patch_sandbox_start_timeout(seconds)
    print(f"HF Sandbox start timeout: {seconds:.0f}s" if patched else "HF Sandbox start timeout: library default",
          flush=True)
    import uvicorn

    sys.argv = ["uvicorn", "harbor_env.server.app:app", *sys.argv[1:]]
    uvicorn.main()


if __name__ == "__main__":
    main()
