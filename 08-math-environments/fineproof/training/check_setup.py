"""Check the APIs this recipe needs before requesting GPUs or creating sandboxes."""

import inspect
import json
import sys
from pathlib import Path
from importlib.metadata import version


def main():
    sys.path.insert(0, str(Path(__file__).parent))
    from openenv.core.harness import TrainingTrace
    from openenv.harbor import seams
    from trl.experimental.async_grpo import AsyncGRPOTrainer, openenv_harness
    from harbor_env.harness import HarborSession
    from fineproof_agent.agent import FineProofAgent

    if "fetch_training_trace" not in inspect.getsource(openenv_harness):
        raise RuntimeError("The installed TRL lacks the TrainingTrace consumer; run bash jobs/install.sh")
    from tasks import HARNESS

    try:
        seam = seams.get(HARNESS)
    except KeyError as exc:
        raise RuntimeError("This OpenEnv cannot run `--harness module:Class`; run bash jobs/install.sh") from exc
    if not seam.host_side or "api_base" not in seam.resolve(base_url="http://proxy", session="s", model="m")[1]:
        raise RuntimeError(f"OpenEnv does not wire {HARNESS} as a host-side agent")
    assert callable(TrainingTrace) and callable(AsyncGRPOTrainer) and callable(HarborSession)
    assert FineProofAgent.name() == "fineproof"
    print(json.dumps({name: version(name) for name in ("trl", "openenv", "transformers", "vllm", "harbor")}, indent=2))
    print("Required APIs are present. GPU training and sandbox connectivity still need a smoke test.")


if __name__ == "__main__":
    main()
