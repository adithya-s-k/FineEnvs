"""Check the APIs this recipe needs before requesting GPUs or creating sandboxes."""

import inspect
import json
from importlib.metadata import version


def main():
    from openenv.core.harness import TrainingTrace
    from openenv.harbor import seams
    from trl.experimental.async_grpo import AsyncGRPOTrainer, openenv_harness
    from harbor_env.harness import HarborSession
    from qed_proof_agent.agent import QedProofAgent

    if "fetch_training_trace" not in inspect.getsource(openenv_harness):
        raise RuntimeError("The installed TRL lacks the TrainingTrace consumer; run bash jobs/install.sh")
    fields = set(inspect.signature(seams.Seam).parameters)
    if not {"import_path", "kwargs", "model_fmt"} <= fields:
        raise RuntimeError(f"OpenEnv's Seam changed shape: {sorted(fields)}")
    assert callable(TrainingTrace) and callable(AsyncGRPOTrainer) and callable(HarborSession)
    assert QedProofAgent.name() == "qed-proof"
    print(json.dumps({name: version(name) for name in ("trl", "openenv", "transformers", "vllm", "harbor")}, indent=2))
    print("Required APIs are present. GPU training and sandbox connectivity still need a smoke test.")


if __name__ == "__main__":
    main()
