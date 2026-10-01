"""Check the required public APIs before requesting GPUs or creating sandboxes."""

import argparse
import inspect
import json
from importlib.metadata import version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["all", "whitebox", "opencode", "multi_harness"],
        default="all",
    )
    args = parser.parse_args()
    from harbor_env.harness import HarborSession
    from opencode_env.harness import OpenCodeSessionFactory
    from openenv.core.harness import TrainingTrace
    from transformers.models.lfm2 import modeling_lfm2
    from trl import GRPOTrainer
    from trl.experimental.async_grpo import AsyncGRPOTrainer, openenv_harness

    if args.mode != "whitebox" and "fetch_training_trace" not in inspect.getsource(
        openenv_harness
    ):
        raise RuntimeError(
            "TRL main does not yet consume TrainingTrace. Wait for huggingface/trl#6947 to merge."
        )
    if "seq_idx" not in inspect.getsource(modeling_lfm2.Lfm2ShortConv):
        raise RuntimeError(
            "This LFM implementation lacks sequence boundaries for packed training; install Transformers main."
        )
    if "environment_factory" not in inspect.signature(GRPOTrainer).parameters:
        raise RuntimeError("GRPOTrainer.environment_factory is required")
    assert all(
        callable(api)
        for api in (
            TrainingTrace,
            OpenCodeSessionFactory,
            HarborSession,
            AsyncGRPOTrainer,
        )
    )
    print(
        json.dumps(
            {
                name: version(name)
                for name in ("trl", "openenv", "transformers", "vllm")
            },
            indent=2,
        )
    )
    print(
        "Required APIs are present. GPU training and sandbox connectivity still need a smoke test."
    )


if __name__ == "__main__":
    main()
