"""Check the trainer environment before allocating GPUs: APIs, the chat template, and the tools.

Runs on CPU in a minute. A model whose chat template TRL cannot train through, or a tool whose
docstring does not become a schema, fails here instead of after vLLM has loaded 55 GB of weights.
"""

import argparse
import inspect
import json
import sys
from importlib.metadata import version
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B")
    args = parser.parse_args()

    from transformers.utils import get_json_schema
    from trl import GRPOTrainer
    from trl.chat_template_utils import (
        get_training_chat_template,
        is_chat_template_prefix_preserving,
        parse_response,
    )
    from trl.experimental.async_grpo import AsyncGRPOTrainer

    if "environment_factory" not in inspect.signature(GRPOTrainer).parameters:
        raise RuntimeError("GRPOTrainer.environment_factory is required; install TRL main")
    if "environment_factory" not in inspect.signature(AsyncGRPOTrainer).parameters:
        raise RuntimeError("AsyncGRPOTrainer.environment_factory is required; install TRL main")

    # Gated DeltaNet layers need FLA's kernels; without them training runs a slow PyTorch loop.
    import importlib.util

    if importlib.util.find_spec("fla") is None:
        raise RuntimeError("flash-linear-attention is missing; Qwen3.5-architecture training would be ~10x slower")

    # The trainer talks to the server through the light client: no RDKit, no task store.
    from retroenv_openenv.client import RemoteRetroRouteEnv

    if any(name.startswith(("rdkit", "retroenv.")) for name in sys.modules):
        raise RuntimeError("Importing the RetroEnv client loaded the server stack")
    tools = [
        getattr(RemoteRetroRouteEnv, name)
        for name, _ in inspect.getmembers(RemoteRetroRouteEnv, predicate=inspect.isfunction)
        if not name.startswith("_") and name not in ("reset", "get_reward")
    ]
    schemas = [get_json_schema(tool) for tool in tools]  # what TRL shows the model

    # Multi-turn training needs a prefix-preserving template, or one TRL knows how to patch.
    from train.sync_grpo import tokenizer_for

    tokenizer = tokenizer_for(args.model)
    preserving = is_chat_template_prefix_preserving(tokenizer)
    if not preserving and get_training_chat_template(tokenizer) is None:
        raise RuntimeError(f"{args.model}: chat template is not prefix-preserving and TRL has no training template")

    # A tool call rendered by the template must parse back into the same call.
    call = {"name": "stock_retrieve", "arguments": {"query": "CCO", "mode": "exact", "limit": 5}}
    messages = [
        {"role": "user", "content": "Is ethanol in stock?"},
        {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": call}]},
    ]
    prompt = tokenizer.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True, return_dict=False)
    full = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    parsed = parse_response(tokenizer, full[len(prompt) :], prefix=prompt)
    got = (parsed.get("tool_calls") or [{}])[0].get("function", {})
    if got.get("name") != call["name"] or json.loads(json.dumps(got.get("arguments"))) != call["arguments"]:
        raise RuntimeError(f"{args.model}: tool call did not round-trip through the template: {parsed}")

    print(
        json.dumps(
            {
                "versions": {name: version(name) for name in ("trl", "transformers", "vllm", "peft", "openenv")},
                "model": args.model,
                "prefix_preserving": preserving,
                "tools": [schema["function"]["name"] for schema in schemas],
            },
            indent=2,
        )
    )
    print("Setup checks passed. GPU training and the server still need the smoke test.")


if __name__ == "__main__":
    main()
