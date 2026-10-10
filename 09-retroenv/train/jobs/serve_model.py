"""Start vLLM in the board's non-thinking configuration, for training or for evaluation."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.sync_grpo import MODEL_REVISIONS, tokenizer_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.8-27B", help="Base model; also the served name")
    parser.add_argument("--checkpoint", help="Serve these (merged) weights under the base model's name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--eval", action="store_true", help="Serve for evaluation: no weight updates")
    parser.add_argument("--data-parallel-size", type=int, default=1, help="Replicas, one per visible GPU")
    parser.add_argument("--trainer", choices=["sync", "async"], default="sync", help="Which trainer syncs weights")
    parser.add_argument(
        "--thinking", action="store_true", help="Evaluation only: the model's own template, thinking on"
    )
    args = parser.parse_args()
    if args.thinking and not args.eval:
        parser.error("--thinking is for evaluation; training runs in non-thinking mode")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    template = output / "chat_template.jinja"
    if args.thinking:
        from transformers import AutoTokenizer

        # The stock template thinks unless told not to; the reasoning parser returns the
        # thinking as its own field, which the evaluator records but never sends back. Sampling
        # follows the model's generation_config.json (top_k 20, top_p 0.95): long reasoning
        # sampled without truncation drifts. A turn may think at length, so sequences get room.
        tokenizer = AutoTokenizer.from_pretrained(args.model, revision=MODEL_REVISIONS[args.model])
        template.write_text(tokenizer.chat_template)
        chat_kwargs, generation, max_len = '{"enable_thinking":true}', "auto", "131072"
    else:
        template.write_text(tokenizer_for(args.model).chat_template)
        chat_kwargs, generation, max_len = '{"enable_thinking":false,"preserve_thinking":true}', "vllm", "65536"
    # The vllm executable of this environment; the job never activates it, so it is not on PATH.
    command = [
        str(Path(sys.executable).parent / "vllm"), "serve", args.checkpoint or args.model,
        "--host", "127.0.0.1",
        "--port", str(args.port),
        "--served-model-name", args.model,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", str(args.data_parallel_size),
        "--dtype", "bfloat16",
        "--max-model-len", max_len,
        "--gpu-memory-utilization", "0.85",
        # Evaluation parses tool calls server-side, as the board's providers do.
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_xml",
        "--chat-template", str(template),
        "--default-chat-template-kwargs", chat_kwargs,
        "--generation-config", generation,
        # Qwen3.5-architecture checkpoints are multimodal; this agent sends text only.
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--gdn-prefill-backend", "triton",
    ]  # fmt: skip
    if args.thinking:
        command += ["--reasoning-parser", "qwen3"]
    if args.checkpoint:
        command += ["--safetensors-load-strategy", "eager"]  # read sequentially, not memory-mapped
    else:
        command += ["--revision", MODEL_REVISIONS[args.model]]
    if not args.eval:
        # Training: the trainer pushes weights over NCCL after each step, and the token ids and
        # logprobs come back with every completion. CUDA graphs stay on: in eager mode a 27B
        # turn took about 11 s. Prefix caching stays on too: every turn resends the episode so
        # far, and without the cache AsyncGRPO spent most of its time re-reading 5-12k-token
        # contexts (about 250 generated tokens/s). GRPOTrainer resets the cache after each weight
        # update; AsyncGRPO does not, which is sound because it trains against the log-probs
        # vLLM reports, whatever weights computed the cached prefix.
        command += [
            "--weight-transfer-config", '{"backend":"nccl"}',
            "--logprobs-mode", "processed_logprobs",
            "--return-tokens-as-token-ids",
        ]  # fmt: skip
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
