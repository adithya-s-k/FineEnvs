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
    parser.add_argument("--max-model-len", type=int, help="Training: tokens per episode (default 65536)")
    parser.add_argument(
        "--thinking", action="store_true", help="Evaluation only: the model's own template, thinking on"
    )
    parser.add_argument(
        "--harness", choices=("board", "training"), default="board", help="With --thinking: whose thinking setup"
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
        if args.harness == "training":
            # As thinking RL samples (train/async_grpo.py): the training template (its reasoning
            # trim removed), earlier turns' thinking kept (the evaluator sends it back), untruncated
            # sampling at the request's temperature, and the training context.
            template.write_text(tokenizer_for(args.model, thinking=True).chat_template)
            chat_kwargs, generation, max_len = '{"enable_thinking":true,"preserve_thinking":true}', "vllm", "65536"
    else:
        template.write_text(tokenizer_for(args.model).chat_template)
        chat_kwargs, generation, max_len = '{"enable_thinking":false,"preserve_thinking":true}', "vllm", "65536"
    if args.max_model_len:
        max_len = str(args.max_model_len)
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
    # Every turn resends the episode so far, so prefix caching is asked for explicitly: vLLM 0.25
    # leaves it off for hybrid (Gated DeltaNet) models like Qwen3.8, and an x8 smoke spent ~33 s a
    # turn re-reading 5-20k-token contexts; an evaluation without it re-read ~9.5k prompt tokens a
    # second to generate ~370. With it, vLLM caches in its experimental "align" mode (the
    # linear-attention state is kept at block boundaries).
    command += ["--enable-prefix-caching"]
    if not args.eval:
        # Training: the trainer pushes weights over NCCL after each step (and resets the cache),
        # and the token ids and logprobs come back with every completion. CUDA graphs stay on: in
        # eager mode a 27B turn took about 11 s.
        command += [
            "--weight-transfer-config", '{"backend":"nccl"}',
            "--logprobs-mode", "processed_logprobs",
            "--return-tokens-as-token-ids",
        ]  # fmt: skip
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
