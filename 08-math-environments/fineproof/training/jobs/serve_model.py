"""Start vLLM in non-thinking mode for training or evaluation. Adapted from 05-multi-harness-rl/jobs/serve_model.py."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train import MODEL_REVISIONS, tokenizer_for  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B", help="Base model identity")
    parser.add_argument("--checkpoint", help="Load these weights but keep the base served-model name")
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    template = output / "chat_template.jinja"
    template.write_text(tokenizer_for(args.model).chat_template)
    command = [
        "vllm", "serve", args.checkpoint or args.model,
        "--host", "127.0.0.1", "--port", str(args.port),
        "--served-model-name", args.model,
        "--tensor-parallel-size", "1",
        "--data-parallel-size", "2" if args.eval else "1",
        "--dtype", "bfloat16",
        "--max-model-len", "65536",
        "--gpu-memory-utilization", "0.85",
        "--chat-template", str(template),
        "--default-chat-template-kwargs", '{"enable_thinking":false,"preserve_thinking":true}',
        "--generation-config", "vllm",
        "--logprobs-mode", "processed_logprobs",
        "--return-tokens-as-token-ids",
        "--enforce-eager",
        "--no-enable-prefix-caching",
        "--limit-mm-per-prompt", '{"image":0,"video":0}',
        "--gdn-prefill-backend", "triton",
    ]
    if not args.checkpoint:
        command += ["--revision", MODEL_REVISIONS[args.model]]
    else:
        command += ["--safetensors-load-strategy", "eager"]  # read bucket checkpoints sequentially
    if not args.eval:
        command += ["--weight-transfer-config", '{"backend":"nccl"}']
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
