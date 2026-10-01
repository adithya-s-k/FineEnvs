"""Start the same non-thinking vLLM configuration for training or evaluation."""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from train.whitebox import MODEL_REVISIONS, tokenizer_for


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default="LiquidAI/LFM2.5-2.6B", help="Base model identity"
    )
    parser.add_argument(
        "--checkpoint", help="Load these weights but retain the base served-model name"
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    tokenizer = tokenizer_for(args.model)
    template = output / "chat_template.jinja"
    template.write_text(tokenizer.chat_template)
    command = [
        "vllm",
        "serve",
        args.checkpoint or args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--served-model-name",
        args.model,
        "--tensor-parallel-size",
        "1",
        "--data-parallel-size",
        "2" if args.eval else "1",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "131072",
        "--gpu-memory-utilization",
        "0.85",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "lfm2" if "lfm" in args.model.lower() else "qwen3_xml",
        "--chat-template",
        str(template),
        "--default-chat-template-kwargs",
        '{"enable_thinking":false,"preserve_thinking":true}',
        "--generation-config",
        "vllm",
        "--logprobs-mode",
        "processed_logprobs",
        "--return-tokens-as-token-ids",
        "--enforce-eager",
        "--no-enable-prefix-caching",
    ]
    if not args.checkpoint:
        command += ["--revision", MODEL_REVISIONS[args.model]]
    else:
        # Read bucket/network checkpoints sequentially instead of memory-mapping them.
        command += ["--safetensors-load-strategy", "eager"]
    if not args.eval:
        command += ["--weight-transfer-config", '{"backend":"nccl"}']
        os.environ["VLLM_SERVER_DEV_MODE"] = "1"
    if "qwen" in args.model.lower():
        command += [
            "--limit-mm-per-prompt",
            '{"image":0,"video":0}',
            "--gdn-prefill-backend",
            "triton",
        ]
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
