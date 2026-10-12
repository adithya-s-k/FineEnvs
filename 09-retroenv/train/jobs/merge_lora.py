"""Merge a trained LoRA adapter into its base model, so vLLM serves it like any checkpoint.

Serving the merged weights avoids depending on vLLM's LoRA support for the Qwen3.5
architecture's linear-attention layers. The model class is the one TRL trained (the
checkpoint's own multimodal class), so the adapter's module names line up.
"""

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.sync_grpo import MODEL_REVISIONS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model the adapter was trained on")
    parser.add_argument("--adapter", required=True, help="A checkpoint-N or final directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    import torch
    from huggingface_hub import snapshot_download
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    revision = MODEL_REVISIONS[args.model]
    model = AutoModelForImageTextToText.from_pretrained(args.model, revision=revision, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
    output = Path(args.output)
    model.save_pretrained(output, max_shard_size="5GB")
    # Tokenizer, chat template and preprocessor files come from the base snapshot unchanged.
    base = Path(
        snapshot_download(
            args.model,
            revision=revision,
            allow_patterns=["*.json", "*.jinja", "*.txt"],
            ignore_patterns=["*.index.json"],
        )
    )
    for source in base.iterdir():
        if source.name not in {"config.json", "generation_config.json"}:
            shutil.copyfile(source, output / source.name)
    print(output)


if __name__ == "__main__":
    main()
