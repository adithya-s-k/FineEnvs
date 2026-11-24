# /// script
# requires-python = ">=3.11"
# dependencies = ["tinker>=0.31.0", "tinker-cookbook>=0.5.7", "huggingface_hub>=1.0", "torch", "safetensors"]
# ///
"""HF Job: a Tinker LoRA checkpoint -> PEFT adapter (vLLM/SGLang-loadable) -> a private Hub model repo.

Runs on a CPU flavor with a large disk: the conversion reads the base model's config and tensor layout, so it
downloads the base weights there rather than on a laptop. Needs the TINKER_API_KEY and HF_TOKEN secrets.

    uv run export_adapter.py --tinker-path tinker://<run>/sampler_weights/final \
        --base-model Qwen/Qwen3.6-35B-A3B --repo <namespace>/designgym-qwen3.6-35b-a3b-rl-tools
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tinker_cookbook import weights


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tinker-path", required=True, help="tinker://.../sampler_weights/<name>")
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--repo", required=True, help="Hub model repo for the PEFT adapter (created private)")
    ap.add_argument("--notes", default="", help="JSON object of training details for the model card")
    ap.add_argument("--merged", action="store_true",
                    help="merge the LoRA into full bf16 weights (serves like the base model; no LoRA support needed)")
    a = ap.parse_args()

    raw = weights.download(tinker_path=a.tinker_path, output_dir="/tmp/tinker-adapter")
    out = Path("/tmp/merged" if a.merged else "/tmp/peft-adapter")
    if a.merged:
        weights.build_hf_model(base_model=a.base_model, adapter_path=raw, output_path=str(out))
    else:
        weights.build_lora_adapter(base_model=a.base_model, adapter_path=raw, output_path=str(out))
    notes = json.loads(a.notes) if a.notes else {}
    (out / "README.md").write_text(
        f"---\nbase_model: {a.base_model}\n" + ("" if a.merged else "library_name: peft\n")
        + "tags: [designgym, openenv, tinker, lora, rl]\n---\n\n"
        f"# {a.repo.split('/')[-1]}\n\n{'Merged weights' if a.merged else 'LoRA adapter'} for `{a.base_model}`, trained with Tinker on DesignGym "
        f"(OpenEnv `designgym`).\n\n- Source checkpoint: `{a.tinker_path}`\n"
        + "".join(f"- {k}: {v}\n" for k, v in notes.items())
        + ("\nServe with vLLM like the base model: `vllm serve <this repo>`. " if a.merged else
           "\nServe with vLLM: `vllm serve {base} --enable-lora --max-lora-rank 32 --lora-modules rl=<this repo>`. ")
        .replace("{base}", a.base_model) + "Thinking off, tool parser `qwen3_coder`.\n")
    url = weights.publish_to_hf_hub(model_path=str(out), repo_id=a.repo, private=True)
    print(f"published {url}", flush=True)


if __name__ == "__main__":
    main()
