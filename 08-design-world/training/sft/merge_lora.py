# /// script
# requires-python = ">=3.11"
# dependencies = ["torch", "numpy", "safetensors>=0.5", "huggingface_hub[hf_xet]>=1.0"]
# ///
"""HF Job: merge a TRL/PEFT LoRA checkpoint into the base weights, shard by shard, and publish the merged repo.

The base checkpoint's files are copied as they are (config, processor, chat template, MTP layers, fused experts);
only the LoRA-targeted `nn.Linear` weights change, W += (B @ A) * alpha / r in fp32, saved back in the base dtype.
vLLM then serves the result like the base model (it cannot load a LoRA on this MoE).

    uv run merge_lora.py --base Qwen/Qwen3.6-35B-A3B --adapter <namespace>/designgym-sft-...-smoke \
        --subfolder checkpoint-24 --repo <namespace>/designgym-sft-...-smoke-ck24-merged
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--adapter", required=True, help="model repo with the PEFT adapter")
    ap.add_argument("--subfolder", default="", help="e.g. checkpoint-120 (empty: repo root)")
    ap.add_argument("--repo", required=True, help="output model repo (public)")
    a = ap.parse_args()

    pattern = f"{a.subfolder}/adapter_*" if a.subfolder else "adapter_*"
    ad = Path(snapshot_download(a.adapter, allow_patterns=[pattern])) / a.subfolder
    cfg = json.loads((ad / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    lora = {}
    with safe_open(ad / "adapter_model.safetensors", "pt") as f:
        for k in f.keys():
            base_key, _, ab = k.removeprefix("base_model.model.").rpartition(".lora_")
            lora.setdefault(base_key + ".weight", {})[ab.split(".")[0]] = f.get_tensor(k).float()
    print(f"{len(lora)} LoRA-adapted weights, scale {scale}", flush=True)

    base = Path(snapshot_download(a.base))
    out = Path("/tmp/merged")
    out.mkdir(parents=True, exist_ok=True)
    done = set()
    for src in sorted(base.iterdir()):
        if src.suffix != ".safetensors":
            if src.is_file():
                shutil.copy(src, out / src.name)
            continue
        with safe_open(src, "pt") as f:
            meta = f.metadata()
            tensors = {}
            for k in f.keys():
                t = f.get_tensor(k)
                if k in lora:
                    w = lora[k]
                    t = (t.float() + (w["B"] @ w["A"]) * scale).to(t.dtype)
                    done.add(k)
                tensors[k] = t
        save_file(tensors, out / src.name, metadata=meta)
        print(f"{src.name}: {sum(1 for k in tensors if k in lora)} merged", flush=True)
    missing = set(lora) - done
    if missing:
        raise SystemExit(f"{len(missing)} adapter weights have no base tensor, e.g. {sorted(missing)[:3]}")
    (out / "README.md").write_text(
        f"---\nbase_model: {a.base}\ntags: [designgym, openenv, trl, sft, lora]\n---\n\n# {a.repo.split('/')[-1]}\n\n"
        f"`{a.base}` with the LoRA `{a.adapter}`{'/' + a.subfolder if a.subfolder else ''} merged in (TRL SFT on "
        f"DesignGym trajectories, `FineEnvs/designworld-sft`). Serve like the base model; thinking off, tool parser "
        f"`qwen3_coder`.\n")
    api = HfApi()
    api.create_repo(a.repo, private=False, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=a.repo, commit_message=f"merge {a.adapter} {a.subfolder}")
    print(f"published https://huggingface.co/{a.repo} ({len(done)} weights merged)", flush=True)


if __name__ == "__main__":
    main()
