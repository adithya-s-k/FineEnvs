"""Merge a trained LoRA adapter into its base model, so vLLM serves it like any checkpoint.

Serving the merged weights avoids depending on vLLM's LoRA support for the Qwen3.5
architecture's linear-attention layers. The merge works on the checkpoint files: the adapter's
module names (TRL trained the checkpoint's own multimodal class) are the base checkpoint's keys.
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from train.sync_grpo import MODEL_REVISIONS


def check_saved(output, model, revision, moved):
    """Diff the saved merged weights against the base checkpoint's, by module type.

    The merge is checked in memory above; this checks what vLLM will load. An adapted module type
    whose saved tensors equal the base's lost its update between the merge and the files (a
    layout conversion on save, say). Fails if any adapted type is unchanged on disk.
    """
    import re
    from collections import defaultdict

    from huggingface_hub import snapshot_download
    from safetensors import safe_open

    base = Path(snapshot_download(model, revision=revision, allow_patterns=["*.safetensors"]))

    def tensors(folder):
        found = {}
        for path in sorted(folder.glob("*.safetensors")):
            with safe_open(str(path), framework="pt") as handle:
                found.update({key: (path, handle) for key in handle.keys()})
        return found

    saved_keys, base_keys = tensors(output), tensors(base)
    only_saved, only_base = sorted(set(saved_keys) - set(base_keys)), sorted(set(base_keys) - set(saved_keys))
    print(
        f"saved {len(saved_keys)} tensors, base {len(base_keys)}; only saved {only_saved[:3]}, only base {only_base[:3]}"
    )
    adapted = re.compile(
        r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)\.weight$"
    )
    by_kind = defaultdict(lambda: [0, 0])  # tensors changed, tensors compared
    changed_layers = defaultdict(list)
    for key in sorted(set(saved_keys) & set(base_keys)):
        match = adapted.search(key)
        if not match or ".layers." not in key or "visual" in key:
            continue
        kind = match.group(1)
        # Every layer of the small out_proj / k_proj, a few of the rest (each read is a full tensor).
        if kind not in ("linear_attn.out_proj", "self_attn.k_proj") and by_kind[kind][1] >= 6:
            continue
        with (
            safe_open(str(saved_keys[key][0]), framework="pt") as a,
            safe_open(str(base_keys[key][0]), framework="pt") as b,
        ):
            changed = bool((a.get_tensor(key) != b.get_tensor(key)).any())
        by_kind[kind][0] += changed
        by_kind[kind][1] += 1
        if changed:
            changed_layers[kind].append(int(key.split(".layers.")[1].split(".")[0]))
    for kind, (changed, compared) in sorted(by_kind.items()):
        print(
            f"  {kind}: {changed}/{compared} saved tensors differ from the base; layers {sorted(changed_layers[kind])}",
            flush=True,
        )
    # A kind the adapter never moved (lora_B still all zero, e.g. after updates with no gradient)
    # is the base on disk by design; only a kind it moved must differ.
    unchanged = [kind for kind, (changed, compared) in by_kind.items() if compared and not changed and kind in moved]
    if unchanged:
        raise RuntimeError(f"The saved merge lost the update for {unchanged}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Base model the adapter was trained on")
    parser.add_argument("--adapter", required=True, help="A checkpoint-N or final directory")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    revision = MODEL_REVISIONS[args.model]
    adapter_dir, output = Path(args.adapter), Path(args.output)
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if config.get("use_rslora") or config.get("use_dora") or config.get("fan_in_fan_out"):
        raise RuntimeError("Only plain LoRA (no rsLoRA, DoRA or fan_in_fan_out) is merged here")
    scale = config["lora_alpha"] / config["r"]
    # adapter key base_model.model.<module>.lora_A.weight -> base checkpoint key <module>.weight
    pairs = {}
    with safe_open(str(adapter_dir / "adapter_model.safetensors"), framework="pt") as handle:
        for key in handle.keys():
            if key.endswith(".lora_A.weight"):
                module = key.removeprefix("base_model.model.").removesuffix(".lora_A.weight")
                pairs[module + ".weight"] = (handle.get_tensor(key), handle.get_tensor(key.replace("lora_A", "lora_B")))
    print(f"{len(pairs)} adapted weights, scale {scale}", flush=True)
    kind = re.compile(r"(self_attn\.(q|k|v|o)_proj|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.(gate|up|down)_proj)\.weight$")
    moved = {kind.search(key).group(1) for key, (_, lora_b) in pairs.items() if kind.search(key) and bool(lora_b.any())}
    if not moved:
        print("note: every lora_B is zero, so this adapter is the base model", flush=True)

    # Merge in the checkpoint files themselves, shard by shard, keeping every key and file name:
    # weight + scale * B @ A in float32, back to the weight's dtype (what PEFT's merge computes).
    # transformers' save_pretrained of a merged Qwen3.5 model wrote the base weights for every
    # layer past the eighth, so a merged checkpoint scored like the base model.
    base = Path(snapshot_download(args.model, revision=revision))
    output.mkdir(parents=True, exist_ok=True)
    for shard in sorted(base.glob("*.safetensors")):
        tensors = {}
        with safe_open(str(shard), framework="pt") as handle:
            metadata = handle.metadata()
            for key in handle.keys():
                weight = handle.get_tensor(key)
                if key in pairs:
                    lora_a, lora_b = pairs.pop(key)
                    weight = (weight.float() + scale * (lora_b.float() @ lora_a.float())).to(weight.dtype)
                tensors[key] = weight
        save_file(tensors, str(output / shard.name), metadata=metadata)
    if pairs:
        raise RuntimeError(f"{len(pairs)} adapted weights have no base tensor, e.g. {next(iter(pairs))}")
    # Config, index, tokenizer and chat template files come from the base snapshot unchanged.
    for source in base.iterdir():
        if source.is_file() and source.suffix != ".safetensors":
            shutil.copyfile(source, output / source.name)
    check_saved(output, args.model, revision, moved)
    print(output)


if __name__ == "__main__":
    main()
