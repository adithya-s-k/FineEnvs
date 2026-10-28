"""Miles GRPO on DesignGym GUI episodes (inside the job container, cwd /root/miles; see job.sh).

Follows `scripts/run_qwen3_5_35b_a3b_lora.py`: Megatron-Bridge (`--megatron-to-hf-mode bridge`, so no torch_dist
conversion and the VLM's HF classes), LoRA, colocated SGLang + Megatron on one node. The rollout is our custom
generate function; the reward comes from DesignGym, so no reward model is configured.

    python train_designgym.py --model Qwen/Qwen3.6-35B-A3B --data /work/data/train.jsonl --save /work/ckpt ...
"""

from __future__ import annotations

import argparse
import os

from miles.utils.external_utils.command_utils.base_backend import default_config

# Per-model defaults: Miles model-arg registry entry (bridge mode reads the HF config; the registry only satisfies
# Megatron's argparse) and the single-node parallel layout.
MODELS = {
    "Qwen/Qwen3.6-35B-A3B": {
        "registry": "qwen3.6-35B-A3B_lora",
        # TP<=2 (num_query_groups=2), EP over the node; GatedDeltaNet rejects packed thd -> bshd, micro-batch 1
        "parallel": "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
                    "--context-parallel-size 1 --expert-model-parallel-size {gpus} --expert-tensor-parallel-size 1",
        "lora_extra": "--experts-shared-outer-loras",
        "engine_gpus": 8,
    },
    "Qwen/Qwen3.8-27B": {
        "registry": "qwen3.8-27B",
        # TP4 hung in the first backward (NCCL all-reduce timeout in the TP group); TP2 + SP is the layout the
        # 35B-A3B recipe validates
        "parallel": "--tensor-model-parallel-size 2 --sequence-parallel --pipeline-model-parallel-size 1 "
                    "--context-parallel-size 1",
        "lora_extra": "",
        "engine_gpus": 2,
    },
    # dev loop on one GPU: the dense Qwen3.5 VLM class the 27B uses
    "Qwen/Qwen3.5-4B": {
        "registry": "qwen3.5-4B",
        "parallel": "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1",
        "lora_extra": "",
        "engine_gpus": 1,
    },
}


def build_args(a) -> str:
    spec = MODELS[a.model]
    gpus = a.gpus
    lora = (f"--lora-rank {a.lora_rank} --lora-alpha {a.lora_alpha} --lora-dropout 0 --target-modules all-linear "
            f"{spec['lora_extra']} --no-gradient-accumulation-fusion --lora-base-cpu-backup "
            f"--sglang-max-lora-rank {a.lora_rank} --sglang-lora-backend triton") if a.lora_rank else ""
    parts = [
        f"--hf-checkpoint {a.hf_dir} --megatron-to-hf-mode bridge",
        lora,
        # rollout: our generate function plays the episode and sets the reward
        "--custom-generate-function-path rl.miles.designgym_generate.generate",
        f"--designgym-url {a.env_url} --designgym-max-turns {a.max_turns} --designgym-max-images {a.max_images} "
        f"--designgym-keep-images {a.keep_images} --designgym-samples-per-episode {a.samples_per_episode}",
        f"--prompt-data {a.data} --input-key prompt --metadata-key metadata --rollout-shuffle",
        f"--num-rollout {a.num_rollout} --rollout-batch-size {a.rollout_batch_size} "
        f"--n-samples-per-prompt {a.group_size} --rollout-max-response-len {a.max_new_tokens} "
        f"--rollout-max-context-len {a.max_context} --rollout-temperature 1.0 "
        f"--global-batch-size {a.global_batch_size}",
        "--advantage-estimator grpo --entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28",
        f"--optimizer adam --lr {a.lr} --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 "
        "--adam-beta2 0.98",
        spec["parallel"].format(gpus=gpus),
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        f"--qkv-format bshd --micro-batch-size 1 --max-tokens-per-gpu {a.max_tokens_per_gpu}",
        f"--rollout-num-gpus-per-engine {a.engine_gpus or min(spec['engine_gpus'], gpus)} "
        f"--sglang-mem-fraction-static {a.sglang_mem} --sglang-dtype bfloat16 --sglang-decode-log-interval 1000",
        f"--save-interval {a.save_interval} --save {a.save}",
        "--attention-dropout 0.0 --hidden-dropout 0.0 --update-weight-buffer-size 536870912",
        f"--actor-num-nodes 1 --actor-num-gpus-per-node {gpus} --colocate",
        a.extra,
    ]
    return " ".join(p for p in parts if p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B", choices=sorted(MODELS))
    ap.add_argument("--hf-dir", required=True, help="local snapshot of the model")
    ap.add_argument("--data", required=True)
    ap.add_argument("--save", required=True)
    ap.add_argument("--env-url", default="http://127.0.0.1:8027")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--engine-gpus", type=int, default=0)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--num-rollout", type=int, default=2)
    ap.add_argument("--rollout-batch-size", type=int, default=4, help="tasks per rollout")
    ap.add_argument("--group-size", type=int, default=4)
    ap.add_argument("--samples-per-episode", type=int, default=0,
                    help="chunks per episode (fixed, padded); 0: the window's bound for --max-turns")
    ap.add_argument("--steps-per-rollout", type=int, default=2,
                    help="optimizer steps per rollout; sets --global-batch-size")
    ap.add_argument("--max-turns", type=int, default=20)
    ap.add_argument("--max-images", type=int, default=8)
    ap.add_argument("--keep-images", type=int, default=3)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--max-context", type=int, default=32768)
    ap.add_argument("--max-tokens-per-gpu", type=int, default=32768)
    ap.add_argument("--sglang-mem", type=float, default=0.4)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--save-interval", type=int, default=1)
    ap.add_argument("--extra", default="", help="more Miles flags, verbatim")
    a = ap.parse_args()
    from rl.common.episode import EpisodeConfig

    a.samples_per_episode = a.samples_per_episode or EpisodeConfig(
        max_turns=a.max_turns, max_images=a.max_images, keep_images=a.keep_images).chunk_bound()
    total = a.rollout_batch_size * a.group_size * a.samples_per_episode
    if total % a.steps_per_rollout:
        raise SystemExit(f"{total} samples per rollout do not split into {a.steps_per_rollout} steps")
    a.global_batch_size = total // a.steps_per_rollout
    print(f"[designgym] {total} samples per rollout ({a.samples_per_episode} per episode) -> global batch "
          f"{a.global_batch_size}, {a.steps_per_rollout} optimizer steps per rollout", flush=True)
    train_args = build_args(a)
    print("[designgym] miles args:", train_args, flush=True)
    backend = default_config().create_backend()
    backend.execute_train(
        train_args=train_args, num_gpus_per_node=a.gpus, megatron_model_type=MODELS[a.model]["registry"],
        megatron_path="/root/Megatron-LM",
        # Ray workers inherit HF_TOKEN from the job's environment; never put secrets here: Miles prints the
        # runtime env (`ray job submit --runtime-env-json=...`) into the log.
        extra_env_vars={"PYTHONPATH": os.environ.get("DESIGNGYM_PYTHONPATH", "/work")},
    )


if __name__ == "__main__":
    main()
