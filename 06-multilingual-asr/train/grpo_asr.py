"""Single-GPU GRPO against the multilingual ASR OpenEnv server."""

import argparse
import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from multilingual_asr.client import connect
from multilingual_asr.data.evalset import load as load_evalset
from multilingual_asr.data.schema import FAMILIES
from multilingual_asr.models import AsrAction
from multilingual_asr.runtime import local_server
from multilingual_asr.training import (
    AssetCache,
    TrainingEnvironment,
    balanced_rows,
    env_reward,
    task_rows,
)

INTEGER_OPTIONS = (
    "train_per_group",
    "eval_per_group",
    "eval_limit",
    "max_steps",
    "num_generations",
    "max_completion_length",
    "seed",
)


@dataclass
class Config:
    snapshot: str = ""
    env_url: str = ""
    evalset: str = ""
    eval_limit: int = 0
    model: str = "google/gemma-4-E2B-it"
    model_revision: str = ""
    languages: tuple[str, ...] = ("en_us", "hi_in")
    families: tuple[str, ...] = ("transcription",)
    lora_target_modules: tuple[str, ...] = ()
    train_per_group: int = 8
    eval_per_group: int = 2
    max_steps: int = 20
    num_generations: int = 4
    max_completion_length: int = 256
    learning_rate: float = 1e-5
    seed: int = 42
    output_dir: str = "artifacts/local-run"
    run_name: str = "asr-grpo"
    smoke: bool = False

    def validate(self):
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("Use a finite positive learning rate")
        if bool(self.snapshot) == bool(self.env_url):
            raise ValueError("Provide exactly one of --snapshot or --env-url")
        if self.num_generations < 2:
            raise ValueError("GRPO needs at least two generations per group")
        if min(self.max_steps, self.train_per_group, self.eval_per_group) < 1:
            raise ValueError("Use positive limits")
        if set(self.families) - set(FAMILIES):
            raise ValueError(f"Unknown task family in {self.families}")


def lora_target_modules(model_id, revision, wanted=("q_proj", "v_proj")):
    """Resolve LoRA targets to full, unambiguous module names.

    PEFT adapts genuine leaves and matches a short target against every module whose name
    ends with it. Gemma 4 wraps some projections in Gemma4ClippableLinear, so a suffix can
    match both a wrapper and a leaf and injection fails outright. Full names name exactly
    one leaf each. Unlike the OCR environment, nothing is skipped here: the audio tower is
    the part this task must adapt.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    try:
        settings = AutoConfig.from_pretrained(model_id, revision=revision or None)
        with torch.device("meta"):
            model = AutoModel.from_config(settings)
    except Exception as error:  # Unknown architecture: keep the plain names.
        print(
            f"LoRA introspection unavailable ({error}); using {list(wanted)}",
            flush=True,
        )
        return list(wanted)
    adaptable = (torch.nn.Linear, torch.nn.Embedding, torch.nn.Conv1d, torch.nn.Conv2d)
    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, adaptable)
        and any(part in wanted for part in name.split("."))
    ]
    return sorted(targets) or list(wanted)


def generate(model, processor, observation, cache, max_tokens):
    import torch

    audio = cache.audio(observation)
    inputs = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio},
                    {"type": "text", "text": observation.prompt},
                ],
            }
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
    return processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]


def evaluate(trainer, rows, url, cache, max_tokens):
    model, processor = trainer.model, trainer.processing_class
    was_training = model.training
    checkpointing = model.is_gradient_checkpointing
    was_cache = model.config.use_cache
    samples, groups = [], defaultdict(list)
    try:
        model.eval()
        if checkpointing:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        with connect(url) as client:
            for row in rows:
                observation = client.reset(task_id=row["task_id"]).observation
                prediction = generate(model, processor, observation, cache, max_tokens)
                result = client.step(AsrAction(transcript=prediction))
                sample = {
                    **row,
                    "prediction": prediction,
                    "reward": float(result.reward),
                    **result.observation.metrics,
                }
                samples.append(sample)
                groups[f"{row['language']}/{row['family']}"].append(sample)
    finally:
        model.config.use_cache = was_cache
        if checkpointing:
            model.gradient_checkpointing_enable()
        model.train(was_training)
    metrics = {}
    for key, values in groups.items():
        metrics[key] = {
            "samples": len(values),
            "reward": sum(v["reward"] for v in values) / len(values),
        }
        for metric in ("wer", "cer", "exact_match"):
            observed = [v[metric] for v in values if metric in v]
            if observed:
                metrics[key][metric] = sum(observed) / len(observed)
    return {
        "macro_reward": sum(v["reward"] for v in metrics.values()) / len(metrics),
        "by_language_task": metrics,
        "samples": samples,
    }


def run(config):
    import torch
    from datasets import Dataset
    from huggingface_hub import model_info
    from peft import LoraConfig
    from transformers import AutoProcessor
    from trl import GRPOConfig, GRPOTrainer

    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GRPO training requires CUDA; asr-smoke checks the env on CPU"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This recipe supports one GPU. Do not launch with torchrun")
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    with ExitStack() as stack:
        url = config.env_url or stack.enter_context(
            local_server(config.snapshot, config.num_generations + 4)
        )
        with connect(url) as client:
            manifest = client.manifest()
        languages, families = list(config.languages), list(config.families)

        # Only immutable identifiers reach the sampler. A "prompt" column would be read
        # by TRL as a conversation, and the environment already owns the prompt.
        def selection(split, per_group):
            rows = balanced_rows(
                task_rows(url, split, languages, families),
                languages,
                families,
                config.seed,
                per_group,
            )
            return [
                {key: row[key] for key in ("task_id", "language", "family")}
                for row in rows
            ]

        train_rows = selection("train", config.train_per_group)
        if config.evalset:
            # A frozen set is used whole unless the caller limits it explicitly, so a
            # score stays comparable against its evalset_id.
            frozen = load_evalset(config.evalset)
            served = {row["task_id"] for row in task_rows(url, "test", None, None)}
            chosen = [e for e in frozen["tasks"] if e["task_id"] in served]
            if not chosen:
                raise ValueError(
                    f"This snapshot holds none of {frozen['name']}'s tasks; prepare it "
                    "with asr-prepare --evalset"
                )
            missing = frozen["size"] - len(chosen)
            if missing:
                print(
                    f"warning: snapshot is missing {missing} of {frozen['size']} "
                    f"{frozen['name']} tasks; scores are not comparable to the full set",
                    flush=True,
                )
            if config.eval_limit:
                chosen = chosen[:: max(1, len(chosen) // config.eval_limit)][
                    : config.eval_limit
                ]
            eval_rows = [
                {key: e[key] for key in ("task_id", "language", "family")}
                for e in chosen
            ]
            evalset_id = frozen["evalset_id"]
        else:
            eval_rows = selection("test", config.eval_per_group)
            evalset_id = None
        revision = model_info(
            config.model, revision=config.model_revision or "main"
        ).sha
        targets = list(config.lora_target_modules) or lora_target_modules(
            config.model, revision
        )
        print(f"LoRA targets: {len(targets)} modules, e.g. {targets[:2]}", flush=True)

        metadata = {
            "config": asdict(config),
            "model_revision": revision,
            "lora_target_modules": targets,
            "manifest": manifest,
            "evalset_id": evalset_id,
            "eval_limit": config.eval_limit or None,
            "train_tasks": train_rows,
            "eval_tasks": eval_rows,
        }
        (output / "run-metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
        )

        processor = AutoProcessor.from_pretrained(config.model, revision=revision)
        cache = AssetCache(url)

        # TRL calls the factory once per parallel environment and owns the pool, so
        # this returns a single session rather than a list of them.
        def factory():
            environment = TrainingEnvironment(url, cache, manifest["snapshot_id"])
            stack.callback(environment._close)
            return environment

        trainer = GRPOTrainer(
            model=config.model,
            processing_class=processor,
            train_dataset=Dataset.from_list(train_rows),
            environment_factory=factory,
            reward_funcs=env_reward,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=targets,
            ),
            args=GRPOConfig(
                output_dir=str(output),
                model_init_kwargs={
                    "revision": revision,
                    "dtype": "bfloat16",
                    "attn_implementation": "sdpa",
                },
                num_generations=config.num_generations,
                per_device_train_batch_size=config.num_generations,
                max_completion_length=config.max_completion_length,
                max_steps=config.max_steps,
                learning_rate=config.learning_rate,
                temperature=0.9,
                seed=config.seed,
                bf16=True,
                report_to=[],
                logging_steps=1,
                save_strategy="no",
                run_name=config.run_name,
            ),
        )

        baseline = evaluate(
            trainer, eval_rows, url, cache, config.max_completion_length
        )
        (output / "baseline.json").write_text(
            json.dumps(baseline, ensure_ascii=False, indent=2) + "\n"
        )

        before = {
            name: p.detach().cpu().clone()
            for name, p in trainer.model.named_parameters()
            if p.requires_grad
        }
        result = trainer.train()
        changed = any(
            not torch.equal(before[name], p.detach().cpu())
            for name, p in trainer.model.named_parameters()
            if p.requires_grad
        )
        zero_std = [
            entry["frac_reward_zero_std"]
            for entry in trainer.state.log_history
            if "frac_reward_zero_std" in entry
        ]
        degenerate = bool(zero_std) and all(value == 1 for value in zero_std)
        problems = []
        if trainer.state.global_step != config.max_steps:
            problems.append(
                f"ran {trainer.state.global_step} steps, expected {config.max_steps}"
            )
        if not math.isfinite(result.training_loss):
            problems.append(f"training loss is {result.training_loss}")
        if not changed and not degenerate:
            problems.append("no adapter weight changed")
        if not changed and degenerate:
            problems.append(
                "every reward group had identical rewards, so the advantage was zero "
                "and no gradient could flow. Raise --num-generations, or select tasks "
                "the model does not already solve exactly"
            )
        if problems:
            raise RuntimeError("Training check failed: " + "; ".join(problems))

        trained = evaluate(trainer, eval_rows, url, cache, config.max_completion_length)
        trainer.save_model(str(output / "adapter"))
        summary = {
            "status": "passed",
            "smoke": config.smoke,
            "steps": trainer.state.global_step,
            "training_loss": result.training_loss,
            "adapter_updated": changed,
            "reward_groups_without_variance": degenerate,
            "lora_target_count": len(targets),
            "evalset_id": evalset_id,
            "baseline": baseline,
            "trained": trained,
            "snapshot_id": manifest["snapshot_id"],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--env-url", default="")
    parser.add_argument(
        "--evalset",
        default="",
        help="Frozen evaluation set JSON; used whole unless --eval-limit is given",
    )
    parser.add_argument("--model", default=Config.model)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--languages", nargs="+", default=list(Config.languages))
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(Config.families)
    )
    parser.add_argument("--lora-target-modules", nargs="+", default=[])
    for name in INTEGER_OPTIONS:
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", Config.output_dir)
    )
    parser.add_argument("--run-name", default=Config.run_name)
    parser.add_argument("--smoke", action="store_true")
    args = vars(parser.parse_args())
    # An explicit flag wins; --smoke only fills what the caller left unset.
    smoke_defaults = {
        "max_steps": 2,
        "num_generations": 4,
        "eval_per_group": 1,
        "train_per_group": 4,
    }
    for name in INTEGER_OPTIONS:
        if args[name] is None:
            args[name] = (
                smoke_defaults[name]
                if args["smoke"] and name in smoke_defaults
                else getattr(Config, name)
            )
    run(Config(**args))


if __name__ == "__main__":
    main()
