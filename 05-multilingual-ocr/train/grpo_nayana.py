"""Single-GPU GRPO against the same local or hosted Nayana OpenEnv server."""

import argparse
import json
import math
import os
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from nayana_ocr.client import connect
from nayana_ocr.corpus_training import BlockTaskStream, CorpusAPI, build_corpus_dataset
from nayana_ocr.data.evalset import load as load_evalset
from nayana_ocr.data.evalset import subsample as subsample_evalset
from nayana_ocr.data.schema import FAMILIES
from nayana_ocr.runtime import local_server
from nayana_ocr.training import (
    AssetCache,
    TrainingEnvironment,
    balanced_rows,
    env_reward,
    step_with_judge_retry,
    task_rows,
)


@dataclass
class Config:
    snapshot: str = ""
    env_url: str = ""
    evalset: str = ""
    eval_limit: int = 0
    lora_target_modules: tuple[str, ...] = ()
    source_root: str = ""
    cache_dir: str = ""
    local_source: bool = False
    model: str = "Qwen/Qwen3-VL-2B-Instruct"
    model_revision: str = ""
    languages: tuple[str, ...] = ("en", "kn", "hi", "ar")
    families: tuple[str, ...] = FAMILIES
    task_input: str = "auto"
    prefetch_blocks: int = 2
    train_per_group: int = 64
    eval_per_group: int = 4
    max_steps: int = 30
    num_generations: int = 4
    max_completion_length: int = 2048
    max_pixels: int = 1_048_576
    learning_rate: float = 1e-5
    seed: int = 42
    output_dir: str = "artifacts/local-run"
    trackio_space: str = ""
    run_name: str = "nayana-ocr-grpo"
    resume: str = ""
    smoke: bool = False

    def validate(self):
        if (
            not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0
            or self.max_pixels < 28 * 28
        ):
            raise ValueError(
                "Use a finite positive learning rate and at least 784 processor pixels"
            )
        if bool(self.snapshot) == bool(self.env_url):
            raise ValueError("Provide exactly one of --snapshot or --env-url")
        if (
            min(
                self.max_steps,
                self.train_per_group,
                self.eval_per_group,
                self.max_completion_length,
            )
            < 1
            or self.num_generations < 2
        ):
            raise ValueError("Use positive limits and at least two generations")
        if not 0 <= self.prefetch_blocks <= 4:
            raise ValueError("Use 0 to 4 prefetched source blocks")
        if self.resume and self.task_input in {"iterable", "corpus"}:
            raise ValueError(
                "Trainer checkpoint replay is supported for map task input only in this milestone"
            )


INTEGER_OPTIONS = (
    "train_per_group",
    "eval_per_group",
    "max_steps",
    "num_generations",
    "max_completion_length",
    "prefetch_blocks",
    "max_pixels",
    "eval_limit",
    "seed",
)


def _iterate_rows(rows):
    yield from rows


def build_dataset(rows, mode):
    from datasets import Dataset, IterableDataset

    if mode == "iterable":
        # Stream small metadata records from a fixed order. TRL owns G-fold repetition.
        return IterableDataset.from_generator(_iterate_rows, gen_kwargs={"rows": rows})
    return Dataset.from_list(rows)  # metadata only; images stay in the environment


def generate(model, processor, observation, cache, max_tokens):
    import torch

    image = cache.image(observation)
    try:
        inputs = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": observation.prompt},
                    ],
                }
            ],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_tokens, do_sample=False
            )
        return processor.batch_decode(
            output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )[0]
    finally:
        image.close()


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
                result = step_with_judge_retry(client, prediction)
                sample = {
                    **row,
                    "prediction": prediction,
                    "reward": float(result.reward),
                    "grading_policy_id": result.observation.grading_policy_id,
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
        for metric in (
            "exact_match",
            "overlong",
            "char_error_rate",
            "mean_f1",
            "f1_at_50",
            "f1_at_75",
            "judge_accepted",
        ):
            observed = [v[metric] for v in values if metric in v]
            if observed:
                metrics[key][metric] = sum(observed) / len(observed)
    return {
        "macro_reward": sum(v["reward"] for v in metrics.values()) / len(metrics),
        "by_language_task": metrics,
        "samples": samples,
    }


def lora_target_modules(model_id, revision, wanted=("q_proj", "v_proj")):
    """Resolve LoRA targets to full, unambiguous module names.

    PEFT adapts genuine leaves, and it matches a short target against every module whose
    name ends with it. Gemma 4 wraps its text projections in Gemma4ClippableLinear while
    leaving the vision and audio towers as plain Linear, so a suffix like "q_proj" matches
    both the wrappers and the leaves and injection fails with "Target module
    Gemma4ClippableLinear(...) is not supported". Returning full module names removes the
    ambiguity: every entry names exactly one adaptable leaf.

    The tree is built on the meta device, which allocates no weights. Introspection is an
    optimisation, never a gate: a model that cannot be built falls back to the plain names
    and behaves exactly as before.
    """
    import torch
    from transformers import AutoConfig, AutoModel

    try:
        settings = AutoConfig.from_pretrained(model_id, revision=revision or None)
        with torch.device("meta"):
            model = AutoModel.from_config(settings)
    except Exception as error:  # Unknown architecture: keep the previous behaviour.
        print(
            f"LoRA target introspection unavailable ({error}); using {list(wanted)}",
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


def run(config):
    import torch
    from huggingface_hub import model_info
    from peft import LoraConfig
    from transformers import AutoProcessor, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "GRPO training requires CUDA; nayana-smoke checks the environment on CPU"
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This first recipe supports one GPU. Do not launch with torchrun"
        )
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        url = config.env_url or stack.enter_context(
            local_server(
                config.snapshot,
                config.num_generations + 4,
                source_root=config.source_root or None,
                cache_dir=config.cache_dir or None,
                local_source=config.local_source,
            )
        )
        with connect(url) as client:
            manifest = client.manifest()
        if "descriptive_vqa" in config.families and not manifest.get("grading", {}).get(
            "descriptive_vqa", {}
        ).get("configured"):
            raise ValueError(
                "Configure the Gemma judge before training descriptive_vqa, or select other --families"
            )
        mode = config.task_input
        if mode == "auto":
            mode = "corpus" if manifest.get("storage") == "bucket-parquet" else "map"
        if config.resume and mode == "corpus":
            raise ValueError(
                "TRL optimizer checkpoint replay for full-corpus iterable input is not yet verified"
            )
        if manifest.get("storage") == "bucket-parquet":
            backend = CorpusAPI(url, manifest["snapshot_id"])
            stack.callback(backend.close)
            if config.evalset:
                # A frozen set is used whole: filtering it would break comparability
                # with every score already reported against this evalset_id.
                frozen = load_evalset(config.evalset, manifest["snapshot_id"])
                selected = (
                    subsample_evalset(frozen, config.eval_limit, config.seed)
                    if config.eval_limit
                    else frozen["tasks"]
                )
                eval_rows = [
                    {
                        key: entry[key]
                        for key in ("task_id", "language", "family", "block_id")
                    }
                    for entry in selected
                ]
                evalset_id = frozen["evalset_id"]
            else:
                # Indexed selection: never enumerate millions of IDs to choose a small eval set.
                eval_rows = backend.sample(
                    "test",
                    config.languages,
                    config.families,
                    config.eval_per_group,
                    config.seed,
                )
                evalset_id = None
            if mode == "corpus":
                stream = BlockTaskStream(
                    backend,
                    languages=config.languages,
                    families=config.families,
                    seed=config.seed,
                    prefetch_blocks=0,
                )
                if not stream.plan:
                    raise ValueError("No indexed training blocks")
                train_rows = {
                    "mode": "full-corpus",
                    "plan_id": stream.plan_id,
                    "blocks": len(stream.plan),
                    "tasks_per_epoch": sum(b["tasks"] for b in stream.plan),
                    "sampling": "hash-shuffled row groups; chunk shuffle; natural task proportions",
                }
                train_dataset = build_corpus_dataset(
                    url,
                    manifest["snapshot_id"],
                    config.languages,
                    config.families,
                    config.seed,
                    config.prefetch_blocks,
                )
            else:
                train_rows = backend.sample(
                    "train",
                    config.languages,
                    config.families,
                    config.train_per_group,
                    config.seed,
                )
                train_dataset = build_dataset(train_rows, mode)
        else:
            if mode == "corpus":
                raise ValueError(
                    "Full-corpus mode requires an indexed bucket-backed environment"
                )
            if config.evalset:
                raise ValueError(
                    "A frozen evaluation set is pinned to a corpus snapshot; "
                    "serve the indexed bucket corpus to use it"
                )
            evalset_id = None
            train_rows = balanced_rows(
                task_rows(url, "train", config.languages, config.families),
                config.languages,
                config.families,
                config.seed,
                config.train_per_group,
            )
            eval_rows = balanced_rows(
                task_rows(url, "test", config.languages, config.families),
                config.languages,
                config.families,
                config.seed,
                config.eval_per_group,
            )
            train_dataset = build_dataset(train_rows, mode)
        revision = model_info(
            config.model, revision=config.model_revision or "main"
        ).sha
        targets = (
            list(config.lora_target_modules)
            if config.lora_target_modules
            else lora_target_modules(config.model, revision)
        )
        print(
            f"LoRA targets: {len(targets)} modules, e.g. {targets[:2]}",
            flush=True,
        )
        metadata = {
            "config": asdict(config),
            "model_revision": revision,
            "lora_target_modules": targets,
            "manifest": manifest,
            "train_tasks": train_rows,
            "eval_tasks": eval_rows,
            "evalset_id": evalset_id,
            # A limited run is a smoke check, not a score comparable with the full set.
            "eval_limit": config.eval_limit or None,
        }
        metadata_path = output / "run-metadata.json"
        if config.resume:
            previous = json.loads(metadata_path.read_text())
            for key in ("model_revision", "manifest", "train_tasks", "eval_tasks"):
                if previous[key] != metadata[key]:
                    raise ValueError(f"Checkpoint replay mismatch: {key}")
            old_config = previous["config"]
            for key, value in json.loads(json.dumps(asdict(config))).items():
                if (
                    key not in {"resume", "trackio_space", "run_name"}
                    and old_config[key] != value
                ):
                    raise ValueError(f"Checkpoint replay config changed: {key}")
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
        cache = AssetCache(url)

        def factory():
            environment = TrainingEnvironment(url, cache, manifest["snapshot_id"])
            stack.callback(environment._close)
            return environment

        processor = AutoProcessor.from_pretrained(
            config.model,
            revision=revision,
            min_pixels=28 * 28,
            max_pixels=config.max_pixels,
        )
        trainer = GRPOTrainer(
            model=config.model,
            processing_class=processor,
            train_dataset=train_dataset,
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
                max_steps=config.max_steps,
                max_completion_length=config.max_completion_length,
                learning_rate=config.learning_rate,
                seed=config.seed,
                temperature=0.9,
                chat_template_kwargs={"enable_thinking": False},
                bf16=True,
                gradient_checkpointing=True,
                mask_truncated_completions=True,
                dataloader_num_workers=0,
                accelerator_config={"dispatch_batches": False},
                logging_steps=1,
                save_strategy="steps",
                save_steps=min(25, config.max_steps),
                save_total_limit=2,
                report_to="trackio" if config.trackio_space else "none",
                trackio_space_id=config.trackio_space or None,
                run_name=config.run_name,
            ),
        )
        baseline = (
            None
            if config.resume
            else evaluate(trainer, eval_rows, url, cache, config.max_completion_length)
        )
        if baseline is not None:
            (output / "baseline.json").write_text(
                json.dumps(baseline, ensure_ascii=False, indent=2) + "\n"
            )

        class CaptureAdapter(TrainerCallback):
            def on_train_begin(self, args, state, control, model=None, **kwargs):
                # Checkpoint weights are loaded before this hook. Compare with the
                # resumed adapter, not the freshly initialized pre-load adapter.
                self.before = {
                    name: p.detach().cpu().clone()
                    for name, p in model.named_parameters()
                    if p.requires_grad
                }

        capture = CaptureAdapter()
        trainer.add_callback(capture)
        result = trainer.train(resume_from_checkpoint=config.resume or None)
        changed = any(
            not torch.equal(capture.before[name], p.detach().cpu())
            for name, p in trainer.model.named_parameters()
            if p.requires_grad
        )
        # GRPO scales each completion by (reward - group mean) / group std, so a group
        # whose completions all score the same contributes exactly zero advantage. When
        # that happens for every group the optimizer correctly does nothing and the
        # adapter cannot change: that is arithmetic, not a broken pipeline, and it must
        # be reported as such instead of as a generic failure.
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
                "and no gradient could flow. The pipeline ran; the batch taught it "
                "nothing. Raise --num-generations, or select task families the model "
                "does not already solve exactly"
            )
        if problems:
            raise RuntimeError("Training check failed: " + "; ".join(problems))
        trained = evaluate(trainer, eval_rows, url, cache, config.max_completion_length)
        trainer.save_model(str(output / "adapter"))
        processor.save_pretrained(str(output / "adapter"))
        summary = {
            "status": "passed",
            "smoke": config.smoke,
            "steps": trainer.state.global_step,
            "training_loss": result.training_loss,
            "adapter_updated": changed,
            "reward_groups_without_variance": degenerate,
            "trained": trained,
            "baseline": json.loads((output / "baseline.json").read_text()),
            "snapshot_id": manifest["snapshot_id"],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", default="")
    parser.add_argument("--env-url", default="")
    parser.add_argument(
        "--evalset",
        default="",
        help="Frozen evaluation set JSON; used whole, in place of ad-hoc sampling",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=[],
        help="Override the resolved LoRA targets",
    )
    parser.add_argument("--source-root", default="")
    parser.add_argument("--cache-dir", default="")
    parser.add_argument("--local-source", action="store_true")
    parser.add_argument("--model", default=Config.model)
    parser.add_argument("--model-revision", default="")
    parser.add_argument("--languages", nargs="+", default=list(Config.languages))
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument(
        "--task-input", choices=("auto", "map", "iterable", "corpus"), default="auto"
    )
    for name in INTEGER_OPTIONS:
        # Default None so --smoke can fill only what the caller left unset.
        parser.add_argument("--" + name.replace("_", "-"), type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=Config.learning_rate)
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", Config.output_dir)
    )
    parser.add_argument("--trackio-space", default="")
    parser.add_argument("--run-name", default=Config.run_name)
    parser.add_argument("--resume", default="")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two optimizer steps; two generations; one eval task per language/family",
    )
    args = vars(parser.parse_args())
    # A smoke run shrinks the defaults, but an explicit flag must still win: silently
    # overriding one makes a failure impossible to investigate by changing it.
    smoke_defaults = {
        "max_steps": 2,
        "num_generations": 2,
        "eval_per_group": 1,
        "eval_limit": 10,
    }
    for name in INTEGER_OPTIONS:
        if args[name] is not None:
            continue
        args[name] = (
            smoke_defaults[name]
            if args["smoke"] and name in smoke_defaults
            else getattr(Config, name)
        )
    run(Config(**args))


if __name__ == "__main__":
    main()
