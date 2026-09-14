"""Single-GPU GRPO recipe shared by the notebook and HF Jobs.

uv run --project 01-latex-ocr/envs/latex_ocr --extra train python \
    01-latex-ocr/train/grpo_latex_ocr.py --smoke
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
from contextlib import ExitStack
from dataclasses import asdict, dataclass
from pathlib import Path

from latex_ocr_env import LatexOCRAction, LatexOCREnv
from latex_ocr_env.runtime import local_server
from PIL import Image

DEFAULT_OUTPUT_DIR = str(Path(__file__).resolve().parents[1] / "results" / "local-run")


@dataclass
class Config:
    model: str = "Qwen/Qwen3-VL-2B-Instruct"
    env_url: str = (
        ""  # Empty starts a bounded materialize-mode server on this GPU host.
    )
    max_steps: int = 30
    num_generations: int = 8
    eval_samples: int = 50
    max_completion_length: int = 256
    image_size: int = 512
    learning_rate: float = 1e-5
    seed: int = 42
    output_dir: str = DEFAULT_OUTPUT_DIR
    trackio_space: str = ""
    run_name: str = "latex-ocr-grpo"
    push_repo_id: str = ""
    smoke: bool = False

    def validate(self):
        if self.max_steps < 1 or self.eval_samples < 1 or self.num_generations < 2:
            raise ValueError(
                "Need positive steps/eval samples and at least two generations"
            )
        if self.max_completion_length < 1 or self.image_size < 28:
            raise ValueError(
                "Completion length must be positive and image size at least 28"
            )


def connect(url):
    return LatexOCREnv(base_url=url, connect_timeout_s=60, message_timeout_s=180).sync()


def decode_image(encoded, size=512):
    image = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    return image


class TrainingEnvironment:
    # Public methods become TRL tools. Keep cleanup private: OCR is single-turn.
    def __init__(self, url, image_size=512):
        self.client = connect(url)
        self.image_size = image_size

    def reset(self, split="train", index=0, **kwargs):
        observation = self.client.reset(split=split, index=int(index)).observation
        if observation.done or not observation.image_base64:
            raise RuntimeError("Environment returned no task image")
        return [
            {
                "type": "image",
                "image": decode_image(observation.image_base64, self.image_size),
            },
            {"type": "text", "text": observation.prompt},
        ]

    def _close(self):
        self.client.close()


def completion_text(completion):
    if isinstance(completion, str):
        return completion
    content = completion[-1]["content"]
    if isinstance(content, str):
        return content
    return "".join(
        block.get("text", "") for block in content if block.get("type") == "text"
    )


def env_reward(completions, environments, **kwargs):
    # Send the RAW completion; stripping whitespace here bypasses the length guard.
    scores = []
    for environment, completion in zip(environments, completions, strict=True):
        result = environment.client.step(
            LatexOCRAction(latex=completion_text(completion))
        )
        if (
            not result.done
            or result.reward is None
            or not math.isfinite(float(result.reward))
        ):
            raise RuntimeError("Environment returned an invalid terminal reward")
        scores.append(float(result.reward))
    return scores


def build_dataset(url, n):
    from datasets import Dataset

    with connect(url) as client:
        count = client.num_tasks("train")
        if count <= 0:
            raise ValueError(
                "Indexed GRPO needs a nonempty materialize-mode train split"
            )
        # Fail before loading the model if the server is in stream mode.
        client.reset(split="train", index=0)
    count = min(n, count)
    return Dataset.from_dict({"split": ["train"] * count, "index": list(range(count))})


def generate_latex(model, processor, image, prompt, max_new_tokens):
    import torch

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False
        )
    return processor.batch_decode(
        output[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )[0]


def evaluate(model, processor, config, url):
    was_training = model.training
    was_checkpointing = getattr(model, "is_gradient_checkpointing", False)
    was_cache = model.config.use_cache
    rows = []
    try:
        model.eval()
        if was_checkpointing:
            model.gradient_checkpointing_disable()
        model.config.use_cache = True
        with connect(url) as client:
            count = min(config.eval_samples, client.num_tasks("test"))
            if count < 1:
                raise ValueError("Evaluation needs a nonempty test split")
            for index in range(count):
                observation = client.reset(split="test", index=index).observation
                prediction = generate_latex(
                    model,
                    processor,
                    decode_image(observation.image_base64, config.image_size),
                    observation.prompt,
                    config.max_completion_length,
                )
                result = client.step(LatexOCRAction(latex=prediction))
                if result.reward is None or not math.isfinite(float(result.reward)):
                    raise RuntimeError("Evaluation returned an invalid reward")
                rows.append(
                    {
                        "index": index,
                        "reward": float(result.reward),
                        "prediction": prediction,
                        "target": result.observation.target_latex,
                    }
                )
                print(
                    f"eval {index + 1}/{count}: reward={result.reward:.4f}", flush=True
                )
    finally:
        model.config.use_cache = was_cache
        if was_checkpointing:
            model.gradient_checkpointing_enable()
        model.train(was_training)
    return {
        "mean_reward": sum(row["reward"] for row in rows) / len(rows),
        "samples": rows,
    }


def run(config: Config):
    import torch
    from peft import LoraConfig
    from trl import GRPOConfig, GRPOTrainer

    config.validate()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "Training requires CUDA. Use latex-ocr-smoke for a CPU environment check."
        )
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This recipe uses one GPU; launch it with python, not torchrun"
        )
    output = Path(config.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(asdict(config), indent=2) + "\n")
    with ExitStack() as stack:
        url = config.env_url or stack.enter_context(
            local_server(
                max_rows=max(config.max_steps, config.eval_samples),
                sessions=config.num_generations + 4,
            )
        )
        dataset = build_dataset(url, config.max_steps)

        def factory():
            environment = TrainingEnvironment(url, config.image_size)
            stack.callback(environment._close)
            return environment

        trainer = GRPOTrainer(
            model=config.model,
            train_dataset=dataset,
            environment_factory=factory,
            reward_funcs=env_reward,
            peft_config=LoraConfig(
                task_type="CAUSAL_LM",
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                target_modules=["q_proj", "v_proj"],
            ),
            args=GRPOConfig(
                output_dir=str(output),
                model_init_kwargs={"dtype": "bfloat16", "attn_implementation": "sdpa"},
                num_generations=config.num_generations,
                per_device_train_batch_size=config.num_generations,
                max_steps=config.max_steps,
                max_completion_length=config.max_completion_length,
                learning_rate=config.learning_rate,
                temperature=0.9,
                seed=config.seed,
                chat_template_kwargs={"enable_thinking": False},
                mask_truncated_completions=True,
                bf16=True,
                gradient_checkpointing=True,
                logging_steps=1,
                save_strategy="steps",
                save_steps=config.max_steps,
                save_total_limit=1,
                log_completions=False,
                report_to="trackio" if config.trackio_space else "none",
                trackio_space_id=config.trackio_space or None,
                run_name=config.run_name,
            ),
        )
        processor = trainer.processing_class
        baseline = evaluate(trainer.model, processor, config, url)
        (output / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
        # A short run need not improve reward, but it must update an adapter.
        before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        }
        result = trainer.train()
        updated = any(
            not torch.equal(before[name], parameter.detach().cpu())
            for name, parameter in trainer.model.named_parameters()
            if parameter.requires_grad
        )
        if trainer.state.global_step != config.max_steps or not math.isfinite(
            result.training_loss
        ):
            raise RuntimeError("Training did not complete with a finite loss")
        if not updated:
            raise RuntimeError(
                "No adapter weights changed; inspect truncated completions and reward variance"
            )
        trained = evaluate(trainer.model, processor, config, url)
        trainer.save_model(str(output / "adapter"))
        processor.save_pretrained(str(output / "adapter"))
        summary = {
            "status": "passed",
            "smoke": config.smoke,
            "steps": trainer.state.global_step,
            "training_loss": result.training_loss,
            "adapter_updated": updated,
            "baseline": baseline,
            "trained": trained,
            "reward_delta": trained["mean_reward"] - baseline["mean_reward"],
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2), flush=True)
        if config.push_repo_id:
            merged = trainer.model.merge_and_unload()
            merged.push_to_hub(config.push_repo_id)
            processor.push_to_hub(config.push_repo_id)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=Config.model)
    parser.add_argument("--env-url", default="")
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--eval-samples", type=int, default=50)
    parser.add_argument("--max-completion-length", type=int, default=256)
    parser.add_argument(
        "--output-dir", default=os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_DIR)
    )
    parser.add_argument("--trackio-space", default="")
    parser.add_argument("--run-name", default="latex-ocr-grpo")
    parser.add_argument("--push-repo-id", default="")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Two optimizer steps, two generations, two test images",
    )
    args = vars(parser.parse_args())
    if args["smoke"]:
        args.update(max_steps=2, num_generations=2, eval_samples=2)
    run(Config(**args))


if __name__ == "__main__":
    main()
