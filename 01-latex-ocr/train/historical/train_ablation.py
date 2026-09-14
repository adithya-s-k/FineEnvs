#!/usr/bin/env python
"""End-to-end GRPO ablation on the LaTeX-OCR **OpenEnv** environment (one model per GPU).

Self-contained per job:
  1. launches its OWN local OpenEnv server (materialize, row-capped) -> no shared session cap;
  2. loads the model with a LoRA adapter (``target_modules="all-linear"`` -> arch-agnostic);
  3. baseline eval on the test split;
  4. trains with GRPO. Two modes:
       --mode env    : environment_factory (env serves image via reset() + reward via step()).
                       Requires a tool-calling-capable chat template (Qwen / GLM / Gemma-4).
       --mode direct : dataset-direct (images baked into the dataset, reward via the env's
                       LatexOCRRubric in a reward_func). No env_factory -> no tool-calling gate,
                       so templates without tool support (Gemma-3) can train too.
  5. **interval evals** during training (every --eval-interval steps, on --eval-samples) -> eval curve;
  6. final eval;
  7. logs EVERYTHING both to Trackio (live space + optional static freeze) and to local JSON files
     (results.json / train_log.jsonl / eval_log.jsonl / completions.json) so nothing is ever lost.

Run one model per GPU via the SLURM array launcher.
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
from io import BytesIO

import requests
from datasets import Dataset, load_dataset
from PIL import Image

# OpenEnv fork submodule: `openenv` core (src/) + the env package (envs/).
_OPENENV = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "OpenEnv"))
for _p in (os.path.join(_OPENENV, "src"), os.path.join(_OPENENV, "envs")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from latex_ocr_env import LatexOCRAction, LatexOCREnv  # noqa: E402
from latex_ocr_env.server.rubric import LatexOCRRubric  # noqa: E402

INSTRUCTION = (
    "Transcribe the mathematical formula in the image to LaTeX. "
    "Output only the LaTeX code — no Markdown code fences, no $ delimiters, no explanation."
)
DATASET = os.environ.get("LATEX_OCR_DATASET", "unsloth/LaTeX_OCR")
ENV_URL = "http://127.0.0.1:8000"  # set in main()
_RUBRIC = LatexOCRRubric()

# Substrings that mark a parameter/module as part of the VISION tower (everything else — language
# model, embeddings, multimodal projector — counts as "LLM side").
_VISION_MARKERS = ("vision", "visual", "image_encoder", "patch_embed")


def _is_vision(name: str) -> bool:
    n = name.lower()
    return any(m in n for m in _VISION_MARKERS)


def resolve_tune(model_path, tune, r, alpha):
    """Map --tune to (peft_config | None, freeze_keep) covering the six training scopes.

    freeze_keep (for full-* modes) is the component to KEEP trainable ('vision' | 'llm'); the
    complement is frozen post-init. LoRA-component modes enumerate the model's Linear layers on a
    meta device (no weights loaded) and pass their exact names to peft.
    """
    from peft import LoraConfig

    if tune == "full":
        return None, None  # nothing frozen — full fine-tune
    if tune == "full-llm":
        return None, "llm"  # train LLM (+projector/embeds), freeze vision tower
    if tune == "full-vision":
        return None, "vision"  # train vision tower, freeze everything else

    if tune == "lora":
        target = "all-linear"
    else:  # lora-llm | lora-vision -> enumerate Linear names on meta (cheap, no weights)
        import torch
        from accelerate import init_empty_weights
        from transformers import AutoConfig, AutoModelForImageTextToText

        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        with init_empty_weights():
            m = AutoModelForImageTextToText.from_config(cfg, trust_remote_code=True)
        lin = [n for n, mod in m.named_modules() if isinstance(mod, torch.nn.Linear)]
        want_vision = tune == "lora-vision"
        target = [n for n in lin if _is_vision(n) == want_vision]
        if not target:
            raise ValueError(f"tune={tune}: no matching Linear modules (found {len(lin)} total)")
        print(f"[tune] {tune}: LoRA on {len(target)}/{len(lin)} Linear modules", flush=True)
    return LoraConfig(task_type="CAUSAL_LM", r=r, lora_alpha=alpha, target_modules=target), None


# --------------------------------------------------------------------------- #
# helpers                                                                      #
# --------------------------------------------------------------------------- #
def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    m = re.search(r"```[a-zA-Z]*\s*(.*?)\s*```", text, flags=re.DOTALL)  # strip ANY code fence
    if m:
        text = m.group(1)
    return text.strip().strip("$").strip()


def _decode(b64: str, max_px: int = 512) -> Image.Image:
    return _fit(Image.open(BytesIO(base64.b64decode(b64))).convert("RGB"), max_px)


def _fit(img: Image.Image, max_px: int = 512) -> Image.Image:
    img = img.convert("RGB")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    return img


def _write_json(path, obj):
    tmp = f"{path}.tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _append_jsonl(path, obj):
    with open(path, "a") as f:
        f.write(json.dumps(obj) + "\n")


def start_local_env(port: int, max_rows: int) -> subprocess.Popen:
    env = dict(os.environ)
    env["LATEX_OCR_MODE"] = "materialize"
    env["LATEX_OCR_MAX_ROWS"] = str(max_rows)
    env["LATEX_OCR_MAX_SESSIONS"] = "64"   # headroom: 8 training-pool sessions + transient eval client (now closed); was 16, which the old eval-client leak hit at eval@4000
    code = (
        "import uvicorn; from latex_ocr_env.server.app import app; "
        f"uvicorn.run(app, host='127.0.0.1', port={port}, "
        "ws_ping_interval=120, ws_ping_timeout=600, log_level='warning')"
    )
    proc = subprocess.Popen([sys.executable, "-c", code], env=env)
    url = f"http://127.0.0.1:{port}"
    for _ in range(150):
        if proc.poll() is not None:
            raise RuntimeError("local env server exited during startup")
        try:
            if requests.get(f"{url}/healthz", timeout=2).ok:
                print(f"[env] local server ready at {url} (max_rows={max_rows})", flush=True)
                return proc
        except Exception:
            pass
        time.sleep(2)
    raise RuntimeError("local env server did not become healthy")


# --------------------------------------------------------------------------- #
# mode: env (environment_factory)                                              #
# --------------------------------------------------------------------------- #
class LatexOCREnvGRPO:
    def __init__(self):
        self.client = LatexOCREnv(base_url=ENV_URL, connect_timeout_s=60, message_timeout_s=180)

    def reset(self, split: str = "train", index=None, **kwargs):
        idx = int(index) if index is not None else None
        obs = self.client.reset(split=split, index=idx).observation
        return [
            {"type": "image", "image": _decode(obs.image_base64)},
            {"type": "text", "text": obs.prompt or INSTRUCTION},
        ]


def env_reward(completions, environments, **kwargs) -> list[float]:
    out = []
    for env, comp in zip(environments, completions):
        # Send the RAW completion: the env rubric cleans it (fences/$) and applies the length
        # guard on the raw length, so whitespace-padding can't farm free reward (was stripped
        # client-side before, which hid the padding from the guard).
        raw = comp[0]["content"] or ""
        out.append(float(env.client.step(LatexOCRAction(latex=raw)).reward or 0.0))
    return out


def build_dataset_env(num_samples: int, split: str) -> Dataset:
    probe = LatexOCREnv(base_url=ENV_URL, connect_timeout_s=60)
    k = min(num_samples, probe.num_tasks(split))
    print(f"[data:env] {split}: {k} tasks", flush=True)
    return Dataset.from_dict({"split": [split] * k, "index": list(range(k))})


# --------------------------------------------------------------------------- #
# mode: direct (images in the dataset, reward via the env's rubric)            #
# --------------------------------------------------------------------------- #
def rubric_reward(completions, target, **kwargs) -> list[float]:
    # Raw completion in; the rubric cleans + length-guards it (see env_reward).
    return [float(_RUBRIC.grade(c[0]["content"] or "", t).reward) for c, t in zip(completions, target)]


def build_dataset_direct(num_samples: int, split: str) -> Dataset:
    raw = load_dataset(DATASET, split=f"{split}[:{num_samples}]")
    imgs = [_fit(x) for x in raw["image"]]
    tgts = [str(t) for t in raw["text"]]
    prompt = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": INSTRUCTION}]}]
    print(f"[data:direct] {split}: {len(imgs)} rows", flush=True)
    return Dataset.from_dict({"prompt": [prompt] * len(imgs), "image": imgs, "target": tgts})


# --------------------------------------------------------------------------- #
# eval (uses the env test split in both modes for a consistent reward)         #
# --------------------------------------------------------------------------- #
def generate_latex(model, processor, image, prompt, is_qwen, no_template, max_new_tokens=256):
    import torch

    if no_template or processor.chat_template is None:
        inputs = processor(text=prompt, images=image, return_tensors="pt").to(model.device)
    else:
        kw = {"enable_thinking": False} if is_qwen else {}
        inputs = processor.apply_chat_template(
            [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]}],
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **kw,
        ).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    return processor.batch_decode(out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True)[0]


def evaluate(model, processor, n_samples, is_qwen, no_template, split="test", collect=0, max_new_tokens=256):
    """Mean env reward over first N test images (greedy). KV cache on / grad-ckpt off for speed."""
    from tqdm.auto import tqdm

    client = LatexOCREnv(base_url=ENV_URL, connect_timeout_s=60, message_timeout_s=180)
    prompt = "ocr\n" if no_template else INSTRUCTION
    was_training, was_gc = model.training, getattr(model, "is_gradient_checkpointing", False)
    model.eval()
    if was_gc:
        model.gradient_checkpointing_disable()
    model.config.use_cache = True
    scores, samples = [], []
    try:
        pbar = tqdm(range(n_samples), desc=f"eval[{split}]")
        for i in pbar:
            obs = client.reset(split=split, index=i).observation
            raw = generate_latex(model, processor, _decode(obs.image_base64), prompt, is_qwen, no_template, max_new_tokens)
            res = client.step(LatexOCRAction(latex=raw))  # raw in; rubric cleans + length-guards
            r = float(res.reward or 0.0)
            scores.append(r)
            pbar.set_postfix(avg_reward=f"{sum(scores) / len(scores):.3f}")
            if len(samples) < collect:
                samples.append({"raw": raw[:300], "target": getattr(res.observation, "target_latex", "")[:300], "reward": r})
    finally:
        # Close the eval client so its websocket / server session is released. Without this, every
        # eval leaks a session (the sync client's run_forever daemon thread keeps it alive, never GC'd);
        # baseline + 7 interval evals + the 8 training-pool sessions hit LATEX_OCR_MAX_SESSIONS at
        # eval@4000 -> SessionCapacityError -> ConnectionClosedOK, which killed every run at ~step 3999.
        try:
            client.close()
        except Exception:
            pass
        model.config.use_cache = False
        if was_gc:
            model.gradient_checkpointing_enable()
        if was_training:
            model.train()
    mean = sum(scores) / len(scores) if scores else 0.0
    return (mean, samples) if collect else mean


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--run-name", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["env", "direct"], default="env")
    p.add_argument("--env-port", type=int, default=8000)
    p.add_argument("--max-steps", type=int, default=20)
    p.add_argument("--num-train-samples", type=int, default=500)
    p.add_argument("--num-generations", type=int, default=8)
    p.add_argument("--eval-samples", type=int, default=200)
    p.add_argument("--eval-interval", type=int, default=0, help="steps between mid-training evals (0=off)")
    p.add_argument("--save-steps", type=int, default=0, help="checkpoint interval (0=end only)")
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument(
        "--mask-truncated", default="true", choices=["true", "false"],
        help="mask completions truncated at max_length (True=TRL default). Set false for models "
             "that don't emit EOS (glm/gemma over-generate to the cap → all-masked → zero gradient).",
    )
    p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--temperature", type=float, default=0.9, help="sampling temperature (lower = more stable)")
    p.add_argument("--beta", type=float, default=0.0, help="KL-to-reference coefficient (>0 stabilizes, prevents collapse)")
    p.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping (lower = more stable)")
    p.add_argument(
        "--tune",
        choices=["full", "full-llm", "full-vision", "lora", "lora-llm", "lora-vision"],
        default="lora",
        help="training scope: full FT / LoRA, optionally restricted to the LLM or vision tower",
    )
    p.add_argument("--optim", default="adamw_torch", help="e.g. adamw_torch, adamw_8bit (for full FT of big models)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument(
        "--vllm", default="off", choices=["off", "colocate", "server"],
        help="generation backend: 'off' uses HF .generate(); 'colocate' runs vLLM in-process on the "
             "training GPU; 'server' connects to a separate `trl vllm-serve` (2nd GPU, NCCL weight-sync).",
    )
    p.add_argument("--vllm-gpu-mem", type=float, default=0.3, help="vLLM colocate GPU memory fraction")
    p.add_argument("--vllm-max-len", type=int, default=16384, help="vLLM colocate max model length (KV cache cap)")
    p.add_argument("--vllm-base-url", default="", help="server mode: trl vllm-serve base URL (or env VLLM_BASE_URL)")
    p.add_argument("--vllm-group-port", type=int, default=51216,
                   help="server mode: weight-sync NCCL group port; must be unique per co-located task")
    p.add_argument("--trackio-space", default="")
    p.add_argument("--trackio-project", default="latex-ocr-ablation")
    p.add_argument("--trackio-static", action="store_true")
    p.add_argument("--trackio-static-space", default="", help="static space id (default: <space>-static)")
    return p.parse_args()


def main():
    global ENV_URL
    args = parse_args()
    is_qwen = "qwen" in args.model.lower()
    no_template = "paligemma" in args.model.lower()
    trackio_on = bool(args.trackio_space)
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "results.json")
    train_log = os.path.join(args.output_dir, "train_log.jsonl")
    eval_log = os.path.join(args.output_dir, "eval_log.jsonl")

    results = {
        "model": args.model, "run": args.run_name, "mode": args.mode, "status": "starting",
        "config": vars(args), "baseline": None, "eval_curve": [], "final": None, "delta": None,
    }
    _write_json(results_path, results)

    max_rows = max(args.num_train_samples, args.eval_samples)
    server = start_local_env(args.env_port, max_rows)
    ENV_URL = f"http://127.0.0.1:{args.env_port}"

    try:
        from transformers import TrainerCallback
        from trl import GRPOConfig, GRPOTrainer

        if args.mode == "env":
            train_dataset = build_dataset_env(args.num_train_samples, "train")
            reward_funcs, env_factory = env_reward, LatexOCREnvGRPO
        else:
            train_dataset = build_dataset_direct(args.num_train_samples, "train")
            reward_funcs, env_factory = rubric_reward, None

        static_space = args.trackio_static_space or (f"{args.trackio_space}-static" if args.trackio_space else "")
        # vLLM colocate: in-process generation on the training GPU. max_model_length caps the KV
        # cache (colocate has no --max-model-len flag, so it would otherwise size for the model's
        # full context and fail to fit alongside the trainer).
        if args.vllm == "colocate":
            vllm_kwargs = {
                "use_vllm": True,
                "vllm_mode": "colocate",
                "vllm_gpu_memory_utilization": args.vllm_gpu_mem,
                "vllm_max_model_length": args.vllm_max_len,
            }
        elif args.vllm == "server":
            base = args.vllm_base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000")
            vllm_kwargs = {
                "use_vllm": True,
                "vllm_mode": "server",
                "vllm_server_base_url": base,
                "vllm_group_port": args.vllm_group_port,
            }
        else:
            vllm_kwargs = {}
        cfg = GRPOConfig(
            output_dir=args.output_dir,
            model_init_kwargs={"dtype": "bfloat16", "trust_remote_code": True},
            learning_rate=args.learning_rate,
            optim=args.optim,
            num_generations=args.num_generations,
            per_device_train_batch_size=args.num_generations,
            steps_per_generation=1,
            gradient_accumulation_steps=1,
            max_steps=args.max_steps,
            max_completion_length=args.max_completion_length,
            temperature=args.temperature,
            beta=args.beta,
            max_grad_norm=args.max_grad_norm,
            chat_template_kwargs=({"enable_thinking": False} if is_qwen else None),
            mask_truncated_completions=(args.mask_truncated == "true"),
            bf16=True,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            report_to=("trackio" if trackio_on else "none"),
            run_name=args.run_name,
            project=args.trackio_project,
            trackio_space_id=(args.trackio_space or None),
            trackio_static_space_id=(static_space if (trackio_on and args.trackio_static) else False),
            logging_steps=1,
            # Kill the tqdm progress bar: it races with Trackio's background HF-Hub upload
            # bars over tqdm's global _instances set ("Set changed size during iteration").
            disable_tqdm=True,
            # Don't log completion tables or images to Trackio — the media/trace uploads are
            # heavy and crash the Space. Metrics (reward/loss/eval) still log fine.
            log_completions=False,
            log_multimodal=False,
            **vllm_kwargs,
            save_strategy=("steps" if args.save_steps else "no"),
            save_steps=(args.save_steps or 500),
        )
        peft_config, freeze_keep = resolve_tune(args.model, args.tune, args.lora_r, args.lora_alpha)
        trainer = GRPOTrainer(
            model=args.model, train_dataset=train_dataset, reward_funcs=reward_funcs,
            environment_factory=env_factory, peft_config=peft_config, args=cfg,
        )
        processor = trainer.processing_class
        if freeze_keep is not None:  # full-llm / full-vision: freeze the complement
            tot = tr = 0
            for n, p in trainer.model.named_parameters():
                keep = _is_vision(n) if freeze_keep == "vision" else (not _is_vision(n))
                p.requires_grad = keep
                tot += p.numel()
                tr += p.numel() if keep else 0
            print(f"[tune] full-{freeze_keep}: trainable {tr / 1e6:.1f}M / {tot / 1e6:.1f}M params", flush=True)

        def _log_eval_point(step, r):
            # local (guaranteed) — dedup by step so interval@max_steps and the post-train final don't double up
            if not results["eval_curve"] or results["eval_curve"][-1]["step"] != step:
                results["eval_curve"].append({"step": step, "reward": r})
            _append_jsonl(eval_log, {"step": step, "reward": r})
            _write_json(results_path, results)

        def _trackio_eval(step, r):
            # best-effort: only valid while the trainer's Trackio run is OPEN (during training)
            if not trackio_on:
                return
            try:
                import trackio
                trackio.log({"eval/test_reward": r}, step=step)
            except Exception as e:
                print(f"[trackio] eval log @{step} skipped: {e}", flush=True)

        # -- local JSONL mirror of every trainer log (train/*) --
        class LocalLogger(TrainerCallback):
            def on_log(self, a, state, control, logs=None, **kw):
                if logs:
                    _append_jsonl(train_log, {"step": state.global_step, **dict(logs)})

        # -- interval eval during training -> eval curve; Trackio only while the run is open --
        class IntervalEval(TrainerCallback):
            def on_train_begin(self, a, state, control, **kw):
                if results["baseline"] is not None:
                    _trackio_eval(0, results["baseline"])  # put baseline on the Trackio curve

            def on_step_end(self, a, state, control, **kw):
                step = state.global_step
                if args.eval_interval and step > 0 and step % args.eval_interval == 0:
                    r = evaluate(trainer.model, processor, args.eval_samples, is_qwen, no_template, max_new_tokens=args.max_completion_length)
                    _log_eval_point(step, r)
                    _trackio_eval(step, r)
                    print(f"[{args.run_name}] eval@{step} test_reward={r:.4f}", flush=True)

        trainer.add_callback(LocalLogger())
        trainer.add_callback(IntervalEval())

        print(f"[{args.run_name}] baseline eval...", flush=True)
        baseline = evaluate(trainer.model, processor, args.eval_samples, is_qwen, no_template, max_new_tokens=args.max_completion_length)
        results["baseline"] = baseline
        _log_eval_point(0, baseline)
        print(f"[{args.run_name}] baseline = {baseline:.4f}", flush=True)

        results["status"] = "training"
        _write_json(results_path, results)
        trainer.train()  # Trackio train run opens/closes inside here

        print(f"[{args.run_name}] final eval...", flush=True)
        final, samples = evaluate(trainer.model, processor, args.eval_samples, is_qwen, no_template, collect=6, max_new_tokens=args.max_completion_length)
        results.update(status="done", final=final, delta=final - baseline)
        _log_eval_point(args.max_steps, final)  # Trackio final already logged via the interval@max_steps
        _write_json(os.path.join(args.output_dir, "completions.json"), samples)
        print(f"[{args.run_name}] RESULT before={baseline:.4f} after={final:.4f} delta={final - baseline:+.4f}", flush=True)
    except Exception as e:
        results["status"] = f"error: {type(e).__name__}: {e}"
        _write_json(results_path, results)
        raise
    finally:
        server.terminate()


if __name__ == "__main__":
    main()
