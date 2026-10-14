"""SFT of a Qwen VLM on DesignGym trajectories with TRL (LoRA), in the evaluation's exact prompt layout.

    torchrun --nproc_per_node 4 train.py --data /data --hub-model-id <namespace>/designgym-sft-... \\
        --max-steps 1500 --save-minutes 45 --hours 7.3

Data: the released `FineEnvs/designworld-sft` parquet files (`data/<config>/*.parquet`), listed under --data when an
HF bucket mirror is mounted there (else listed from the dataset repo), streamed per GPU rank and dataloader worker.
Configs are mixed by weight; each trajectory contributes --windows-per-row random windows per pass (`cua_format.windows`, 8/3 screenshots). `sft_format.conversation` lays each window out exactly as
evaluation prompts the model and `collator.render_text` / `Collator.labels_for` put loss on trainable assistant
turns only (checked offline by check_format.py). Windows longer than --max-tokens are skipped and counted.

Training: TRL SFTTrainer with a PEFT LoRA (rank 32) on the language model's attention, linear attention and
shared-expert projections; bf16, gradient checkpointing, cosine LR. Checkpoints (model only) are pushed to the Hub
as `checkpoint-<step>/` at every save, so the run can be stopped at any time; a wall-clock guard saves and stops
before --hours.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, TrainerCallback
from trl import SFTConfig, SFTTrainer

import sft_format
from collator import Collator

LORA_TARGETS = (r".*language_model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)|mlp\.shared_expert\.(gate_proj|up_proj|down_proj))")


def held_out(task_id: str) -> bool:
    return int(hashlib.sha256(f"sft:{task_id}".encode()).hexdigest()[:8], 16) / 16**8 < 0.01


def _hub_files(repo: str, root: Path, configs) -> dict[str, list[Path]]:
    """`data/<config>/*.parquet` of the dataset repo, as paths under `root` (only their last two parts are used)."""
    from huggingface_hub import HfApi

    files: dict[str, list[Path]] = {c: [] for c in configs}
    for name in HfApi().list_repo_files(repo, repo_type="dataset"):
        parts = name.split("/")
        if len(parts) == 3 and parts[0] == "data" and parts[1] in files and parts[2].endswith(".parquet"):
            files[parts[1]].append(root / parts[1] / parts[2])
    return {c: sorted(v) for c, v in files.items()}


class Windows(IterableDataset):
    """Processed training windows, streamed: each (rank, worker) reads its own share of the parquet files."""

    def __init__(self, root: Path, weights: dict[str, float], processor, *, windows_per_row: int, max_tokens: int,
                 split: str, seed: int, shard: bool = True, source: str = "hub"):
        self.mirror, self.source = "FineEnvs/designworld-sft", source
        if root.is_dir():
            self.files = {c: sorted((root / c).glob("*.parquet")) for c in weights}
        else:  # no mount: the same listing from the dataset repo (files are then read over HTTP)
            self.files = _hub_files(self.mirror, root, weights)
        self.weights = {c: w for c, w in weights.items() if self.files[c]}
        self.processor, self.windows_per_row, self.max_tokens = processor, windows_per_row, max_tokens
        self.split, self.seed, self.shard = split, seed, shard
        self.collator = Collator(processor)
        self.skipped = 0

    def _shard(self):
        if not self.shard:
            return 0, 1
        rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        info = get_worker_info()
        wid, nw = (info.id, info.num_workers) if info else (0, 1)
        return rank * nw + wid, world * nw

    def _read(self, path: Path) -> list[dict]:
        rows = pq.read_table(pa.BufferReader(self._bytes(path))).to_pylist()
        return [r for r in rows if held_out(r["task_id"]) == (self.split == "eval")]

    def _bytes(self, path: Path) -> bytes:
        """The whole file in one read. Default: over HTTP from the dataset repo (Xet), which under load beats the bucket
        mount: that answers EIO under many concurrent readers and pyarrow's small random reads crawl on it (~2 MB/s).
        `--data-source mount` reads the mount (sequentially), falling back to HTTP on errors."""
        if self.source == "mount":
            for attempt in range(2):
                try:
                    return path.read_bytes()
                except OSError as e:
                    print(f"[data] {path}: {e} (attempt {attempt + 1})", flush=True)
                    time.sleep(2)
        from huggingface_hub import hf_hub_download

        rel = f"data/{path.parent.name}/{path.name}"
        for attempt in range(5):
            try:
                local = Path(hf_hub_download(self.mirror, rel, repo_type="dataset", cache_dir=f"/tmp/hf-data-{os.getpid()}"))
                data = local.read_bytes()
                local.unlink(missing_ok=True)
                return data
            except Exception as e:  # noqa: BLE001 - transient network errors: back off and retry
                print(f"[data] {rel}: {e!r} (HTTP attempt {attempt + 1})", flush=True)
                time.sleep(5 * (attempt + 1))
        raise OSError(f"could not read {rel}")

    def _rows(self, config: str, rng: random.Random, shard: int, shards: int):
        """Rows of this shard's files, in random file order; the next file is read in a background thread while
        this one is used (the
        next two, in background threads)."""
        files = self.files[config][shard::shards] or self.files[config]
        with ThreadPoolExecutor(2) as pool:
            while True:
                order = rng.sample(files, len(files))
                ahead = [pool.submit(self._read, f) for f in order[:2]]
                for i in range(len(order)):
                    rows = ahead.pop(0).result()
                    if i + 2 < len(order):
                        ahead.append(pool.submit(self._read, order[i + 2]))
                    yield from rows
                if self.split == "eval":
                    for f in ahead:
                        f.cancel()
                    return

    def __iter__(self):
        shard, shards = self._shard()
        rng = random.Random(self.seed * 1000 + shard)
        streams = {c: self._rows(c, rng, shard, shards) for c in self.weights}
        while streams:
            c = rng.choices(list(streams), [self.weights[k] for k in streams])[0]
            try:
                row = next(streams[c])
            except StopIteration:
                del streams[c]
                continue
            ws = sft_format.windows(row)
            if self.windows_per_row and len(ws) > self.windows_per_row:
                ws = rng.sample(ws, self.windows_per_row)
            for w in ws:
                conv = sft_format.conversation(w)
                if not any(conv["trainable"]):
                    continue
                item = self.collator([conv])
                n = item["input_ids"].shape[1]
                if n > self.max_tokens:
                    self.skipped += 1
                    continue
                yield {k: v[0] if k in ("input_ids", "attention_mask", "labels") else v for k, v in item.items()}


def pad_collate(items: list[dict]) -> dict:
    """Batches of already-processed windows (right-padded; pads are masked and unlabelled)."""
    n = max(it["input_ids"].shape[0] for it in items)
    out = {"input_ids": torch.zeros(len(items), n, dtype=torch.long),
           "attention_mask": torch.zeros(len(items), n, dtype=torch.long),
           "labels": torch.full((len(items), n), -100, dtype=torch.long)}
    for i, it in enumerate(items):
        m = it["input_ids"].shape[0]
        out["input_ids"][i, :m], out["attention_mask"][i, :m], out["labels"][i, :m] = it["input_ids"], it["attention_mask"], it["labels"]
    for k in ("pixel_values", "image_grid_thw"):
        parts = [it[k] for it in items if it.get(k) is not None]
        if parts:
            out[k] = torch.cat(parts)
    if "mm_token_type_ids" in items[0]:
        out["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence([it["mm_token_type_ids"][0] for it in items], batch_first=True)
    return out


class Trainer(SFTTrainer):
    """Each rank streams its own share of the data (`Windows` shards by RANK), so the train loader is not wrapped by
    accelerate, which would otherwise read the whole stream on every rank and keep one batch in WORLD_SIZE."""

    def training_step(self, *args, **kwargs):
        t = time.time()
        out = super().training_step(*args, **kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.timer.compute += time.time() - t
        return out

    def get_train_dataloader(self) -> DataLoader:
        a = self.args
        return DataLoader(self.train_dataset, batch_size=a.per_device_train_batch_size, collate_fn=self.data_collator,
                          num_workers=a.dataloader_num_workers, pin_memory=True,
                          persistent_workers=a.dataloader_persistent_workers and a.dataloader_num_workers > 0,
                          prefetch_factor=a.dataloader_prefetch_factor if a.dataloader_num_workers else None)


class GpuMemory(TrainerCallback):
    """Peak GPU memory and the share of wall time spent in forward/backward since the last log (the rest is mostly
    waiting for data), in the logs: they size the window cap and the dataloader workers."""

    def __init__(self):
        self.compute, self.since = 0.0, time.time()

    def on_log(self, args, state, control, logs=None, **kw):
        if logs is None or "loss" not in logs:
            return
        if torch.cuda.is_available():
            logs["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
            torch.cuda.reset_peak_memory_stats()
        logs["compute_frac"] = round(self.compute / max(time.time() - self.since, 1e-6), 3)
        self.compute, self.since = 0.0, time.time()


class WallClock(TrainerCallback):
    """Checkpoint (pushed to the Hub) and evaluate every `save_minutes` of wall time, and save and stop before the
    job's time cap, so a checkpoint is never more than one interval old and the last one is never lost."""

    def __init__(self, hours: float, save_minutes: float, evaluate: bool = True):
        self.deadline, self.evaluate = time.time() + hours * 3600, evaluate
        self.every, self.last = save_minutes * 60, time.time()

    def on_step_end(self, args, state, control, **kw):
        now = time.time()
        if self.every and now - self.last > self.every:
            control.should_save, control.should_evaluate = True, self.evaluate
            self.last = now
        if now > self.deadline:
            control.should_save = True
            control.should_training_stop = True
        return control


class _TorchLogProbs:
    """Plain-torch stand-in for TRL's Triton chunked log-prob kernel (Triton is Linux-only): laptop tests only."""

    @staticmethod
    def apply(hidden, weight, bias, labels, temperature, _chunk, _softcap, scale, outputs):
        logits = torch.nn.functional.linear(hidden, weight, bias) * scale / temperature
        lp = logits.float().log_softmax(-1)
        return (lp.gather(-1, labels[:, None]).squeeze(-1), -(lp.exp() * lp).sum(-1) if "entropy" in outputs else None,
                None, None, (logits.argmax(-1) == labels).float() if "is_top1" in outputs else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    ap.add_argument("--data", default="/data", help="the bucket mount (file listing; data too with --data-source mount)")
    ap.add_argument("--data-source", choices=("hub", "mount"), default="hub")
    ap.add_argument("--mix", default="gui_studio=1,gui_template=1,gui_pixel=1,gui_deck=1,gui_present=1,gui_vector=1,tools=1,html=1")
    ap.add_argument("--windows-per-row", type=int, default=1, help="windows sampled per trajectory per pass (0: all)")
    ap.add_argument("--max-tokens", type=int, default=40000)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=50)
    ap.add_argument("--save-steps", type=int, default=0, help="checkpoint every N steps (0: only by --save-minutes)")
    ap.add_argument("--save-minutes", type=float, default=45)
    ap.add_argument("--eval-windows", type=int, default=32)
    ap.add_argument("--hours", type=float, default=7.6)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--output", default="/work/out")
    ap.add_argument("--hub-model-id", default=None)
    ap.add_argument("--report-to", default="none")
    ap.add_argument("--tiny", action="store_true", help="random miniature model (local pipeline test)")
    a = ap.parse_args()

    from peft import LoraConfig

    root = Path(a.data)
    root = root / "data" if (root / "data").exists() else root
    weights = {k: float(v) for k, v in (p.split("=") for p in a.mix.split(","))}
    processor = AutoProcessor.from_pretrained(a.model)
    train = Windows(root, weights, processor, windows_per_row=a.windows_per_row, max_tokens=a.max_tokens, split="train", seed=0, source=a.data_source)
    ev = Windows(root, weights, processor, windows_per_row=2, max_tokens=a.max_tokens, split="eval", seed=1, shard=False, source=a.data_source)
    eval_items = []
    for item in ev if a.eval_windows else ():
        eval_items.append(item)
        if len(eval_items) >= a.eval_windows:
            break

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    if a.tiny:  # a randomly initialised miniature of the architecture, for pipeline tests on a laptop
        config = AutoConfig.from_pretrained(a.model)
        t, v = config.text_config, config.vision_config
        t.update(dict(num_hidden_layers=4, layer_types=t.layer_types[:4], hidden_size=128, num_experts=8,
                      num_experts_per_tok=2, moe_intermediate_size=64, shared_expert_intermediate_size=64,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=64, linear_num_key_heads=2,
                      linear_num_value_heads=2, linear_key_head_dim=32, linear_value_head_dim=32, mtp_num_hidden_layers=0))
        v.update(dict(depth=2, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=128))
        model = AutoModelForImageTextToText.from_config(config, dtype=torch.float32)
        import trl.trainer.utils as trl_utils
        if trl_utils._ChunkedLogProbFunction is None:
            trl_utils._ChunkedLogProbFunction = _TorchLogProbs
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                            device_map={"": device})
    cfg = SFTConfig(
        output_dir=a.output, max_steps=a.max_steps, per_device_train_batch_size=1, gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * a.max_steps)), bf16=not a.tiny, use_cpu=not torch.cuda.is_available(),
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=1, logging_first_step=True, include_num_input_tokens_seen="all",
        save_strategy="steps", save_steps=a.save_steps or 10**9, save_only_model=True,
        eval_strategy="steps" if eval_items else "no", eval_steps=a.save_steps or 10**9, per_device_eval_batch_size=1,
        push_to_hub=bool(a.hub_model_id), hub_model_id=a.hub_model_id, hub_strategy="all_checkpoints",
        hub_private_repo=False, max_length=None, dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False, dataloader_num_workers=a.workers, dataloader_persistent_workers=True,
        dataloader_prefetch_factor=4 if a.workers else None,
        ddp_find_unused_parameters=False, prediction_loss_only=True, router_aux_loss_coef=0.0, report_to=a.report_to, seed=0,
    )
    placeholder = datasets.Dataset.from_dict({"input_ids": [[0]]})
    trainer = Trainer(
        model=model, args=cfg, train_dataset=placeholder, eval_dataset=placeholder if eval_items else None,
        processing_class=processor,
        data_collator=pad_collate, callbacks=[WallClock(a.hours, a.save_minutes, bool(eval_items)), timer := GpuMemory()],
        peft_config=LoraConfig(r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.0, target_modules=LORA_TARGETS,
                               task_type="CAUSAL_LM"),
    )
    trainer.timer = timer
    # SFTTrainer only accepts `datasets` objects; the windows are already processed, so the streams go in afterwards
    trainer.train_dataset, trainer.eval_dataset = train, (eval_items or None)
    if trainer.is_world_process_zero():
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(json.dumps({"trainable_params": trainable, "eval_windows": len(eval_items), "args": vars(a)}), flush=True)
    trainer.train()
    trainer.save_model(f"{a.output}/final")
    if a.hub_model_id:
        trainer.push_to_hub(commit_message="final")
    print(json.dumps({"skipped_long_windows": train.skipped}), flush=True)


if __name__ == "__main__":
    main()
