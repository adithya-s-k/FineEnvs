"""Durable scalar logs and checkpoint handoffs for a separate evaluation watcher."""
import json
import math
from pathlib import Path
import time

from transformers import TrainerCallback
from recipe import write_json


class Logs(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        values = {k: v for k, v in (logs or {}).items()
                  if isinstance(v, (int, float)) and math.isfinite(v)}
        path = Path(args.output_dir) / "metrics.jsonl"
        with path.open("a") as stream:
            stream.write(json.dumps({"step": state.global_step, "time": time.time(), **values}) + "\n")
            stream.flush()


def publish_checkpoint(path, cfg, step, *, final=False):
    from runtime.checkpoints import saved

    path = Path(path)
    saved(path, cfg, step)
    write_json(path / "recipe.json", cfg)
    write_json(path / "eval.request.json", {"step": step, "final": final,
        "evaluate": final or step % cfg["eval_steps"] == 0, "created_at": time.time()})


class Checkpoints(TrainerCallback):
    def __init__(self, cfg):
        self.cfg = cfg

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            publish_checkpoint(Path(args.output_dir) / f"checkpoint-{state.global_step}", self.cfg,
                               state.global_step, final=control.should_training_stop)
