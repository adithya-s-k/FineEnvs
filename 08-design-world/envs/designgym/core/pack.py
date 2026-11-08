"""The task pack: prepared tasks on disk, and the catalog the server reads them from.

Layout:

    <pack>/
      manifest.json                 version, source, split counts
      NOTICE.md                     the data licence
      fonts/manifest.json + files   the font families the designs use
      tasks/<split>.jsonl           one task per line (ground truth included)
      designs/<design_id>/          assets/<aid>.png|webp and preview.jpg

A task is one design template. How it is played is chosen at reset: `mode`
(reference image or text description) and `surface` (tools, HTML or GUI) are
episode parameters, not separate tasks, so every split serves every variant and
the Task API stays a plain list of designs.

DesignGym's pack has the splits `sft` (2k designs), `rl` (2k more, disjoint)
and `test` (the frozen DesignWorldBench set). Download it from the Hub with
`designgym pull` (or point DESIGN_CANVAS_PACK at a pack directory).
"""

from __future__ import annotations

import json
import os
import threading
from collections import OrderedDict
from pathlib import Path

from .doc import DesignDoc
from .fonts import FontRegistry
from .render import Renderer

PACK_VERSION = 2  # 2: tasks name their design by `design_id`
# sft / rl / test are DesignGym's pack; train / validation are the splits of earlier and local packs
SPLITS = ("sft", "rl", "train", "validation", "test")
DEFAULT_PACK = Path(os.environ.get("DESIGN_CANVAS_PACK", Path.home() / ".cache/designgym/pack"))
DEFAULT_PACK_REPO = os.environ.get("DESIGNGYM_PACK_REPO", "FineEnvs/designgym-tasks")


def difficulty(n: int) -> str:
    return "easy" if n <= 6 else "medium" if n <= 12 else "hard"


class TaskPack:
    """Read-only view of a prepared pack. Safe to share across sessions and threads."""

    def __init__(self, root: Path | str):
        self.root = Path(root)
        manifest = self.root / "manifest.json"
        if not manifest.exists():
            raise FileNotFoundError(
                f"no task pack at {self.root}. Download one with `designgym pull` "
                f"or point DESIGN_CANVAS_PACK at an existing pack."
            )
        self.manifest = json.loads(manifest.read_text())
        self.fonts = FontRegistry(self.root / "fonts")
        self.renderer = Renderer(self.fonts)
        self._tasks: dict[str, list[dict]] = {}
        self._by_id: dict[str, dict] = {}
        self._docs: dict[str, DesignDoc] = {}
        self._renders: OrderedDict[tuple, object] = OrderedDict()
        self._lock = threading.Lock()
        for split in self.splits():
            path = self.root / "tasks" / f"{split}.jsonl"
            tasks = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self._tasks[split] = tasks
            for t in tasks:
                self._by_id[t["task_id"]] = t

    # ------------------------------------------------------------ index

    def splits(self) -> list[str]:
        return [s for s in SPLITS if s in self.manifest.get("splits", {})]

    def count(self, split: str) -> int:
        if split not in self._tasks:
            raise KeyError(f"unknown split {split!r}; have {', '.join(self.splits())}")
        return len(self._tasks[split])

    def at(self, split: str, index: int) -> dict:
        tasks = self._tasks.get(split)
        if tasks is None:
            raise KeyError(f"unknown split {split!r}; have {', '.join(self.splits())}")
        if not 0 <= index < len(tasks):
            raise IndexError(f"{split} has {len(tasks)} tasks; index {index} is out of range")
        return tasks[index]

    def get(self, task_id: str) -> dict:
        try:
            return self._by_id[task_id]
        except KeyError:
            raise KeyError(f"unknown task_id {task_id!r}") from None

    @staticmethod
    def public(task: dict) -> dict:
        """The Task API view: enough to choose a task, nothing that gives it away."""
        return {
            "task_id": task["task_id"],
            "index": task["index"],
            "split": task["split"],
            "design_id": task["design_id"],
            "title": task["title"],
            "format": task["format"],
            "category": task["category"],
            "canvas": task["canvas"],
            "n_elements": task["stats"]["n_elements"],
            "n_text": task["stats"]["n_text"],
            "n_assets": len(task["assets"]),
            "difficulty": task["difficulty"],
            "modes": ["reference", "description"],
            "surfaces": ["tools", "html", "gui"],
        }

    def task_range(self, split: str, start=None, stop=None) -> list[dict]:
        tasks = self._tasks.get(split)
        if tasks is None:
            raise KeyError(f"unknown split {split!r}")
        return [self.public(t) for t in tasks[slice(start, stop)]]

    # --------------------------------------------------------- per task

    def gt_doc(self, task: dict) -> DesignDoc:
        tid = task["task_id"]
        with self._lock:
            doc = self._docs.get(tid)
            if doc is None:
                doc = DesignDoc.model_validate(task["doc"])
                self._docs[tid] = doc
        return doc

    def asset_paths(self, task: dict) -> dict[str, Path]:
        return {a["id"]: self.root / a["file"] for a in task["assets"]}

    def asset_ctx(self, task: dict) -> dict[str, dict]:
        return {a["id"]: {"aspect": a["aspect"], "kind": a["kind"]} for a in task["assets"]}

    def preview_path(self, task: dict) -> Path:
        return self.root / task["preview"]

    def _cached(self, key: tuple, make):
        with self._lock:
            if key in self._renders:
                self._renders.move_to_end(key)
                return self._renders[key]
        value = make()
        with self._lock:
            self._renders[key] = value
            while len(self._renders) > 256:
                self._renders.popitem(last=False)
        return value

    def target_png(self, task: dict, max_size: int = 512) -> bytes:
        return self._cached(
            ("png", task["task_id"], max_size),
            lambda: self.renderer.render_png(self.gt_doc(task), self.asset_paths(task), max_size),
        )

    def target_jpeg(self, task: dict, max_size: int = 768) -> bytes:
        return self._cached(
            ("jpg", task["task_id"], max_size),
            lambda: self.renderer.render_jpeg(self.gt_doc(task), self.asset_paths(task), max_size),
        )

    def target_pixels(self, task: dict, max_size: int):
        return self._cached(
            ("arr", task["task_id"], max_size),
            lambda: self.renderer.render_array(self.gt_doc(task), self.asset_paths(task), max_size),
        )

    def asset_similarity(self, task: dict):
        """`sim(a, b)` in [0, 1]: how alike two of this task's assets look.

        Compared on 32 px thumbnails composited over grey (so the cut-out shape
        counts), on average colour, and on aspect ratio. Squared, so only
        genuine look-alikes earn real credit.
        """

        def make():
            import numpy as np
            from PIL import Image

            from .score import ssim

            thumbs, colors, aspect = {}, {}, {}
            for a in task["assets"]:
                with Image.open(self.root / a["file"]) as im:
                    im = im.convert("RGBA")
                    bg = Image.new("RGBA", im.size, (128, 128, 128, 255))
                    flat = Image.alpha_composite(bg, im).convert("RGB")
                thumbs[a["id"]] = np.asarray(flat.resize((32, 32), Image.BOX), dtype=np.float64)
                colors[a["id"]] = np.asarray(flat.resize((4, 4), Image.BOX), dtype=np.float64)
                aspect[a["id"]] = a["aspect"]
            table = {}
            ids = list(thumbs)
            for i, x in enumerate(ids):
                for y in ids[i + 1:]:
                    shape = ssim(thumbs[x], thumbs[y])
                    color = float(np.exp(-np.abs(colors[x] - colors[y]).mean() / 255 / 0.1))
                    ratio = float(np.exp(-abs(np.log(aspect[x] / aspect[y])) / 0.25))
                    sim = (0.4 * shape + 0.3 * color + 0.3 * ratio) ** 2
                    table[(x, y)] = table[(y, x)] = round(sim, 4)
            return table

        table = self._cached(("assetsim", task["task_id"]), make)
        return lambda a, b: 1.0 if a == b else table.get((a, b), 0.0)

    def grade(self, task: dict, doc: DesignDoc, mode: str = "reference", version: int | None = None,
              assets: dict | None = None):
        """The one scoring entry point: episodes, tests and scripts all call this. `assets`: the episode's images
        when it has more than the task's own (a workspace's decoys), so an inserted decoy renders and costs score."""
        from .score import CONFIGS, grade

        size = CONFIGS[mode].pixel_size
        gt = self.gt_doc(task)
        # A real app's read-back canvas can differ from the target's (a resized board, a rounded page size):
        # the attempt is rendered on the target's canvas, so the pixels compare like for like.
        canvas = doc if (doc.width, doc.height) == (gt.width, gt.height) else doc.model_copy(
            update={"width": gt.width, "height": gt.height})
        attempt = self.renderer.render_array(canvas, {**self.asset_paths(task), **(assets or {})}, size)
        return grade(doc, gt, mode, attempt, self.target_pixels(task, size),
                     self.blank_pixels(task, size), fonts=self.fonts,
                     asset_sim=self.asset_similarity(task),
                     version=version or int(os.environ.get("DESIGN_CANVAS_REWARD", "1")))

    def blank_pixels(self, task: dict, max_size: int):
        def make():
            doc = self.gt_doc(task)
            return self.renderer.render_array(
                DesignDoc(width=doc.width, height=doc.height), {}, max_size
            )

        return self._cached(("blank", task["task_id"], max_size), make)


_PACKS: dict[str, TaskPack] = {}
_PACKS_LOCK = threading.Lock()


def load_pack(root: Path | str | None = None) -> TaskPack:
    """The task catalog the server plays from: a pack directory (`root`, `DESIGN_CANVAS_PACK`, or the default)."""
    key = str(Path(root or DEFAULT_PACK).expanduser().resolve())
    with _PACKS_LOCK:
        pack = _PACKS.get(key)
        if pack is None:
            pack = TaskPack(key)
            _PACKS[key] = pack
        return pack


def pull_pack(out: Path | str | None = None, repo: str = DEFAULT_PACK_REPO, revision: str | None = None,
              splits: list[str] | None = None) -> Path:
    """Download a task pack from a Hub dataset repo into `out` (default: the default pack directory).

    With `splits`, only those splits' tasks and designs are fetched (the fonts always are)."""
    from huggingface_hub import snapshot_download

    out = Path(out or DEFAULT_PACK).expanduser()
    patterns = None
    if splits:
        snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=out,
                          allow_patterns=["manifest.json", "NOTICE.md", "fonts/**",
                                          *[f"tasks/{s}.jsonl" for s in splits]])
        ids = {json.loads(line)["design_id"] for s in splits
               for line in (out / "tasks" / f"{s}.jsonl").read_text().splitlines() if line.strip()}
        patterns = [f"designs/{i}/**" for i in sorted(ids)]
    snapshot_download(repo, repo_type="dataset", revision=revision, local_dir=out, allow_patterns=patterns)
    return out
