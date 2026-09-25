"""Full-corpus task iteration with row-group locality and bounded prefetch."""

import hashlib
import random
from copy import deepcopy

import requests

from .data.schema import FAMILIES, canonical_json


class CorpusAPI:
    def __init__(self, url, snapshot_id=None):
        self.url = url.rstrip("/")
        self.session = requests.Session()
        response = self.session.get(self.url + "/manifest", timeout=120)
        response.raise_for_status()
        self.manifest = response.json()
        if self.manifest.get("storage") != "bucket-parquet":
            raise ValueError("This server does not expose a full-corpus index")
        self.snapshot_id = self.manifest["snapshot_id"]
        if snapshot_id is not None and snapshot_id != self.snapshot_id:
            raise ValueError("Corpus snapshot changed")

    def _post(self, route, **data):
        response = self.session.post(
            self.url + "/data/" + route,
            json={"snapshot_id": self.snapshot_id, **data},
            timeout=180,
        )
        response.raise_for_status()
        return response.json()

    def blocks(self, split, languages=None, families=None):
        return self._post(
            "blocks", split=split, languages=languages, families=families
        )["blocks"]

    def block_tasks(self, block_id, split, families=None, start=0, limit=512):
        return self._post(
            "block-tasks",
            block_id=block_id,
            split=split,
            families=families,
            start=start,
            limit=limit,
        )["tasks"]

    def prefetch(self, *, block_ids=(), task_ids=()):
        return self._post(
            "prefetch", block_ids=list(block_ids), task_ids=list(task_ids)
        )

    def sample(self, split, languages, families, per_group=4, seed=42):
        return self._post(
            "sample",
            split=split,
            languages=languages,
            families=families,
            per_group=per_group,
            seed=seed,
        )["tasks"]

    def close(self):
        self.session.close()


class BlockTaskStream:
    """One finite epoch. State records consumed IDs, not optimizer checkpoint state.

    Hash-shuffle source blocks, partition them across ranks, then shuffle only a
    bounded chunk of task metadata at a time. Every selected task appears exactly
    once per epoch across ranks. TRL owns completion repetition, not this iterator.
    """

    def __init__(
        self,
        backend,
        split="train",
        languages=None,
        families=None,
        *,
        seed=42,
        epoch=0,
        rank=0,
        world_size=1,
        prefetch_blocks=2,
        chunk_size=128,
    ):
        if (
            not 0 <= rank < world_size
            or not 0 <= prefetch_blocks <= 4
            or not 1 <= chunk_size <= 1000
        ):
            raise ValueError("Invalid rank, prefetch window, or chunk size")
        self.backend = backend
        self.settings = {
            "snapshot_id": backend.manifest["snapshot_id"],
            "split": split,
            "languages": list(languages or backend.manifest["config"]["languages"]),
            "families": list(families or FAMILIES),
            "seed": seed,
            "epoch": epoch,
            "rank": rank,
            "world_size": world_size,
            "chunk_size": chunk_size,
        }
        plan = backend.blocks(
            split, self.settings["languages"], self.settings["families"]
        )
        plan.sort(
            key=lambda block: hashlib.sha256(
                canonical_json([seed, epoch, block["block_id"]]).encode()
            ).digest()
        )
        self.plan = plan[rank::world_size]
        self.plan_id = hashlib.sha256(
            canonical_json([self.settings, self.plan]).encode()
        ).hexdigest()
        self.prefetch_blocks = prefetch_blocks
        self.block_index, self.task_offset = 0, 0

    def state_dict(self):
        return {
            "plan_id": self.plan_id,
            "settings": deepcopy(self.settings),
            "block_index": self.block_index,
            "task_offset": self.task_offset,
        }

    def load_state_dict(self, state):
        if (
            state.get("plan_id") != self.plan_id
            or state.get("settings") != self.settings
        ):
            raise ValueError("Cursor plan, source, seed, or partition changed")
        block, offset = state["block_index"], state["task_offset"]
        if (
            type(block) is not int
            or type(offset) is not int
            or not 0 <= block <= len(self.plan)
            or offset < 0
        ):
            raise ValueError("Invalid cursor position")
        if (block == len(self.plan) and offset) or (
            block < len(self.plan) and offset > self.plan[block]["tasks"]
        ):
            raise ValueError("Cursor exceeds source block")
        self.block_index, self.task_offset = block, offset

    def __iter__(self):
        size = self.settings["chunk_size"]
        while self.block_index < len(self.plan):
            block = self.plan[self.block_index]
            if self.prefetch_blocks:
                self.backend.prefetch(
                    block_ids=[
                        b["block_id"]
                        for b in self.plan[
                            self.block_index : self.block_index + self.prefetch_blocks
                        ]
                    ]
                )
            while self.task_offset < block["tasks"]:
                start = self.task_offset // size * size
                rows = self.backend.block_tasks(
                    block["block_id"],
                    self.settings["split"],
                    self.settings["families"],
                    start,
                    size,
                )
                if not rows:
                    raise RuntimeError("Indexed block unexpectedly lost tasks")
                rng = random.Random(
                    hashlib.sha256(
                        canonical_json(
                            [
                                self.settings["seed"],
                                self.settings["epoch"],
                                block["block_id"],
                                start,
                            ]
                        ).encode()
                    ).hexdigest()
                )
                rng.shuffle(rows)
                for row in rows[self.task_offset - start :]:
                    self.task_offset += 1
                    yield row
            self.block_index += 1
            self.task_offset = 0


def iter_corpus_epochs(
    url, snapshot_id, languages, families, seed=42, prefetch_blocks=2
):
    """Infinite epochs for an explicitly max_steps-bounded TRL training run."""
    backend = CorpusAPI(url, snapshot_id)
    try:
        epoch = 0
        while True:
            stream = BlockTaskStream(
                backend,
                languages=languages,
                families=families,
                seed=seed,
                epoch=epoch,
                prefetch_blocks=prefetch_blocks,
            )
            if not stream.plan:
                raise ValueError("No training tasks in selected groups")
            yield from stream
            epoch += 1
    finally:
        backend.close()


def build_corpus_dataset(
    url, snapshot_id, languages, families, seed=42, prefetch_blocks=2
):
    from datasets import IterableDataset

    return IterableDataset.from_generator(
        iter_corpus_epochs,
        gen_kwargs={
            "url": url,
            "snapshot_id": snapshot_id,
            "languages": tuple(languages),
            "families": tuple(families),
            "seed": seed,
            "prefetch_blocks": prefetch_blocks,
        },
    )
