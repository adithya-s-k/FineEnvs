"""What produced a MiMo rollout: the explorer's version and source, the pinned harness, the references the graders
were taken from, and the dataset revision."""

from __future__ import annotations

from functools import lru_cache

from .. import version as _v
from . import config

REFERENCES = {"XiaomiMiMo/verl": "a2ad9f6", "XiaomiMiMo/mimoagent": "467f0a1"}
app_version, source_hash = _v.app_version, _v.source_hash


@lru_cache(maxsize=1)
def dataset_revision() -> str | None:
    try:
        from huggingface_hub import HfApi

        return HfApi().dataset_info(config.DATASET).sha
    except Exception:  # offline: the rollout still records everything else
        return None


def provenance() -> dict:
    from .runner import opencode

    return {**_v.provenance(config.DATASET, dataset_revision()), "harness": {"name": "opencode", "version": opencode.VERSION},
            "reference": dict(REFERENCES)}
