"""Shared fixtures: a tiny FLEURS-shaped corpus indexed offline."""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from multilingual_asr.data.corpus import CorpusCatalog
from multilingual_asr.data.evalset import record, select
from multilingual_asr.data.index import build_index
from multilingual_asr.data.schema import FAMILIES
from multilingual_asr.fixtures import tone

LANGUAGES = ("en_us", "hi_in")
SPLIT_ROWS = {"train": 6, "validation": 4, "test": 6}


def _rows(language, split, count):
    return [
        {
            "id": index // 2,  # two recordings share a sentence id, as FLEURS does
            "path": f"/cache/{language}-{split}-{index}.wav",
            "num_samples": 16_000,
            "audio": {"bytes": tone(1.0, 220 + index), "path": "a.wav"},
            "transcription": f"utterance {index} in {language}",
            "raw_transcription": f"Utterance {index} in {language}.",
            "language": language.upper(),
            "lang_id": 0,
            "lang_group_id": 0,
        }
        for index in range(count)
    ]


@pytest.fixture(scope="session")
def corpus_source(tmp_path_factory):
    root = tmp_path_factory.mktemp("fleurs-source")
    for language in LANGUAGES:
        directory = root / "parquet-data" / language
        directory.mkdir(parents=True)
        for split, count in SPLIT_ROWS.items():
            pq.write_table(
                pa.Table.from_pylist(_rows(language, split, count)),
                directory / f"{split}-00000-of-00001.parquet",
            )
    return root


@pytest.fixture(scope="session")
def corpus_index(tmp_path_factory, corpus_source):
    index = tmp_path_factory.mktemp("fleurs-index")
    build_index(index, list(LANGUAGES), source_root=str(corpus_source), workers=2)

    metadata = {
        language: [
            {**row, "split": "test"}
            for row in _rows(language, "test", SPLIT_ROWS["test"])
        ]
        for language in LANGUAGES
    }
    frozen = record(
        select(metadata, list(LANGUAGES), list(FAMILIES), 6),
        name="t",
        languages=list(LANGUAGES),
        families=list(FAMILIES),
        split="test",
        size=6,
        seed=42,
        validated=False,
    )
    cache = tmp_path_factory.mktemp("fleurs-cache")
    catalog = CorpusCatalog(
        index / "manifest.json",
        cache,
        source_root=str(corpus_source),
        evalsets=[frozen],
    )
    catalog.manifest_path = index / "manifest.json"
    catalog.cache_dir = cache
    yield catalog, frozen
    catalog.close()
