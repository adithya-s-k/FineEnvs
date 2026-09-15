"""Check Gradio's session state, indexed navigation, and scoring over its public API."""

import argparse
import json
import tempfile
import time
from pathlib import Path

from gradio_client import Client
from nayana_ocr.data.corpus import CorpusCatalog


def value(item):
    return item.get("value") if isinstance(item, dict) else item


def verify(url, manifest, output):
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="nayana-gradio-audit-") as temporary:
        catalog = CorpusCatalog(manifest, temporary)
        client = Client(
            url.rstrip("/") + "/web",
            download_files=False,
            httpx_kwargs={"timeout": 240},
        )
        groups = []
        try:
            for lang in catalog.languages:
                for family in ("section_ocr", "mcq_vqa", "page_ocr"):
                    expected = catalog.group_at("train", lang, family, 0)
                    selected = client.predict(
                        "train", lang, family, 1, api_name="/jump"
                    )
                    assert expected["page_id"] in selected[1]
                    assert value(selected[3]) == "" and value(selected[5]) == ""
                    empty = client.predict("", api_name="/submit")
                    assert (
                        "0.000" in empty[0] and value(empty[1]) == expected["reference"]
                    )
                    exact = client.predict(expected["reference"], api_name="/submit")
                    assert "1.000" in exact[0] and exact[2]["exact_match"]
                    groups.append(f"{lang}/{family}")
            count = catalog.group_count("train", "zh", "page_ocr")
            last = catalog.group_at("train", "zh", "page_ocr", count - 1)
            selected = client.predict(
                "train", "zh", "page_ocr", count, api_name="/jump"
            )
            assert last["page_id"] in selected[1] and value(selected[5]) == ""
            exact = client.predict(last["reference"], api_name="/submit")
            assert "1.000" in exact[0]
            wrapped = client.predict("train", "zh", "page_ocr", api_name="/next")
            assert "Task 1 of" in wrapped[1] and value(wrapped[5]) == ""
            other = Client(
                url.rstrip("/") + "/web",
                download_files=False,
                httpx_kwargs={"timeout": 240},
            )
            other.predict("train", "en", "mcq_vqa", 2, api_name="/jump")
            first_zh = catalog.group_at("train", "zh", "page_ocr", 0)
            assert (
                "1.000" in client.predict(first_zh["reference"], api_name="/submit")[0]
            )
            result = {
                "status": "passed",
                "url": url,
                "snapshot_id": catalog.snapshot_id,
                "coverage": groups,
                "last_chinese_page": last["page_id"],
                "last_chinese_source": last["_source_path"],
                "index_wrap_and_reference_clear": True,
                "independent_sessions": True,
                "elapsed_seconds": round(time.monotonic() - started, 3),
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2))
        finally:
            catalog.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.url, args.manifest, args.output)


if __name__ == "__main__":
    main()
