"""Check Gradio's session state, indexed navigation, and scoring over its public API."""

import argparse
import json
import tempfile
import time
from pathlib import Path

from gradio_client import Client
from nayana_ocr.data.corpus import CorpusCatalog
from nayana_ocr.data.schema import FAMILIES


def value(item):
    return item.get("value") if isinstance(item, dict) else item


def verify(url, manifest, output, languages=None, families=None):
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
            languages = languages or catalog.languages
            for lang in languages:
                for family in families or FAMILIES:
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
                    assert "1.000" in exact[0] and (
                        exact[2].get("exact_match") or exact[2].get("judge_accepted")
                    )
                    groups.append(f"{lang}/{family}")
                    print(f"{lang}/{family} passed", flush=True)
            navigation_language = languages[-1]
            count = catalog.group_count("train", navigation_language, "page_ocr")
            last = catalog.group_at("train", navigation_language, "page_ocr", count - 1)
            selected = client.predict(
                "train", navigation_language, "page_ocr", count, api_name="/jump"
            )
            assert last["page_id"] in selected[1] and value(selected[5]) == ""
            exact = client.predict(last["reference"], api_name="/submit")
            assert "1.000" in exact[0]
            wrapped = client.predict(
                "train", navigation_language, "page_ocr", api_name="/next"
            )
            assert "Task 1 of" in wrapped[1] and value(wrapped[5]) == ""
            other = Client(
                url.rstrip("/") + "/web",
                download_files=False,
                httpx_kwargs={"timeout": 240},
            )
            other.predict("train", "en", "mcq_vqa", 2, api_name="/jump")
            first_zh = catalog.group_at("train", navigation_language, "page_ocr", 0)
            assert (
                "1.000" in client.predict(first_zh["reference"], api_name="/submit")[0]
            )
            result = {
                "status": "passed",
                "url": url,
                "snapshot_id": catalog.snapshot_id,
                "coverage": groups,
                "navigation_language": navigation_language,
                "last_page": last["page_id"],
                "last_source": last["_source_path"],
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
    parser.add_argument("--languages", nargs="+")
    parser.add_argument("--families", nargs="+", choices=FAMILIES)
    args = parser.parse_args()
    verify(args.url, args.manifest, args.output, args.languages, args.families)


if __name__ == "__main__":
    main()
