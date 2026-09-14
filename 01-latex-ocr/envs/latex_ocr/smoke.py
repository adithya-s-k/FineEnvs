"""Exercise Task API + WebSocket episodes. Default: a tiny offline fixture."""

import argparse
import base64
import io
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw

from . import LatexOCRAction, LatexOCREnv
from .runtime import local_server


def make_fixture(directory):
    from datasets import Dataset

    rows = []
    for text in ("x^2 + 1", "y = 3"):
        image = Image.new("RGB", (160, 48), "white")
        ImageDraw.Draw(image).text((8, 12), text, fill="black")
        rows.append({"image": image, "text": text})
    for split in ("train", "test"):
        Dataset.from_list(rows).to_parquet(str(Path(directory) / f"{split}.parquet"))


def probe(url, mode="materialize"):
    kwargs = {"base_url": url, "connect_timeout_s": 60, "message_timeout_s": 180}
    with LatexOCREnv(**kwargs) as first, LatexOCREnv(**kwargs) as second:
        splits = first.list_splits()
        assert "train" in splits and "test" in splits, splits
        count = first.num_tasks("test")
        assert count >= 2, f"Smoke test needs two test tasks, got {count}"
        tasks = first.get_task_range("test", 0, 2)
        assert len(tasks) == 2 and first.get_task("test", 0)["index"] == 0
        reset = {"split": "test"}
        if mode == "materialize":
            reset["index"] = 0
        observation = first.reset(**reset).observation
        assert not observation.done and not observation.target_latex
        assert observation.prompt and observation.image_base64
        image = Image.open(io.BytesIO(base64.b64decode(observation.image_base64)))
        assert image.format == "PNG" and min(image.size) > 0
        assert (
            second.reset(**reset).observation.image_base64 == observation.image_base64
        )
        # The first session reveals the answer only after a terminal action;
        # grading it in the second session checks independent episode state.
        empty = first.step(LatexOCRAction(latex=""))
        assert empty.done and empty.reward == 0.0
        target = empty.observation.target_latex
        assert target
        exact = second.step(LatexOCRAction(latex=target))
        assert exact.done and exact.reward == 1.0 and exact.observation.exact_match
        assert exact.observation.task_id == observation.task_id
        assert exact.metadata and "char_error_rate" in exact.metadata
        scores = {"empty": empty.reward, "exact": exact.reward}
        if mode == "materialize":
            for name, prediction in (
                ("partial", target + "z"),
                ("padded", target + " " * 10000),
            ):
                first.reset(**reset)
                scores[name] = first.step(LatexOCRAction(latex=prediction)).reward
            assert 0 < scores["partial"] < 1
            assert scores["padded"] == 0
            assert (
                first.reset(split="test", index=1).observation.task_id
                != observation.task_id
            )
        else:
            next_obs = first.reset(split="test").observation
            assert next_obs.index == observation.index + 1
            assert next_obs.task_id != observation.task_id
            assert first.reset(split="test").observation.exhausted
            assert first.reset(split="test").observation.index == observation.index
    return {"mode": mode, "splits": splits, "test_tasks": count, "scores": scores}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Check an existing materialize-mode server")
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use two unsloth/LaTeX_OCR rows per split",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.url:
        results = [probe(args.url)]
    else:
        with tempfile.TemporaryDirectory(prefix="latex-ocr-fixture-") as directory:
            if not args.real_data:
                make_fixture(directory)
            results = []
            for mode in ("materialize", "stream"):
                with local_server(
                    mode=mode, max_rows=2, dataset=None if args.real_data else directory
                ) as url:
                    results.append(probe(url, mode))
    summary = {
        "status": "passed",
        "real_data": bool(args.real_data or args.url),
        "source": args.url
        or ("unsloth/LaTeX_OCR" if args.real_data else "offline-fixture"),
        "results": results,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
