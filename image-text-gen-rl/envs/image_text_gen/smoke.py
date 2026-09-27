"""Check real HTTP, the Task API, concurrent WebSocket sessions and reward ordering.

Offline by default: synthetic source CSVs and the fixture verifier, so no dataset
download and no inference-provider call. `--url` probes a running server instead
(that path calls its real verifier and costs a few provider requests).
"""

import argparse
import json
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

from .client import connect, encode_image
from .fixtures import png_bytes, render, write_source
from .models import ImageTextGenAction


def _submit(env, task_id, variant, fixture):
    observation = env.reset(task_id=task_id).observation
    image, expected = render(observation.target_text, variant, seed=7, size=(768, 384))
    data = png_bytes(image, [expected, expected] if fixture else None)
    return env.step(ImageTextGenAction(image=encode_image(data)))


def probe(url, fixture=True):
    started = time.monotonic()
    with ExitStack() as stack:
        first = stack.enter_context(connect(url))
        second = stack.enter_context(connect(url))
        manifest = first.manifest()
        assert manifest["splits"], "Server has no splits"
        for split in manifest["splits"]:
            assert first.num_tasks(split) == manifest["counts"][split]
        task = first.get_task_range(manifest["splits"][0], 0, 1)[0]
        a = first.reset(task_id=task["task_id"]).observation
        b = second.reset(task_id=task["task_id"]).observation
        assert a.task_id == b.task_id and a.prompt == b.prompt and not a.done
        assert a.target_text in a.prompt and not a.metrics

        invalid = first.step(ImageTextGenAction(image="not base64!"))
        assert invalid.done and invalid.reward == 0.0
        assert invalid.observation.metrics["invalid_image"] is True

        rewards = {
            variant: _submit(second, task["task_id"], variant, fixture).reward
            for variant in ("clean", "malformed_glyph", "typo_swap", "blank")
        }
        assert rewards["blank"] == 0.0
        if fixture:
            assert rewards["clean"] == 1.0, rewards
            assert rewards["clean"] > rewards["malformed_glyph"] > rewards["blank"], rewards
            assert rewards["clean"] > rewards["typo_swap"], rewards
    return {
        "status": "passed",
        "verifier": manifest["grading"]["verifier"].get("backend"),
        "counts": manifest["counts"],
        "rewards": rewards,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Probe a running server (uses its real verifier)")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.url:
            result = probe(args.url, fixture=False)
            result["source"] = "remote"
        else:
            from .runtime import local_server

            source = write_source(
                stack.enter_context(tempfile.TemporaryDirectory(prefix="itg-smoke-"))
            )
            url = stack.enter_context(local_server(source, fixture_verifier=True))
            result = probe(url, fixture=True)
            result["source"] = "synthetic-fixture"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
