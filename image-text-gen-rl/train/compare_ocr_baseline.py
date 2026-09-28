# /// script
# requires-python = ">=3.11,<3.12"
# dependencies = [
#   "paddlepaddle==2.6.2",
#   "paddleocr==2.9.1",
#   "easyocr==1.7.2",
#   "rapidfuzz>=3.14,<4",
#   "numpy<2",
#   "pillow>=11,<13",
#   "setuptools<70",
# ]
# ///
"""Is a conventional OCR model enough for the reward? Compare it with Gemma 4 + Qwen3.6.

Reads the 280 labelled calibration renders (artifacts/calibration-40.json: real test
targets x clean / swapped / missing / doubled letter / wrong case / broken glyph / gibberish)
with two conventional OCR models, PaddleOCR 2.9.1 and EasyOCR 1.7.2, and scores each reading
two ways: the simple reward (1 - edit distance to the target) and this environment's scoring.
Next to them: the environment's reward from the saved Gemma 4 31B + Qwen3.6-35B-A3B readings,
and the reward the literal truth deserves.

Three phases, because the renders need the host's fonts and paddlepaddle 2.6.2 needs Linux
(it segfaults on Apple Silicon):

  # 1. host: rebuild the images, upload to the bucket
  ../../launch image-text-gen-rl --exec python image-text-gen-rl/train/compare_ocr_baseline.py render
  hf buckets sync image-text-gen-rl/artifacts/ocr-baseline hf://buckets/AdithyaSK/image-text-gen-rl/ocr-baseline
  # 2. HF Jobs (Linux, Flow-GRPO's pins): OCR every image into the bucket
  hf jobs uv run --flavor cpu-upgrade -v hf://buckets/AdithyaSK/image-text-gen-rl:/bucket \
      image-text-gen-rl/train/compare_ocr_baseline.py ocr --dir /bucket/ocr-baseline
  # 3. host: pull the readings, score and summarise
  hf buckets sync hf://buckets/AdithyaSK/image-text-gen-rl/ocr-baseline image-text-gen-rl/artifacts/ocr-baseline
  ../../launch image-text-gen-rl --exec python image-text-gen-rl/train/compare_ocr_baseline.py report
"""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DEFAULT_DIR = HERE / "artifacts" / "ocr-baseline"
PAIR = ("google/gemma-4-31B-it:deepinfra", "Qwen/Qwen3.6-35B-A3B:deepinfra")


def render_phase(args):
    sys.path.insert(0, str(HERE / "envs"))
    from image_text_gen.fixtures import render
    from image_text_gen.server.scoring import normalise

    calibration = json.loads(args.calibration.read_text())
    (args.dir / "images").mkdir(parents=True, exist_ok=True)
    order = list(dict.fromkeys(item["task_id"] for item in calibration["readings"]))
    manifest = []
    for item in calibration["readings"]:
        n = order.index(item["task_id"])
        image, expected = render(item["target"], item["variant"], seed=args.seed + n,
                                 size=(1024, 640), font_path=item["font"])
        if normalise(expected) != normalise(item["expected"]):
            raise SystemExit(f"calibration render drifted for {item['task_id']} {item['variant']}")
        name = f"{n:02d}-{item['variant']}.png"
        image.save(args.dir / "images" / name)
        manifest.append({**{k: item[k] for k in ("task_id", "variant", "target", "prompt", "expected")},
                         "image": name,
                         "env_texts": [item["by_model"][m]["text"] for m in PAIR]})
    (args.dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print(f"rendered {len(manifest)} images into {args.dir}")


def ocr_phase(args):
    import easyocr
    import numpy as np
    from paddleocr import PaddleOCR
    from PIL import Image

    # Default English PaddleOCR on CPU (the configuration Flow-GRPO also uses).
    paddle = PaddleOCR(use_angle_cls=False, lang="en", use_gpu=False, show_log=False)
    easy = easyocr.Reader(["en"], gpu=False, verbose=False)
    manifest = json.loads((args.dir / "manifest.json").read_text())
    readings = {}
    for n, row in enumerate(manifest):
        array = np.array(Image.open(args.dir / "images" / row["image"]).convert("RGB"))
        result = paddle.ocr(array, cls=False)
        paddle_lines = [r[1][0] for r in result[0] if r[1][1] > 0] if result and result[0] else []
        easy_lines = [text for _, text, conf in easy.readtext(array) if conf > 0]
        readings[row["image"]] = {"paddleocr": paddle_lines, "easyocr": easy_lines}
        print(f"{n + 1}/{len(manifest)} {row['image']}: paddle={paddle_lines} easy={easy_lines}", flush=True)
    (args.dir / "ocr_readings.json").write_text(json.dumps(readings, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {args.dir / 'ocr_readings.json'}")


def edit_distance_reward(text, target):
    """The simple OCR reward: 1 - Levenshtein(OCR output, target) / len(target), case- and
    whitespace-insensitive, clipped at 0. Extra text counts as insertions."""
    from rapidfuzz.distance.Levenshtein import distance

    def norm(value):
        return " ".join(value.lower().split())

    return max(0.0, 1 - distance(norm(text), norm(target)) / max(1, len(norm(target))))


def report_phase(args):
    sys.path.insert(0, str(HERE / "envs"))
    from image_text_gen.server.scoring import MALFORMED, combine, normalise

    manifest = json.loads((args.dir / "manifest.json").read_text())
    readings = json.loads((args.dir / "ocr_readings.json").read_text())
    rows = []
    for row in manifest:
        truth = row["expected"]
        # A perfect conventional OCR has no symbol for a broken glyph, so the literal truth
        # for "literal exact" comparisons writes it as a wrong character, never the letter.
        truth_ocr = truth.replace(MALFORMED, "#")
        out = {**{k: row[k] for k in ("task_id", "variant", "target", "expected", "image")},
               "truth_reward": combine(row["target"], [truth], row["prompt"])[0]}
        env_reward, env_metrics, _ = combine(row["target"], row["env_texts"], row["prompt"])
        out.update(env_reward=env_reward, env_texts=row["env_texts"],
                   env_malformed=env_metrics["malformed_glyphs"], env_extra=env_metrics["extra_chars"])
        for engine in ("paddleocr", "easyocr"):
            lines = readings[row["image"]][engine]
            text = "\n".join(lines)
            out[f"{engine}_lines"] = lines
            out[f"{engine}_literal"] = normalise(text) == normalise(truth_ocr)
            # A defect read back as the clean target: the OCR "repaired" it.
            out[f"{engine}_read_as_target"] = normalise(text).lower() == normalise(row["target"]).lower()
            out[f"{engine}_edit_reward"] = round(edit_distance_reward(text, row["target"]), 4)
            # The same OCR reading through this environment's scoring, isolating the reader.
            out[f"{engine}_env_reward"] = combine(row["target"], [text], row["prompt"])[0]
        rows.append(out)

    def mean(values):
        return round(statistics.fmean(values), 3)

    def full(values):
        return round(sum(v >= 0.999 for v in values) / len(values), 3)

    by_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    summary = {}
    for variant, group in by_variant.items():
        entry = {"images": len(group), "truth_reward": mean(r["truth_reward"] for r in group),
                 "env_mean": mean(r["env_reward"] for r in group),
                 "env_full": full([r["env_reward"] for r in group])}
        for engine in ("paddleocr", "easyocr"):
            entry[f"{engine}_literal"] = mean(r[f"{engine}_literal"] for r in group)
            entry[f"{engine}_read_as_target"] = mean(r[f"{engine}_read_as_target"] for r in group)
            entry[f"{engine}_edit_mean"] = mean(r[f"{engine}_edit_reward"] for r in group)
            entry[f"{engine}_edit_full"] = full([r[f"{engine}_edit_reward"] for r in group])
        summary[variant] = entry
    defects = [r for r in rows if r["variant"] != "clean"]
    clean = [r for r in rows if r["variant"] == "clean"]
    readers = {"paddleocr + edit distance": "paddleocr_edit_reward",
               "easyocr + edit distance": "easyocr_edit_reward",
               "paddleocr + env scoring": "paddleocr_env_reward",
               "easyocr + env scoring": "easyocr_env_reward",
               "gemma + qwen (env)": "env_reward"}
    totals = {
        "images": len(rows), "defect_images": len(defects),
        "defects_given_full_reward": {k: sum(r[v] >= 0.999 for r in defects) for k, v in readers.items()},
        "clean_mean_reward": {k: mean(r[v] for r in clean) for k, v in readers.items()},
        "mae_vs_literal_truth": {k: mean(abs(r[v] - r["truth_reward"]) for r in rows) for k, v in readers.items()},
        "clean_ranked_above_its_defect": {},
        "literal_exact": {"paddleocr": mean(r["paddleocr_literal"] for r in rows),
                          "easyocr": mean(r["easyocr_literal"] for r in rows)},
    }
    by_task = defaultdict(dict)
    for row in rows:
        by_task[row["task_id"]][row["variant"]] = row
    for name, key in readers.items():
        pairs = [v["clean"][key] > d[key] for v in by_task.values() for k, d in v.items() if k != "clean"]
        totals["clean_ranked_above_its_defect"][name] = mean(pairs)
    report = {"summary": summary, "totals": totals, "rows": rows}
    (args.dir / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n")
    print(json.dumps({"summary": summary, "totals": totals}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("phase", choices=["render", "ocr", "report"])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--calibration", type=Path, default=HERE / "artifacts/calibration-40.json")
    parser.add_argument("--seed", type=int, default=1, help="seed the calibration was run with")
    args = parser.parse_args()
    {"render": render_phase, "ocr": ocr_phase, "report": report_phase}[args.phase](args)


if __name__ == "__main__":
    main()
