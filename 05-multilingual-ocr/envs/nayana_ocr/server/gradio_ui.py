"""LaTeX OCR-style playground with independent per-browser task selection."""

import random
import re

import gradio as gr
from PIL import Image

from ..data.catalog import SPLITS
from ..models import NayanaAction
from .environment import NayanaEnvironment

TASK_LABELS = [
    ("Full-page OCR", "page_ocr"),
    ("Section OCR", "section_ocr"),
    ("Multiple-choice VQA", "mcq_vqa"),
]
LANGUAGE_NAMES = {"en": "English", "kn": "Kannada", "hi": "Hindi", "ar": "Arabic"}


class Playground:
    def __init__(self, catalog):
        self.catalog = catalog

    def candidates(self, split, language, family):
        with self.catalog._connect() as db:
            rows = db.execute(
                """SELECT id,json_extract(payload,'$.page_id'),json_extract(payload,'$.unit')
                FROM tasks WHERE split=? AND json_extract(payload,'$.language')=?
                AND json_extract(payload,'$.family')=?""",
                (split, language, family),
            ).fetchall()
        return [
            row[0]
            for row in sorted(
                rows,
                key=lambda row: (
                    tuple(int(n) for n in re.findall(r"\d+", row[1])),
                    row[2],
                    row[0],
                ),
            )
        ]

    def choose(self, split, language, family, current=None, direction=0):
        candidates = self.candidates(split, language, family)
        if not candidates:
            return (
                "",
                None,
                "No tasks in this selection. Try another split or task.",
                "",
                "",
                "",
                "",
                None,
            )
        index = (
            (candidates.index(current) + direction) % len(candidates)
            if current in candidates and direction
            else random.randrange(len(candidates))
        )
        task = self.catalog.get(candidates[index])
        with Image.open(self.catalog.asset(task["asset_sha256"])[0]) as image:
            preview = image.copy()
        progress = f"**Task {index + 1} of {len(candidates)}** · `{task['page_id']}`"
        if family == "section_ocr":
            progress += f" · region {task['unit']}"
        if language == "ar":
            progress += (
                "\n\nSome original Arabic pages contain missing or distorted glyphs. "
                "This preview retains the source rendering for inspection."
            )
        return (
            task["task_id"],
            preview,
            progress,
            task["prompt"],
            gr.update(value="", rtl=language == "ar"),
            "",
            gr.update(value="", rtl=language == "ar"),
            None,
        )

    def submit(self, task_id, answer):
        if not task_id:
            raise gr.Error("Load a task first.")
        env = NayanaEnvironment(self.catalog)
        try:
            env.reset(task_id=task_id)
            result = env.step(NayanaAction(answer=answer))
            exact = (
                "Exact match" if result.metrics["exact_match"] else "Not an exact match"
            )
            summary = f"### Reward: {result.reward:.3f}\n**{exact}**"
            if "char_error_rate" in result.metrics:
                summary += (
                    f" · Character error rate: {result.metrics['char_error_rate']:.2%}"
                )
            if result.metrics["overlong"]:
                summary += "\nAnswer exceeded the length limit."
            # This demonstration endpoint reveals the reference after grading,
            # like LaTeX OCR. OpenEnv observations/discovery still exclude it.
            return (
                summary,
                self.catalog.get(task_id)["reference"],
                {"reward": result.reward, **result.metrics},
            )
        finally:
            env.close()


def build_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    catalog = web_manager.env.catalog
    playground = Playground(catalog)
    languages = catalog.manifest["config"]["languages"]
    splits = [split for split in SPLITS if catalog.count(split)]
    page_count = sum(catalog.manifest["pages"].values())
    task_count = sum(item["tasks"] for item in catalog.manifest["counts"])

    def choose(split, language, family):
        return playground.choose(split, language, family)

    def previous(split, language, family, current):
        return playground.choose(split, language, family, current, -1)

    def following(split, language, family, current):
        return playground.choose(split, language, family, current, 1)

    def submit(task_id, answer):
        return playground.submit(task_id, answer)

    with gr.Blocks(title="Nayana multilingual OCR") as demo:
        gr.Markdown(
            "# Nayana multilingual OCR\nRead a whole page, transcribe a region, or answer a question about a document."
        )
        gr.Markdown(
            f"**{len(languages)} languages · {page_count} pages · {task_count} tasks** in this preview"
        )
        task_id = gr.State("")
        with gr.Row():
            family = gr.Dropdown(
                TASK_LABELS, value="page_ocr", label="Task", interactive=True
            )
            language = gr.Dropdown(
                [(LANGUAGE_NAMES.get(lang, lang), lang) for lang in languages],
                value=languages[0],
                label="Language",
                interactive=True,
            )
            split = gr.Dropdown(
                splits, value=splits[0], label="Split", interactive=True
            )
        with gr.Row():
            with gr.Column(scale=6):
                with gr.Row():
                    previous_button = gr.Button("← Previous")
                    next_button = gr.Button("Next →")
                    random_button = gr.Button("Shuffle task", variant="primary")
                progress = gr.Markdown("")
                image = gr.Image(
                    type="pil",
                    format="png",
                    label="Document",
                    height=560,
                    interactive=False,
                )
                prompt = gr.Textbox(label="Instructions", interactive=False, lines=3)
            with gr.Column(scale=5):
                answer = gr.Textbox(
                    label="Your answer",
                    placeholder="Type the transcription, or the option letter for VQA…",
                    lines=12,
                )
                score_button = gr.Button("Score answer", variant="primary")
                result = gr.Markdown("")
                reference = gr.Textbox(
                    label="Reference · revealed after scoring",
                    lines=8,
                    max_lines=16,
                    interactive=False,
                )
                with gr.Accordion("Scoring details", open=False):
                    metrics = gr.JSON(label="Metrics")
        inputs = [split, language, family]
        outputs = [task_id, image, progress, prompt, answer, result, reference, metrics]
        random_button.click(choose, inputs, outputs, api_name="choose")
        previous_button.click(
            previous, [*inputs, task_id], outputs, api_name="previous"
        )
        next_button.click(following, [*inputs, task_id], outputs, api_name="next")
        for control in inputs:
            control.change(choose, inputs, outputs, api_name=False)
        demo.load(choose, inputs, outputs, api_name=False)
        score_button.click(
            submit, [task_id, answer], [result, reference, metrics], api_name="submit"
        )
        with gr.Accordion("How these tasks are scored", open=False):
            gr.Markdown(
                "OCR reward combines character similarity (80%) and exact match (20%). "
                "It preserves each script's characters and normalizes whitespace. VQA requires one uppercase option letter.\n\n"
                "Full-page images preserve page size and layout but mask areas without text annotations. "
                "VQA uses the original page. Full-page references join the corpus's text regions using geometric column and reading order, "
                "with right-to-left columns for Arabic. This is annotation-based OCR, not table-format reconstruction. "
                "Pages with incomplete or overlapping text-region annotations are excluded from full-page OCR. "
                "The reference is revealed only after you score your answer in this playground."
            )
        gr.Markdown(
            "[Nayana corpus · CognitiveLab](https://huggingface.co/datasets/Cognitive-Lab/NayanaOCR_Corpus_2025) "
            "· CC BY-NC 4.0 · [Source and reproduction](https://github.com/adithya-s-k/HuggingEnvs/tree/codex/multilingual-ocr/05-multilingual-ocr)"
        )
    return demo
