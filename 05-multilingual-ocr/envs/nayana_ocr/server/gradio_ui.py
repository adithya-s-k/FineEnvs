"""LaTeX OCR-style playground with independent per-browser task selection."""

import random

import gradio as gr

from ..data.catalog import SPLITS
from ..models import NayanaAction
from .environment import NayanaEnvironment
from .judge import JudgeUnavailable
from .layout import parse_regions

TASK_LABELS = [
    ("Full-page OCR", "page_ocr"),
    ("Section OCR", "section_ocr"),
    ("Multiple-choice VQA", "mcq_vqa"),
    ("Layout detection", "layout_detection"),
    ("Descriptive VQA · Gemma judge", "descriptive_vqa"),
]
LANGUAGE_NAMES = {
    "ar": "Arabic",
    "bn": "Bengali",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "gu": "Gujarati",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "ko": "Korean",
    "ml": "Malayalam",
    "mr": "Marathi",
    "or": "Odia",
    "pa": "Punjabi",
    "ru": "Russian",
    "sa": "Sanskrit",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "zh": "Chinese",
}


class Playground:
    def __init__(self, catalog):
        self.catalog = catalog

    def choose(self, split, language, family, current=None, direction=0, index=None):
        count = self.catalog.group_count(split, language, family)
        if not count:
            return ("", None, "No tasks in this selection.", "", "", "", "", None)
        position = (
            self.catalog.group_position(current, split, language, family)
            if current
            else None
        )
        if index is None:
            index = (
                (position + direction) % count
                if position is not None and direction
                else random.randrange(count)
            )
        if not 0 <= index < count:
            raise gr.Error(f"Choose a task number from 1 to {count:,}.")
        task = self.catalog.group_at(split, language, family, index)
        task = self.catalog.materialize(task)
        preview = self.catalog.image(task)
        progress = f"**Task {index + 1:,} of {count:,}** · `{task['page_id']}`"
        if family == "section_ocr":
            progress += f" · region {task['unit']}"
        if language == "ar":
            progress += (
                "\n\nSome original Arabic pages contain missing or distorted glyphs. "
                "This preview retains the source rendering for inspection."
            )
        if hasattr(self.catalog, "prefetch"):
            upcoming = [
                self.catalog.group_at(
                    split, language, family, (index + offset) % count
                )["task_id"]
                for offset in (1, 2)
            ]
            self.catalog.prefetch(task_ids=upcoming)
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
                (
                    "Accepted by Gemma"
                    if result.metrics["judge_accepted"]
                    else "Rejected by Gemma"
                )
                if "judge_accepted" in result.metrics
                else (
                    "Exact match"
                    if result.metrics.get("exact_match")
                    else "Not an exact match"
                )
            )
            summary = f"### Reward: {result.reward:.3f}\n**{exact}**"
            if "char_error_rate" in result.metrics:
                summary += (
                    f" · Character error rate: {result.metrics['char_error_rate']:.2%}"
                )
            if "mean_f1" in result.metrics:
                summary += " · Class-aware region F1 averaged over IoU 0.50–0.95"
            if result.metrics.get("overlong") or result.metrics.get("invalid_answer"):
                summary += "\nAnswer exceeded the length limit."
            # This demonstration endpoint reveals the reference after grading,
            # like LaTeX OCR. OpenEnv observations/discovery still exclude it.
            return (
                summary,
                self.catalog.get(task_id)["reference"],
                {
                    "reward": result.reward,
                    **result.metrics,
                    "grading_policy_id": result.grading_policy_id,
                },
            )
        except JudgeUnavailable as error:
            raise gr.Error(str(error)) from error
        finally:
            env.close()

    def overlays(self, task_id, answer, reference):
        if not task_id or not reference:
            return gr.update(value=None, visible=False), gr.update(
                value=None, visible=False
            )
        task = self.catalog.get(task_id)
        if task["family"] != "layout_detection":
            return gr.update(value=None, visible=False), gr.update(
                value=None, visible=False
            )
        image = self.catalog.image(task)

        def annotated(text):
            try:
                regions = parse_regions(text, image.width, image.height)
            except (ValueError, TypeError, RecursionError):
                regions = []
            return gr.update(
                value=(image, [(tuple(r["bbox"]), r["label"]) for r in regions]),
                visible=True,
            )

        return annotated(answer), annotated(reference)


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

    def jump(split, language, family, number):
        return playground.choose(split, language, family, index=int(number) - 1)

    def submit(task_id, answer):
        return playground.submit(task_id, answer)

    with gr.Blocks(title="Nayana multilingual OCR", delete_cache=(300, 600)) as demo:
        gr.Markdown(
            "# Nayana multilingual OCR\nRead a page, transcribe a region, detect its layout, or answer a document question."
        )
        gr.Markdown(
            f"**{len(languages)} languages · {page_count:,} pages · {task_count:,} indexed tasks**"
        )
        task_id = gr.State("")
        with gr.Row():
            family = gr.Dropdown(
                TASK_LABELS, value="page_ocr", label="Task", interactive=True
            )
            language = gr.Dropdown(
                [(LANGUAGE_NAMES.get(lang, lang), lang) for lang in languages],
                value="en" if "en" in languages else languages[0],
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
                with gr.Row():
                    number = gr.Number(
                        label="Jump to task (1-based)", value=1, minimum=1, precision=0
                    )
                    jump_button = gr.Button("Go to index")
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
                    placeholder="Enter text, a VQA answer, or layout regions as a JSON array…",
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
        with gr.Row():
            predicted_layout = gr.AnnotatedImage(
                label="Your layout", visible=False, height=480
            )
            reference_layout = gr.AnnotatedImage(
                label="Reference layout · after scoring", visible=False, height=480
            )
        reference.change(
            playground.overlays,
            [task_id, answer, reference],
            [predicted_layout, reference_layout],
            api_name=False,
        )
        inputs = [split, language, family]
        outputs = [task_id, image, progress, prompt, answer, result, reference, metrics]
        random_button.click(choose, inputs, outputs, api_name="choose")
        jump_button.click(jump, [*inputs, number], outputs, api_name="jump")
        previous_button.click(
            previous, [*inputs, task_id], outputs, api_name="previous"
        )
        next_button.click(following, [*inputs, task_id], outputs, api_name="next")
        for control in inputs:
            control.change(choose, inputs, outputs, api_name=False)
        demo.load(choose, inputs, outputs, api_name=False)
        score_button.click(
            submit, [task_id, answer], [result, reference, metrics], api_name="submit"
        ).then(
            playground.overlays,
            [task_id, answer, reference],
            [predicted_layout, reference_layout],
            api_name=False,
        )
        if hasattr(catalog, "stats"):
            with gr.Accordion("Data loading and cache", open=False):
                gr.Markdown(
                    "The whole corpus is indexed. Task selection fetches the needed image row group; "
                    "next tasks are prefetched and cached within byte limits. A cold random jump can "
                    "load about 100 pages' image bytes; training visits tasks in shuffled row-group blocks "
                    "to reuse those downloads. Image bounds are checked when a task is loaded."
                )
                cache_status = gr.JSON(label="Cache usage")
                cache_button = gr.Button("Refresh cache stats")
                cache_button.click(
                    catalog.stats, outputs=cache_status, api_name="cache_stats"
                )
        with gr.Accordion("How these tasks are scored", open=False):
            gr.Markdown(
                "OCR reward combines character similarity (80%) and exact match (20%). "
                "It preserves each script's characters and normalizes whitespace. Multiple-choice VQA requires one uppercase letter. "
                "Layout reward is mean class-aware region F1 over IoU thresholds 0.50–0.95; missed and duplicate regions lower it. "
                "Descriptive VQA uses a strict Gemma judge: all six checks must pass for reward 1. "
                "The judge compares against the corpus answer and can make mistakes; it does not independently verify the image. "
                "Judge outages do not assign a reward.\n\n"
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
