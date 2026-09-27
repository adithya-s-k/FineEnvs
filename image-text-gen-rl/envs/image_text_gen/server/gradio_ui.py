"""Playground: pick a prompt, upload (or synthesise) an image, see every verifier's reading."""

import random

import gradio as gr

from ..client import encode_image
from ..fixtures import VARIANTS, png_bytes, render
from ..models import ImageTextGenAction
from .environment import ImageTextGenEnvironment
from .verifier import VerifierUnavailable


class Playground:
    def __init__(self, catalog):
        self.catalog = catalog

    def choose(self, split, current=None, direction=0):
        count = self.catalog.count(split)
        index = random.randrange(count)
        if current and direction:
            task = self.catalog.get(current)
            if task["split"] == split:
                rows = self.catalog.task_range(split, 0, min(count, 1000))
                ids = [row["task_id"] for row in rows]
                if current in ids:
                    index = (ids.index(current) + direction) % len(ids)
        task = self.catalog.at(split, index)
        summary = (
            f"**Task {index + 1:,} of {count:,}** · `{task['task_id']}` · "
            f"{task['text_len']} characters · SD3 baseline OCR {task['baseline_ocr']:.2f}"
        )
        return task["task_id"], summary, task["prompt"], task["target_text"], None, "", [], None

    def synthesise(self, task_id, variant):
        if not task_id:
            raise gr.Error("Load a task first.")
        image, _ = render(self.catalog.get(task_id)["target_text"], variant, seed=random.randrange(10**6))
        return image

    def submit(self, task_id, image):
        if not task_id:
            raise gr.Error("Load a task first.")
        if image is None:
            raise gr.Error("Upload or synthesise an image first.")
        env = ImageTextGenEnvironment(self.catalog)
        try:
            env.reset(task_id=task_id)
            result = env.step(ImageTextGenAction(image=encode_image(png_bytes(image))))
        except VerifierUnavailable as error:
            raise gr.Error(str(error)) from error
        finally:
            env.close()
        m = result.metrics
        if m.get("invalid_image"):
            return f"### Reward: 0.000\nInvalid image: {result.metadata.get('error')}", [], m
        verdict = "Exact match" if m["exact_match"] else "Not an exact match"
        summary = (
            f"### Reward: {result.reward:.3f}\n**{verdict}** · text accuracy "
            f"{m['text_accuracy']:.1%} · malformed glyphs {m['malformed_glyphs']} · "
            f"extra characters {m['extra_chars']} · case {'matches' if m['case_match'] else 'differs'}"
        )
        rows = [
            [t.get("model", ""), t.get("text", ""), f"{t.get('text_accuracy', 0):.1%}",
             t.get("malformed_glyphs", 0), t.get("extra_chars", 0)]
            for t in result.transcriptions
        ]
        return summary, rows, {**m, "grading_policy_id": result.grading_policy_id}


def build_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    catalog = web_manager.env.catalog
    playground = Playground(catalog)
    splits = catalog.splits()
    with gr.Blocks(title="Image text generation RL", delete_cache=(300, 600)) as demo:
        gr.Markdown(
            "# Image text generation RL\nRender the quoted text from the prompt. Each verifier model "
            "transcribes the image **without seeing the target**; the reward compares its literal "
            "reading with the target and penalises malformed glyphs and spurious text."
        )
        counts = " · ".join(f"{s} {catalog.count(s):,}" for s in splits)
        gr.Markdown(f"**{counts} tasks** from `{catalog.manifest['source']}`")
        task_id = gr.State("")
        with gr.Row():
            split = gr.Dropdown(splits, value=splits[0], label="Split", interactive=True)
            previous_button = gr.Button("← Previous")
            next_button = gr.Button("Next →")
            shuffle_button = gr.Button("Shuffle task", variant="primary")
        progress = gr.Markdown("")
        with gr.Row():
            with gr.Column(scale=5):
                prompt = gr.Textbox(label="Prompt", lines=4, interactive=False)
                target = gr.Textbox(label="Target text", interactive=False)
                image = gr.Image(type="pil", label="Generated image", height=420)
                with gr.Row():
                    variant = gr.Dropdown(list(VARIANTS), value="clean", label="Synthetic variant")
                    synth_button = gr.Button("Synthesise test image")
                gr.Markdown(
                    "> ℹ️ **About synthetic images:** no image-generation model is used here. "
                    "The target text is drawn with Pillow in a system font on a flat background, and "
                    "each variant is a programmatic defect (swapped, missing or doubled letter, wrong "
                    "case, an erased-and-scribbled glyph, added gibberish) whose exact literal "
                    "transcription is known. These images calibrate the verifiers and test the reward; "
                    "they are much cleaner than real diffusion output. Upload a generated image to "
                    "score a real model."
                )
            with gr.Column(scale=5):
                score_button = gr.Button("Score image", variant="primary")
                result = gr.Markdown("")
                readings = gr.Dataframe(
                    headers=["Verifier", "Transcription", "Accuracy", "Malformed", "Extra"],
                    label="Blind transcriptions", wrap=True, interactive=False,
                )
                with gr.Accordion("Scoring details", open=False):
                    metrics = gr.JSON(label="Metrics")
        outputs = [task_id, progress, prompt, target, image, result, readings, metrics]
        shuffle_button.click(playground.choose, [split], outputs, api_name="choose")
        previous_button.click(
            lambda s, c: playground.choose(s, c, -1), [split, task_id], outputs, api_name="previous"
        )
        next_button.click(
            lambda s, c: playground.choose(s, c, 1), [split, task_id], outputs, api_name="next"
        )
        split.change(playground.choose, [split], outputs, api_name=False)
        demo.load(playground.choose, [split], outputs, api_name=False)
        synth_button.click(playground.synthesise, [task_id, variant], [image], api_name="synthesise")
        score_button.click(
            playground.submit, [task_id, image], [result, readings, metrics], api_name="submit"
        )
        with gr.Accordion("How images are scored", open=False):
            gr.Markdown(
                "The image is resized so its long side is 1536 px, then each verifier model "
                "transcribes it literally, marking broken glyphs with `�`. Text accuracy is "
                "1 − CER of the closest span to the target (best verifier). Each malformed glyph "
                "(worst verifier) multiplies the score by 0.8; text beyond the target costs up to "
                "0.3; wrong letter case costs 10%. Provider failures assign no reward."
            )
    return demo
