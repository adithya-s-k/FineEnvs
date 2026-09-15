import random

import gradio as gr
from PIL import Image

from ..data.catalog import SPLITS
from ..data.schema import FAMILIES
from ..models import NayanaAction
from .environment import NayanaEnvironment


def build_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    catalog = web_manager.env.catalog
    languages = catalog.manifest["config"]["languages"]

    def choose(split, language, family):
        # SQL filters metadata only; the playground never advances a training cursor.
        with catalog._connect() as db:
            candidates = db.execute(
                """SELECT id FROM tasks WHERE split=?
                AND json_extract(payload,'$.language')=? AND json_extract(payload,'$.family')=?""",
                (split, language, family),
            ).fetchall()
        if not candidates:
            raise gr.Error(
                "No matching tasks in this prepared window. Choose another split or language."
            )
        task = catalog.get(random.choice(candidates)[0])
        with Image.open(catalog.asset(task["asset_sha256"])[0]) as image:
            preview = image.copy()
        return task["task_id"], preview, task["prompt"], "", None

    def submit(task_id, answer):
        if not task_id:
            raise gr.Error("Load a task first.")
        env = NayanaEnvironment(catalog)
        try:
            env.reset(task_id=task_id)
            result = env.step(NayanaAction(answer=answer))
            return {"reward": result.reward, **result.metrics}
        finally:
            env.close()

    with gr.Blocks(title="Nayana multilingual OCR") as demo:
        gr.Markdown(
            "# Nayana multilingual OCR\nRead a document region or answer a multiple-choice question. "
            "Tasks come from a fixed, prepared window of the Nayana corpus."
        )
        task_id = gr.State("")
        with gr.Row():
            split = gr.Dropdown(list(SPLITS), value="train", label="Split")
            language = gr.Dropdown(languages, value=languages[0], label="Language")
            family = gr.Dropdown(list(FAMILIES), value=FAMILIES[0], label="Task")
        load = gr.Button("Load task", variant="primary")
        image = gr.Image(type="pil", label="Document", height=500)
        prompt = gr.Textbox(label="Instructions", interactive=False, lines=5)
        answer = gr.Textbox(label="Your answer", lines=4)
        score_button = gr.Button("Score answer")
        result = gr.JSON(label="Result")
        load.click(
            choose, [split, language, family], [task_id, image, prompt, answer, result]
        )
        score_button.click(submit, [task_id, answer], result)
        gr.Markdown(
            "Source: [Cognitive-Lab/NayanaOCR_Corpus_2025](https://huggingface.co/datasets/"
            "Cognitive-Lab/NayanaOCR_Corpus_2025), CC BY-NC 4.0. "
            "Section OCR scores NFC-normalized character error and exact match; VQA scores the option letter."
        )
    return demo
