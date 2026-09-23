"""Playground with independent per-browser task selection.

Mirrors the OCR environment's interaction: pick a task, answer it, see the score and only
then the reference. The OpenEnv discovery and observation APIs never return a reference;
this deliberately reveals it after grading, which is what makes the playground useful for
judging whether a reward is fair.
"""

import random

import gradio as gr

from ..data.catalog import SPLITS
from ..data.schema import character_scored
from ..models import AsrAction
from .environment import AsrEnvironment

TASK_LABELS = [
    ("Transcription", "transcription"),
    ("Verbatim transcription", "verbatim_transcription"),
    ("Language identification", "language_id"),
]


class Playground:
    def __init__(self, catalog):
        self.catalog = catalog

    def choose(self, split, language, family, current=None, direction=0, index=None):
        empty = (None, "", "No tasks in this selection.", "", "", "", "")
        count = self.catalog.group_count(split, language, family)
        if not count:
            return empty
        if index is None:
            position = (
                self.catalog.group_position(current, split, language, family)
                if current
                else None
            )
            if direction and position is not None:
                index = (position + direction) % count
            else:
                index = random.randrange(count)
        index = max(0, min(int(index), count - 1))
        task = self.catalog.group_at(split, language, family, index)
        path, _ = self.catalog.asset(task["asset_sha256"])
        unit = "character error rate" if character_scored(language) else "word error rate"
        details = (
            f"**{task['language_name']}** · {task['duration_seconds']}s · "
            f"utterance `{task['sample_id']}` · scored by **{unit}**"
        )
        return (
            str(path),
            task["task_id"],
            task["prompt"],
            details,
            f"{index + 1} / {count}",
            "",
            "",
        )

    def submit(self, task_id, answer):
        if not task_id:
            return "Select a task first.", ""
        environment = AsrEnvironment(self.catalog)
        environment.reset(task_id=task_id)
        # The environment returns the observation itself, carrying reward and metrics.
        result = environment.step(AsrAction(transcript=answer or ""))
        metrics = result.metrics
        parts = [f"**Reward {result.reward:.3f}**"]
        for key in ("wer", "cer"):
            if key in metrics:
                parts.append(f"{key.upper()} {metrics[key]:.3f}")
        if metrics.get("exact_match"):
            parts.append("exact match")
        reference = self.catalog.get(task_id)["reference"]
        return " · ".join(parts), reference


def build_ui(web_manager, action_fields, metadata, is_chat_env, title, quick_start_md):
    catalog = web_manager.env.catalog
    playground = Playground(catalog)
    languages = catalog.manifest["config"]["languages"]
    families = [
        (label, name)
        for label, name in TASK_LABELS
        if name in catalog.manifest["config"]["families"]
    ]
    splits = [split for split in SPLITS if catalog.count(split)]
    tasks = sum(item["tasks"] for item in catalog.manifest["counts"])

    with gr.Blocks(title="Multilingual ASR", delete_cache=(300, 600)) as demo:
        gr.Markdown(
            "# Multilingual ASR · FLEURS\n"
            "Listen, transcribe, and see how the reward is computed. Scripts without word "
            "spacing are scored per character; everything else per word.\n\n"
            f"**{tasks:,} tasks** across **{len(languages)} languages**."
        )
        with gr.Row():
            split = gr.Dropdown(splits, value=splits[0], label="Split", scale=1)
            language = gr.Dropdown(
                languages, value=languages[0], label="Language", scale=2
            )
            family = gr.Dropdown(
                [(label, name) for label, name in families],
                value=families[0][1],
                label="Task",
                scale=2,
            )
        with gr.Row():
            previous = gr.Button("← Previous")
            shuffle = gr.Button("Shuffle", variant="primary")
            following = gr.Button("Next →")
            position = gr.Textbox(label="Position", interactive=False, scale=1)

        details = gr.Markdown()
        audio = gr.Audio(label="Utterance", type="filepath", interactive=False)
        prompt = gr.Textbox(label="Prompt", interactive=False, lines=2)
        answer = gr.Textbox(label="Your transcript", lines=3, placeholder="Type what you hear")
        grade = gr.Button("Submit", variant="primary")
        score = gr.Markdown()
        reference = gr.Textbox(label="Reference (revealed after scoring)", lines=3, interactive=False)
        task_id = gr.State("")

        outputs = [audio, task_id, prompt, details, position, score, reference]

        def pick(s, lang, fam):
            return playground.choose(s, lang, fam)

        def step_back(s, lang, fam, current):
            return playground.choose(s, lang, fam, current, -1)

        def step_forward(s, lang, fam, current):
            return playground.choose(s, lang, fam, current, 1)

        for control in (split, language, family):
            control.change(pick, [split, language, family], outputs)
        shuffle.click(pick, [split, language, family], outputs)
        previous.click(step_back, [split, language, family, task_id], outputs)
        following.click(step_forward, [split, language, family, task_id], outputs)
        grade.click(playground.submit, [task_id, answer], [score, reference])
        demo.load(pick, [split, language, family], outputs)
    return demo
