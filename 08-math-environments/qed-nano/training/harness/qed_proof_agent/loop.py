"""The qed-proof harness: write a proof, check it like a strict grader, rewrite it.

QED-Nano's best scaffolds (DeepSeek-Math, Self-Check) all end in the same move: the model reads its own proof
critically and writes a better one. This harness trains that move directly. One episode is one conversation,
append-only so every turn is a prefix of the next (what OpenEnv's capture needs for training):

    system   how replies are used
    user     the task instruction (the problem)
    assistant   a first proof
    user     "check it as a grader would, then write the corrected proof under '## Final proof'"   x rounds
    assistant   the critique and the final proof

The final proof (the text under the last '## Final proof' heading) is what gets graded. The loop takes an
async `complete(messages) -> str` function so it runs the same against the training proxy, a local vLLM, or a test stub.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

SYSTEM = (
    "You are solving an olympiad problem by writing a proof. There are no tools: reply in Markdown, with LaTeX for "
    "formulas. Your proof is read by a strict grader who gives credit only for steps that are fully justified."
)
FIRST = (
    "{instruction}\n\nThis harness has no tools: reply with your complete proof in Markdown. You will then be asked "
    "to check it, and your final version is saved to /workspace/proof.md for you."
)
CHECK = (
    "Now grade your proof as a strict olympiad grader would. Go through it step by step and list every gap, "
    "unjustified claim, computational error or missing case you find. Then write the complete corrected proof, from "
    "the beginning, under a line that says exactly `## Final proof`. If you find nothing wrong, restate the full "
    "proof under that heading anyway."
)
FINAL = re.compile(r"^\s*#{1,6}\s*Final proof\s*:?\s*$", re.I | re.M)


def extract_final(reply: str) -> str | None:
    """The text after the last '## Final proof' heading, or None if there is no such heading."""
    matches = list(FINAL.finditer(reply or ""))
    if not matches:
        return None
    text = reply[matches[-1].end():].strip()
    return text or None


async def run(complete: Callable[[list[dict]], Awaitable[str]], instruction: str,
              rounds: int = 1) -> tuple[str, list[dict]]:
    """Run one episode. Returns (the proof to grade, the conversation)."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": FIRST.format(instruction=instruction.strip())}]
    reply = await complete(messages)
    messages.append({"role": "assistant", "content": reply})
    proof = extract_final(reply) or reply.strip()
    for _ in range(rounds):
        messages.append({"role": "user", "content": CHECK})
        reply = await complete(messages)
        messages.append({"role": "assistant", "content": reply})
        # A round that forgets the heading keeps the previous proof rather than grading a critique as a proof.
        proof = extract_final(reply) or proof
    return proof, messages
