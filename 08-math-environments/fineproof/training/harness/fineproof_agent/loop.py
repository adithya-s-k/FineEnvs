"""The FineProof harness: write a proof in one call, and optionally check it like a strict grader and rewrite it.

By default (`rounds=0`, what the published run trained) an episode is one model call, and its reply is the proof.
QED-Nano's best scaffolds (DeepSeek-Math, Self-Check) all end in a further move: the model reads its own proof
critically and writes a better one. `rounds=N` adds N such check-and-rewrite turns. One episode is one
conversation, and its messages only ever grow (whether the token sequences also extend each other depends on the chat
template; see training/README.md):

    system   how replies are used
    user     the task instruction (the problem)
    assistant   a proof
    user     "check it as a grader would, then write the corrected proof under '## Final proof'"   x rounds (default 0)
    assistant   the critique and the final proof

The graded proof is the first reply (its text under a '## Final proof' heading, if it has one), replaced by each
rewrite's text under its last such heading; a rewrite without the heading keeps the proof before it. `run` also
returns which call the graded proof came from, so a caller can tell whether that call was cut off at the output cap.
The loop takes an async `complete(messages) -> str` function so it runs the same against the training proxy, a local
vLLM, or a test stub.
"""

from __future__ import annotations

import re
from typing import Awaitable, Callable

SYSTEM = (
    "You are solving an olympiad problem by writing a proof. There are no tools: reply in Markdown, with LaTeX for "
    "formulas. Your proof is read by a strict grader who gives credit only for steps that are fully justified."
)
ONLY = (
    "{instruction}\n\nThis harness has no tools: reply with your complete proof in Markdown. It is saved to "
    "/workspace/proof.md for you."
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
              rounds: int = 0) -> tuple[str, list[dict], int]:
    """Run one episode. Returns (the proof to grade, the conversation, the 0-based index of the model call whose
    reply the proof came from)."""
    messages = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": (FIRST if rounds else ONLY).format(instruction=instruction.strip())}]
    reply = await complete(messages)
    messages.append({"role": "assistant", "content": reply})
    proof, submitted = extract_final(reply) or reply.strip(), 0
    for call in range(1, rounds + 1):
        messages.append({"role": "user", "content": CHECK})
        reply = await complete(messages)
        messages.append({"role": "assistant", "content": reply})
        # A round that forgets the heading keeps the previous proof rather than grading a critique as a proof.
        final = extract_final(reply)
        if final:
            proof, submitted = final, call
    return proof, messages, submitted
