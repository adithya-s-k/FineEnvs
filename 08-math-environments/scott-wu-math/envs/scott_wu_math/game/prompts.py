"""What the model reads. The same text is shown on the play page under "Prompt"."""

from __future__ import annotations

from .cards import Card

SYSTEM_PROMPT = (
    "You are playing a mental arithmetic card game. Think it through, check your arithmetic, "
    "and finish with your expression inside <answer></answer>."
)

RULES = (
    "Use every card exactly once and combine them with + - * / and parentheses to make exactly the target. "
    "A = 1, J = 11, Q = 12, K = 13. Division is exact, so fractions along the way are fine. "
    "You may not join cards into longer numbers, use powers, or put a minus sign in front of a card."
)


def user_prompt(cards: list[Card], target: int) -> str:
    labels = " ".join(c.label for c in cards)
    values = ", ".join(str(c.rank) for c in cards)
    return (
        f"Cards: {labels}\n"
        f"Values: {values}\n"
        f"Target: {target}\n\n"
        f"{RULES}\n\n"
        "Give only the final expression inside the tags, for example <answer>(3 + 5) * 2 - 1</answer>."
    )


def messages(cards: list[Card], target: int) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(cards, target)},
    ]
