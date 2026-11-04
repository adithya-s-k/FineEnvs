"""A standard 52-card deck. Ace is 1, Jack 11, Queen 12, King 13."""

from __future__ import annotations

import random
from dataclasses import dataclass

RANK_NAMES = {1: "A", 11: "J", 12: "Q", 13: "K"}
SUITS = ("S", "H", "D", "C")
SUIT_SYMBOLS = {"S": "♠", "H": "♥", "D": "♦", "C": "♣"}


@dataclass(frozen=True, order=True)
class Card:
    rank: int
    suit: str

    @property
    def name(self) -> str:
        return RANK_NAMES.get(self.rank, str(self.rank))

    @property
    def label(self) -> str:
        """Short label with the suit symbol, e.g. "Q♥" or "10♣"."""
        return f"{self.name}{SUIT_SYMBOLS[self.suit]}"

    def to_dict(self) -> dict:
        return {"rank": self.rank, "suit": self.suit, "label": self.label}

    @classmethod
    def from_dict(cls, data: dict) -> "Card":
        return cls(rank=int(data["rank"]), suit=str(data["suit"]))


FULL_DECK = tuple(Card(rank, suit) for suit in SUITS for rank in range(1, 14))


def deal(n: int, rng: random.Random) -> list[Card]:
    """Deal n distinct cards from one shuffled deck, so no value appears more than four times."""
    if not 1 <= n <= len(FULL_DECK):
        raise ValueError(f"can deal 1..52 cards, not {n}")
    return rng.sample(FULL_DECK, n)


def cards_from_ranks(ranks: list[int], rng: random.Random) -> list[Card]:
    """Build cards for given values, giving each value a different suit where the deck allows."""
    used: dict[int, list[str]] = {}
    cards = []
    for rank in ranks:
        if not 1 <= rank <= 13:
            raise ValueError(f"card values run from 1 (A) to 13 (K), not {rank}")
        taken = used.setdefault(rank, [])
        if len(taken) == 4:
            raise ValueError(f"a deck has only four cards of value {rank}")
        suit = rng.choice([s for s in SUITS if s not in taken])
        taken.append(suit)
        cards.append(Card(rank, suit))
    return cards
