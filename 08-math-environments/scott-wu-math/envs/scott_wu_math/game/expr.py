"""Parse, evaluate and check an arithmetic answer.

An answer is an ordinary expression over the card values with + - * / and parentheses,
for example ``(13 - 12 + 11 - 10 + 7) * 9 * 5 * 5``. Every card is used exactly once.
Division is exact (rational), so intermediate fractions are allowed, as in the 24 game.
Unary minus, powers, decimals and joining digits ("1" and "3" into "13") are not allowed.

Expressions are trees: an ``int`` leaf or a ``(op, left, right)`` tuple.
"""

from __future__ import annotations

import ast
import re
from collections import Counter
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Union

Tree = Union[int, tuple]

OPS = ("+", "-", "*", "/")
_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2}
_AST_OPS = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
_LETTERS = {"A": "1", "J": "11", "Q": "12", "K": "13"}
MAX_LENGTH = 400

# What people and models type for the four operators.
_SYMBOLS = str.maketrans(
    {
        "×": "*",
        "x": "*",
        "X": "*",
        "·": "*",
        "∙": "*",
        "÷": "/",
        "−": "-",
        "–": "-",
        "—": "-",
        "[": "(",
        "]": ")",
        "{": "(",
        "}": ")",
    }
)


class ExpressionError(ValueError):
    pass


def apply(op: str, a: Fraction, b: Fraction) -> Fraction:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    if b == 0:
        raise ZeroDivisionError
    return a / b


def evaluate(tree: Tree) -> Fraction:
    if isinstance(tree, int):
        return Fraction(tree)
    op, left, right = tree
    return apply(op, evaluate(left), evaluate(right))


def leaves(tree: Tree) -> list[int]:
    if isinstance(tree, int):
        return [tree]
    return leaves(tree[1]) + leaves(tree[2])


def to_text(tree: Tree, pretty: bool = False) -> str:
    """Write a tree with the fewest parentheses; pretty=True uses × ÷ − like a person would."""
    symbols = {"+": "+", "-": "−", "*": "×", "/": "÷"} if pretty else {o: o for o in OPS}

    def walk(node: Tree) -> tuple[str, int]:
        if isinstance(node, int):
            return str(node), 3
        op, left, right = node
        prec = _PRECEDENCE[op]
        ltext, lprec = walk(left)
        rtext, rprec = walk(right)
        if lprec < prec:
            ltext = f"({ltext})"
        if rprec < prec or (rprec == prec and op in "-/"):
            rtext = f"({rtext})"
        return f"{ltext} {symbols[op]} {rtext}", prec

    return walk(tree)[0]


def steps(tree: Tree) -> list[str]:
    """The solution as spoken, one combination per line: "13 − 12 = 1", "1 + 1 = 2", ..."""
    lines: list[str] = []

    def walk(node: Tree) -> Fraction:
        if isinstance(node, int):
            return Fraction(node)
        op, left, right = node
        a, b = walk(left), walk(right)
        value = apply(op, a, b)
        sym = {"+": "+", "-": "−", "*": "×", "/": "÷"}[op]
        lines.append(f"{fmt(a)} {sym} {fmt(b)} = {fmt(value)}")
        return value

    walk(tree)
    return lines


def fmt(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def normalize(text: str) -> str:
    text = text.strip().translate(_SYMBOLS).replace("**", "^")
    # "... = 2025": judge the left-hand side; the value is checked against the target anyway.
    if "=" in text:
        text = text.split("=", 1)[0]
    # Card letters as values: A, J, Q, K (only as whole tokens).
    text = re.sub(r"\b([AJQK])\b", lambda m: _LETTERS[m.group(1)], text)
    return text.strip()


def parse(text: str) -> Tree:
    """Parse an answer into a tree, rejecting anything but integers, + - * / and parentheses."""
    if not isinstance(text, str) or not text.strip():
        raise ExpressionError("empty expression")
    if len(text) > MAX_LENGTH:
        raise ExpressionError(f"expression longer than {MAX_LENGTH} characters")
    source = normalize(text)
    if "^" in source:
        raise ExpressionError("powers are not allowed")
    try:
        node = ast.parse(source, mode="eval").body
    except SyntaxError as exc:
        raise ExpressionError(f"not an arithmetic expression: {source!r}") from exc

    def build(n: ast.AST) -> Tree:
        if isinstance(n, ast.Constant) and type(n.value) is int and n.value >= 0:
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _AST_OPS:
            return (_AST_OPS[type(n.op)], build(n.left), build(n.right))
        if isinstance(n, ast.UnaryOp):
            raise ExpressionError("unary minus or plus is not allowed; every card must be combined with a binary operator")
        if isinstance(n, ast.Constant):
            raise ExpressionError(f"only whole card values are allowed, not {n.value!r}")
        raise ExpressionError(f"only + - * / and parentheses are allowed (found {type(n).__name__})")

    return build(node)


@dataclass
class Check:
    """Everything the grader knows about one answer."""

    expression: str
    parsed: bool = False
    uses_all_cards: bool = False
    value: Fraction | None = None
    exact: bool = False
    error: str | None = None
    missing: list[int] = field(default_factory=list)
    extra: list[int] = field(default_factory=list)
    canonical: str | None = None

    def to_dict(self) -> dict:
        return {
            "expression": self.expression,
            "parsed": self.parsed,
            "uses_all_cards": self.uses_all_cards,
            "value": None if self.value is None else fmt(self.value),
            "exact": self.exact,
            "error": self.error,
            "missing": self.missing,
            "extra": self.extra,
            "canonical": self.canonical,
        }


def check(expression: str, card_values: list[int], target: int) -> Check:
    result = Check(expression=expression)
    try:
        tree = parse(expression)
    except ExpressionError as exc:
        result.error = str(exc)
        return result
    result.parsed = True
    result.canonical = to_text(tree)
    used, have = Counter(leaves(tree)), Counter(card_values)
    result.missing = sorted((have - used).elements())
    result.extra = sorted((used - have).elements())
    result.uses_all_cards = not result.missing and not result.extra
    try:
        result.value = evaluate(tree)
    except ZeroDivisionError:
        result.error = "division by zero"
        return result
    if not result.uses_all_cards:
        parts = []
        if result.missing:
            parts.append("unused cards " + ", ".join(map(str, result.missing)))
        if result.extra:
            parts.append("numbers that are not cards on the table " + ", ".join(map(str, result.extra)))
        result.error = "; ".join(parts)
        return result
    result.exact = result.value == target
    return result
