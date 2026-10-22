"""Cheap probes fitted only on train. TF-IDF uses cosine nearest class centroids."""

from __future__ import annotations

import json
import math
import re
from collections import Counter

from .render import render_record, task_text
from .tool_semantics import TAU2_READ_TOOLS, TAU2_WRITE_TOOLS

CLAIM = re.compile(
    r"\b(?:successfully|(?:I(?:'ve| have)?\s+)?completed|all done|task (?:is )?done)\b", re.I
)


def closing_claim(row: dict) -> bool:
    return bool(CLAIM.search(last_agent_message(row)))


def last_agent_message(row: dict) -> str:
    steps = row["trajectory"]["steps"]
    return next(
        (
            s["message"]
            for s in reversed(steps)
            if s["source"] == "agent" and isinstance(s["message"], str)
        ),
        "",
    )


def last_tool_error(row: dict) -> bool:
    for step in reversed(row["trajectory"]["steps"]):
        results = (step.get("observation") or {}).get("results", [])
        if results:
            return any(
                ((r.get("extra") or {}).get("error") or {}).get("is_error") is True for r in results
            )
    return False


def last_shell_nonzero_exit(row: dict) -> bool:
    for step in reversed(row["trajectory"]["steps"]):
        for result in reversed((step.get("observation") or {}).get("results", [])):
            status = (result.get("extra") or {}).get("command_status")
            if status is not None:
                return status["nonzero_exit"]
    return False


def tool_call_count(row: dict) -> int:
    def count(trajectory):
        own = sum(
            len(s.get("tool_calls") or [])
            + len((s.get("extra") or {}).get("participant_tool_calls") or [])
            for s in trajectory["steps"]
        )
        return own + sum(count(child) for child in trajectory.get("subagent_trajectories") or [])

    return count(row["trajectory"])


def write_tool_call_count(row: dict) -> int | None:
    """Count attempted agent DB writes, not user actions or successful state changes.

    Shell commands and unknown schemas are unclassified.
    """
    domain = row["source"].get("domain")
    dataset = row["source"]["dataset"].casefold()
    if (
        dataset == "microsoft/agentrx"
        and row["group"]["task_family_id"].startswith("tau2:retail:")
        and (domain == "retail" or row["source"].get("source_split") in {"tau_retail", "retail"})
    ):
        domain = "retail"  # Explicit canonical tau2 retail join.
    elif dataset != "sierra-research/tau2-bench":
        return None
    if domain not in TAU2_WRITE_TOOLS:
        return None

    return _write_count(row["trajectory"], domain)


class TfidfCentroid:
    """Sparse stdlib baseline; scores are similarities, not calibrated posterior estimates."""

    def __init__(self, texts: list[str], labels: list):
        self.labels = sorted(set(labels))
        self.priors = Counter(labels)
        df = Counter(word for text in texts for word in set(self._words(text)))
        self.idf = {w: math.log((1 + len(texts)) / (1 + n)) + 1 for w, n in df.items()}
        self.centroids = {label: Counter() for label in self.labels}
        for text, label in zip(texts, labels, strict=False):
            self.centroids[label].update(self._vector(text))
        self.centroids = {
            label: self._normalize(vector) for label, vector in self.centroids.items()
        }

    @staticmethod
    def _words(text):
        return re.findall(r"\b\w+\b", text.casefold())

    @staticmethod
    def _normalize(vector):
        norm = math.sqrt(sum(v * v for v in vector.values())) or 1
        return {key: value / norm for key, value in vector.items()}

    def _vector(self, text):
        return self._normalize(
            {
                w: (1 + math.log(n)) * self.idf[w]
                for w, n in Counter(self._words(text)).items()
                if w in self.idf
            }
        )

    def scores(self, text):
        vector = self._vector(text)
        return {
            label: sum(v * self.centroids[label].get(w, 0) for w, v in vector.items())
            for label in self.labels
        }

    def probability(self, text):
        scores = self.scores(text)
        total = sum(scores.values())
        return (
            scores.get(True, 0) / total if total else self.priors[True] / sum(self.priors.values())
        )


def fit_baselines(train: list[dict], *, render=None) -> dict:
    render = render or render_record
    labels = [row["outcome"]["failed"] for row in train]
    prior = sum(labels) / len(labels)

    # Use public rendering even for task-only input so agent/model names receive identical masking.
    def task(row):
        return task_text(json.loads(render(row)))

    task_model = TfidfCentroid([task(row) for row in train], labels)
    trace_model = TfidfCentroid([render(row) for row in train], labels)
    return {
        "training_prior": lambda row: prior,
        "task_only_tfidf": lambda row: task_model.probability(task(row)),
        "trace_tfidf": lambda row: trace_model.probability(render(row)),
        "closing_keyword": lambda row: 0.9 if closing_claim(row) else 0.1,
        "last_tool_error": lambda row: 0.9 if last_tool_error(row) else 0.1,
        "last_shell_nonzero_exit": lambda row: 0.9 if last_shell_nonzero_exit(row) else 0.1,
    }


def _write_count(trajectory, domain):
    total = 0
    for step in trajectory["steps"]:
        for call in step.get("tool_calls") or []:
            name = call["function_name"]
            if name in TAU2_WRITE_TOOLS[domain]:
                total += 1
            elif name not in TAU2_READ_TOOLS[domain]:
                return None
    for child in trajectory.get("subagent_trajectories") or []:
        value = _write_count(child, domain)
        if value is None:
            return None
        total += value
    return total
