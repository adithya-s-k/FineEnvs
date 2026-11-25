"""Select DB-only task criteria and label from recorded end-state hashes."""

from __future__ import annotations

import copy
import re

from trace_audit.core.provenance import resolve_model
from trace_audit.core.records import training_split
from trace_audit.core.tool_semantics import TAU2_WRITE_TOOLS

from .common import DataError, chat_to_atif, make_record


def db_only_task(task: dict) -> bool:
    criteria = task.get("evaluation_criteria")
    if not isinstance(criteria, dict):
        return False
    basis = criteria.get("reward_basis", ["DB", "COMMUNICATE"])
    # Missing/empty communicate_info is neutral upstream: no judge call and no added requirement.
    return (
        isinstance(basis, list)
        and set(basis) in ({"DB"}, {"DB", "COMMUNICATE"})
        and ("COMMUNICATE" not in basis or not criteria.get("communicate_info"))
    )


def db_change_task(task: dict, domain: str) -> bool:
    """Necessary screen, not reference-route grading: does the target require a DB write?

    No actual trace calls are consulted. Read-only/no-op tasks need human outcome
    review because DB equality cannot establish their informational/policy success.
    """
    return db_only_task(task) and any(
        isinstance(action, dict) and action.get("name") in TAU2_WRITE_TOOLS.get(domain, ())
        for action in (task["evaluation_criteria"].get("actions") or [])
    )


def convert(row: dict, *, revision: str, license: str = "MIT") -> dict:
    task, simulation = row.get("task", {}), row.get("simulation", {})
    domain = row.get("domain")
    if domain not in {"airline", "telecom", "retail"}:
        raise DataError("Only the spec's three domains are admitted")
    task_id, trial_id = task.get("id"), simulation.get("id")
    if task_id is None or trial_id is None:
        raise DataError("Task and trial IDs required")
    failed, checker, requires_change = _db_outcome(row, task, domain, simulation, revision)
    messages = copy.deepcopy(simulation.get("messages"))
    if isinstance(messages, list):
        for message in messages:
            if message.get("role") == "tool" and "tool_call_id" not in message:
                message["tool_call_id"] = message.get("id")
    family = f"tau2:{domain}:{task_id}"
    source = {
        "dataset": "sierra-research/tau2-bench",
        "revision": revision,
        "license": license,
        "original_id": str(trial_id),
        "domain": domain,
        "source_split": row.get("source_split", "base"),
        "termination_reason": simulation.get("termination_reason"),
        "agent_model": resolve_model(row.get("agent_model"), simulation.get("agent_model")),
        "user_model": resolve_model(row.get("user_model"), simulation.get("user_model")),
    }
    if failed is None:
        source["label_exclusion_reason"] = _exclusion_reason(task, requires_change, simulation)
    return make_record(
        source=source,
        task={
            "instruction": row.get("instruction", ""),
            "policy": row.get("policy"),
            "success_criteria": None,
        },
        trajectory=chat_to_atif(messages, tools=row.get("tools"), allow_user_tools=True),
        family=family,
        split="test" if domain == "retail" else training_split(family),
        failed=failed,
        label_source="verified_outcome",
        checker=checker,
    )


def convert_results(
    results: dict,
    *,
    revision: str,
    domain: str,
    tools: list,
    policy: str,
    agent_model: str | None = None,
    user_model: str | None = None,
) -> list[dict]:
    """Join a native monolithic half-duplex Results export without matching reference actions."""
    tasks = {str(task["id"]): task for task in results["tasks"]}
    converted = []
    info = results.get("info") or {}
    agent_model = resolve_model(agent_model, (info.get("agent_info") or {}).get("llm"))
    user_model = resolve_model(user_model, (info.get("user_info") or {}).get("llm"))
    for simulation in results["simulations"]:
        task = tasks[str(simulation["task_id"])]
        messages = simulation.get("messages") or []
        instruction = next(
            (
                m["content"]
                for m in messages
                if m.get("role") == "user"
                and isinstance(m.get("content"), str)
                and m["content"].strip()
            ),
            "",
        )
        converted.append(
            convert(
                {
                    "task": task,
                    "simulation": simulation,
                    "domain": domain,
                    "agent_model": agent_model,
                    "user_model": user_model,
                    "instruction": instruction,
                    "policy": simulation.get("policy") or policy,
                    "tools": tools,
                },
                revision=revision,
            )
        )
    return converted


def _db_outcome(row, task, domain, simulation, revision):
    verification = row.get("db_verification") or {}
    expected, actual = verification.get("expected_db_hash"), verification.get("actual_db_hash")
    hashes = all(
        isinstance(v, str) and re.fullmatch(r"[a-fA-F0-9]{32,128}", v) for v in (expected, actual)
    )
    requires_change, change_evidence = _required_change(
        task, domain, verification, hashes, expected
    )
    eligible = db_only_task(task) and requires_change
    trusted = eligible and hashes and verification.get("completed") is True
    failed = expected.casefold() != actual.casefold() if trusted else None
    checker = (
        {
            "kind": "db_state",
            "revision": revision,
            "expected_db_hash": expected,
            "actual_db_hash": actual,
        }
        if trusted
        else None
    )
    # Native DBCheck is a boolean end-state equality check, distinct from db_reward and
    # aggregate reward. A collector may instead supply independently captured hashes above.
    db_check = (simulation.get("reward_info") or {}).get("db_check") or {}
    if not trusted and eligible and type(db_check.get("db_match")) is bool:
        failed = not db_check["db_match"]
        checker = {
            "kind": "db_state",
            "revision": revision,
            "field": "reward_info.db_check.db_match",
            "db_match": db_check["db_match"],
            "method": "upstream final-state equality; no reward float used",
        }
    if checker:
        checker.update(task_requires_db_change=True, db_change_evidence=change_evidence)
        if change_evidence == "expected_vs_initial_hash":
            checker["initial_db_hash"] = verification["initial_db_hash"]
    return failed, checker, requires_change


def _required_change(task, domain, verification, hashes, expected):
    requires_change = db_change_task(task, domain)
    change_evidence = "expected_mutation_action"
    if "initial_db_hash" in verification:
        initial = verification["initial_db_hash"]
        requires_change = bool(
            hashes
            and isinstance(initial, str)
            and re.fullmatch(r"[a-fA-F0-9]{32,128}", initial)
            and initial.casefold() != expected.casefold()
        )
        change_evidence = "expected_vs_initial_hash"
    return requires_change, change_evidence


def _exclusion_reason(task, requires_change, simulation):
    stops = {
        "max_steps",
        "timeout",
        "too_many_errors",
        "agent_error",
        "user_error",
        "infrastructure_error",
        "context_window_exceeded",
        "unexpected_error",
    }
    stop = simulation.get("termination_reason")
    return (
        "ineligible_outcome_criteria"
        if not db_only_task(task)
        else "db_only_no_verified_required_change"
        if not requires_change
        else f"unverified_early_stop:{stop}"
        if stop in stops
        else "missing_db_outcome"
    )
