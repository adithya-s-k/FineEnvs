"""Sample outcome pairs within source/domain/task/model, with bounded record reuse."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from fractions import Fraction

from .provenance import sampling_key
from .records import DataError, scoring_exclusion_reason, validate_collection
from .review import assert_training_audits


def _default_pair_count(capacities: dict, source_cap: float) -> int:
    """Largest feasible sample under both finite pool capacity and source caps."""
    cap = Fraction(str(source_cap))
    total, prefix = sum(capacities.values()), 0
    upper = total
    # For each subset of small sources, the others contribute at most P*cap each.
    for k, capacity in enumerate(sorted(capacities.values()), 1):
        prefix += capacity
        denominator = 1 - (len(capacities) - k) * cap
        if denominator > 0:
            upper = min(upper, int(Fraction(prefix) / denominator))
    for pairs in range(upper, 0, -1):
        limit = int(2 * pairs * source_cap) // 2
        if sum(min(n, limit) for n in capacities.values()) >= pairs:
            return pairs
    return 0


def balanced_batch(
    records: list[dict],
    size: int | None = None,
    *,
    source_cap: float = 0.5,
    seed: int = 42,
    allow_fixtures: bool = False,
    max_record_reuse: int = 1,
    validated: bool = False,
    audit_records: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    if not validated:
        validate_collection(records)
        if audit_records is not None:
            validate_collection(audit_records)
    _check_sampling_inputs(size, source_cap, max_record_reuse)
    pools, excluded = _sampling_pools(records, allow_fixtures)
    by_source, remaining = _pair_capacities(pools, excluded, max_record_reuse)
    # Allocate indivisible success/failure pairs. Never fill one source's missing
    # outcome using another source, even if the global class totals would fit.
    capacities = {source: sum(remaining[key] for key in keys) for source, keys in by_source.items()}
    if size is None:
        size = 2 * _default_pair_count(capacities, source_cap)
    pair_cap = int(size * source_cap) // 2
    if size < 2 or pair_cap < 1 or sum(min(n, pair_cap) for n in capacities.values()) < size // 2:
        raise DataError(
            "Cannot balance paired task/model outcomes under this source cap "
            "and record reuse limit; "
            "collect more paired traces"
        )
    assert_training_audits(
        [
            row
            for row in records
            if row["split"] == "train"
            and row["source"]["dataset"] in by_source
            and scoring_exclusion_reason(row, allow_fixtures=allow_fixtures) is None
            and sampling_key(row, allow_fixtures=allow_fixtures) is not None
        ],
        audit_records=records if audit_records is None else audit_records,
    )
    batch, counts = _draw_pairs(
        pools, remaining, by_source, capacities, size, pair_cap, max_record_reuse, seed
    )
    reuse = Counter(row["record_id"] for row in batch)
    return batch, {
        "source_counts": dict(Counter(row["source"]["dataset"] for row in batch)),
        "source_outcome_counts": {
            source: {"succeeded": n, "failed": n} for source, n in sorted(counts.items())
        },
        "max_record_reuse_limit": max_record_reuse,
        "max_record_reuse": max(reuse.values()),
        "unique_records": len(reuse),
        "sampled_records": len(batch),
        "available_pair_capacity": sum(capacities.values()),
        "task_weighting": "remaining min(success_count, failure_count) pair tickets",
        "balance_unit": "source/domain/task_family/agent_model/user_model",
        "excluded_records": dict(excluded),
    }


def _sampling_pools(records, allow_fixtures):
    pools = defaultdict(lambda: {False: [], True: []})
    excluded = Counter()
    for row in records:
        if row["split"] != "train":
            continue
        reason = scoring_exclusion_reason(row, allow_fixtures=allow_fixtures)
        if reason:
            excluded[reason] += 1
            continue
        key = sampling_key(row, allow_fixtures=allow_fixtures)
        if key is None:
            excluded["missing_model_provenance"] += 1
            continue
        pools[key][row["outcome"]["failed"]].append(row)
    return pools, excluded


def _draw_pairs(pools, remaining, by_source, capacities, size, pair_cap, max_record_reuse, seed):
    rng = random.Random(seed)
    # Each stratum has min(successes, failures) pair tickets per allowed reuse.
    # Drawing tickets weights abundant mixed-outcome tasks above scarce tasks.
    decks = {}
    for key in sorted(remaining):
        outcomes = pools[key]
        decks[key] = {}
        for label, values in outcomes.items():
            deck = values * max_record_reuse
            rng.shuffle(deck)
            decks[key][label] = deck
    counts, batch, source_remaining = Counter(), [], dict(capacities)
    for _ in range(size // 2):
        sources = sorted(s for s in by_source if counts[s] < pair_cap and source_remaining[s])
        source = rng.choices(
            sources, weights=[min(source_remaining[s], pair_cap - counts[s]) for s in sources]
        )[0]
        keys = [key for key in by_source[source] if remaining[key]]
        key = rng.choices(keys, weights=[remaining[key] for key in keys])[0]
        batch.extend(decks[key][failed].pop() for failed in (False, True))
        counts[source] += 1
        remaining[key] -= 1
        source_remaining[source] -= 1
    rng.shuffle(batch)
    return batch, counts


def _pair_capacities(pools, excluded, max_record_reuse):
    by_source = defaultdict(list)
    remaining = {}
    for key, outcomes in sorted(pools.items()):
        if not all(outcomes.values()):
            excluded["task_model_missing_paired_outcome"] += sum(map(len, outcomes.values()))
            continue
        for values in outcomes.values():
            values.sort(key=lambda row: row["record_id"])
        by_source[key[0]].append(key)
        remaining[key] = min(map(len, outcomes.values())) * max_record_reuse
    return by_source, remaining


def _check_sampling_inputs(size, source_cap, max_record_reuse):
    if (
        (size is not None and (type(size) is not int or size < 2 or size % 2))
        or type(source_cap) not in (int, float)
        or not math.isfinite(source_cap)
        or not 0 < source_cap <= 1
    ):
        raise DataError("An even batch size >=2 and a source cap in (0,1] are required")
    if type(max_record_reuse) is not int or max_record_reuse < 1:
        raise DataError("max_record_reuse must be a positive integer")
