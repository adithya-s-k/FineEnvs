"""Private generating-model identities for sampling and descriptive shortcut probes."""

from .records import DataError


def resolve_model(*values) -> str | None:
    known = []
    for value in values:
        if value is None:
            continue
        if not isinstance(value, str) or not value.strip():
            raise DataError("Model provenance must be nonempty text or null")
        if value.strip().casefold() in {"unknown", "none", "null", "missing"}:
            continue
        known.append(value.strip())
    if len(set(known)) > 1:
        raise DataError("Conflicting generating-model provenance; resolve before import")
    return known[0] if known else None


def generator_key(row: dict, *, allow_fixtures=False) -> tuple[str, str] | None:
    source = row["source"]
    agent, user = (resolve_model(source.get(field)) for field in ("agent_model", "user_model"))
    if allow_fixtures and source.get("data_kind") == "smoke_fixture":
        return (agent or "fixture-generator", user or "not_applicable")
    return (agent, user) if agent and user else None


def sampling_key(row: dict, *, allow_fixtures=False) -> tuple | None:
    models = generator_key(row, allow_fixtures=allow_fixtures)
    return (
        (
            row["source"]["dataset"],
            row["source"].get("domain") or "",
            row["group"]["task_family_id"],
            *models,
        )
        if models
        else None
    )
