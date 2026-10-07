"""A scripted chemist that replays a task's reference routes through a RetroEnv server.

This generates SFT trajectories. The expert reads the private task, so it knows
the patent route, but it acts through ``agent.run_episode`` like any model: every
tool result and the reward come from the server, and the stored episode has the
same format as an evaluation episode.

It plans the way a chemist does:

1. Look before cutting. Most episodes start by inspecting the target and
   searching the training precedents for close analogues.
2. Propose disconnections in order. Sometimes the first is a plausible cut the
   patent did not use (``retroenv.disconnections`` ranks them by reliability and
   convergence); ``validate_disconnection`` rejects it, and the expert moves on to
   the next candidate.
3. Validate the patent's cut and look up every new piece by exact stock search. A
   piece the stock lacks gets its own disconnection; one the stock holds stays a
   leaf, even where the patent made it.
4. Emit the trees, naming each reaction's family, reagent roles and evidence.

Text and metadata cite only tool results and chemistry read from the structures.
A precedent is cited only when its reaction type matches the cut. The task's own
patent, reaction IDs and reagents are never written: the student cannot observe
them and would learn to invent them. Choices are seeded by the episode ID, so a
rerun reproduces an episode and another attempt explores differently.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from retroenv.benchmark import load_benchmark
from retroenv.chemistry import canonicalize_smiles
from retroenv.disconnections import (
    UNCERTAIN,
    Disconnection,
    describe,
    reaction_phrase,
    role,
    step_family,
    strategic_disconnections,
)
from retroenv.models import RetroTask
from retroenv.reactions import ReactionLibrary
from retroenv.verifier import compliant_references

from . import agent
from .client import RetroEnvClient

MODEL = "reference-expert"
# Headroom under the server's 32 tool calls and the loop's 16 turns.
CALL_BUDGET = 30
TURN_BUDGET = 14


@dataclass(frozen=True)
class _Function:
    name: str
    arguments: str


@dataclass(frozen=True)
class _ToolCall:
    """The parts of an OpenAI tool call that ``agent.run_episode`` reads."""

    id: str
    function: _Function

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {"name": self.function.name, "arguments": self.function.arguments},
        }


def _routes(task: RetroTask, stock: frozenset[str]) -> list[dict[str, tuple[str, ...]]]:
    """One product -> reactants map per submitted route, one route per distinct first cut.

    Only known routes that satisfy the task's depth, stock and class constraints are replayed."""
    target = canonicalize_smiles(task.target_smiles)
    routes: list[dict[str, tuple[str, ...]]] = []
    first_cuts: set[tuple[str, ...]] = set()
    available = stock - frozenset(canonicalize_smiles(s) for s in task.constraints.excluded_stock)
    for route in compliant_references(task, available, frozenset(task.constraints.forbidden_classes)):
        steps = {canonicalize_smiles(step.product): tuple(step.reactants) for step in route.steps}
        first = tuple(sorted(steps.get(target, ())))
        if not first or first in first_cuts:
            continue
        first_cuts.add(first)
        routes.append(steps)
        if len(routes) == task.max_routes:
            break
    return routes


@dataclass
class _Observed:
    """What the conversation so far has shown the expert."""

    stock: dict[str, bool] = field(default_factory=dict)
    validated: dict[tuple, bool] = field(default_factory=dict)
    inspect: dict[str, Any] | None = None
    precedents: list[dict[str, Any]] | None = None
    researched: bool = False
    last: list[tuple[str, dict[str, Any], dict[str, Any]]] = field(default_factory=list)


def _observe(messages: list[dict[str, Any]]) -> _Observed:
    seen = _Observed()
    calls: dict[str, tuple[str, dict[str, Any]]] = {}
    last_assistant = max((i for i, m in enumerate(messages) if m.get("role") == "assistant"), default=-1)
    for message in messages:
        for call in message.get("tool_calls") or []:
            calls[call["id"]] = (call["function"]["name"], json.loads(call["function"]["arguments"]))
    for index, message in enumerate(messages):
        if message.get("role") != "tool" or message.get("tool_call_id") not in calls:
            continue
        name, arguments = calls[message["tool_call_id"]]
        result = json.loads(message["content"])
        if name == "stock_retrieve":
            seen.stock[canonicalize_smiles(arguments["query"])] = bool(result.get("results"))
        elif name == "validate_disconnection":
            seen.validated[(arguments["product_smiles"], tuple(arguments["reactants"]))] = bool(result.get("supported"))
        elif name == "inspect_molecule":
            seen.inspect, seen.researched = result, True
        elif name == "reaction_precedent_search":
            seen.precedents, seen.researched = result.get("results") or [], True
        if index > last_assistant:
            seen.last.append((name, arguments, result))
    return seen


class ReferenceExpert:
    """OpenAI-compatible stand-in for a model whose next turn is computed from the conversation.

    ``research`` and ``alternatives`` override the seeded choices (tests use them):
    whether to look before cutting, and how many plausible wrong cuts to try at the target.
    """

    def __init__(
        self,
        task: RetroTask,
        library: ReactionLibrary,
        stock: frozenset[str],
        seed: str | None = None,
        *,
        research: bool | None = None,
        alternatives: int | None = None,
    ):
        self.task = task
        self.seed = seed or task.task_id
        self.target = canonicalize_smiles(task.target_smiles)
        self.routes = _routes(task, stock)
        self.library = library
        self.research = self._u("research") < 0.75 if research is None else research
        self.inspect = self.research and self._u("inspect") < 0.6
        self.precedent = self.research and (not self.inspect or self._u("precedent") < 0.85)
        self.alternatives = self._plan_alternatives(alternatives)
        self._fit_budget()
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    # --- planning -------------------------------------------------------------

    def _u(self, key: str) -> float:
        return int(hashlib.sha256(f"{self.seed}:{key}".encode()).hexdigest()[:12], 16) / 16**12

    def _pick(self, options: list[str], key: str) -> str:
        return options[int(self._u(key) * len(options))]

    def _plan_alternatives(self, forced: int | None) -> dict[str, list[Disconnection]]:
        """Plausible cuts the patent did not make, to try first on the first route's products."""
        if not self.routes:
            return {}
        references: dict[str, set[tuple[str, ...]]] = {}
        for steps in self.routes:
            for product, reactants in steps.items():
                references.setdefault(product, set()).add(tuple(sorted(reactants)))
        plan: dict[str, list[Disconnection]] = {}
        for product in self.routes[0]:
            if forced is not None:
                count = forced if product == self.target else 0
            elif product == self.target:
                u = self._u(f"alternatives:{product}")
                count = 2 if u < 0.10 else 1 if u < 0.45 else 0
            else:
                count = 1 if self._u(f"alternatives:{product}") < 0.15 else 0
            if not count:
                continue
            candidates = [
                d
                for d in strategic_disconnections(product)
                if tuple(sorted(d.reactants)) not in references[product]
                # A cut the train-visible library supports would be accepted, so it is no alternative.
                and not self.library.support(product, list(d.reactants)).supported
            ]
            if candidates[:count]:
                plan[product] = candidates[:count]
        return plan

    def _cost(self) -> tuple[int, int]:
        """Upper bounds on tool calls and turns, as if every intermediate needed its own cut."""
        calls = turns = 1  # the emit turn
        if self.research:
            calls += self.inspect + self.precedent
            turns += 1
        looked: set[str] = set()
        for steps in self.routes:
            for reactants in steps.values():
                new = [r for r in reactants if r not in looked]
                looked.update(new)
                calls += 1 + len(new)
                turns += 1
        for options in self.alternatives.values():
            for option in options:
                calls += 1 + len(option.reactants)
                turns += 1
        return calls, turns

    def _fit_budget(self) -> None:
        while self._cost()[0] > CALL_BUDGET or self._cost()[1] > TURN_BUDGET:
            if self.alternatives:
                # Drop the deepest product's alternatives first, keep the target's longest.
                product = next(p for p in reversed(list(self.routes[0])) if p in self.alternatives)
                del self.alternatives[product]
            elif self.research:
                self.research = self.inspect = self.precedent = False
            else:
                return

    # --- the conversation -------------------------------------------------------

    def _expands(self, smiles: str, steps: dict[str, tuple[str, ...]], stock: dict[str, bool]) -> bool:
        # The target is never in stock (the task builder drops those); anything
        # else is expanded only after a lookup came back empty.
        return smiles in steps and (smiles == self.target or stock.get(smiles) is False)

    def _next(self, seen: _Observed, validate: bool):
        """The next cut to propose, breadth-first per route: (route index, product, reactants, alternative)."""
        for route_index, steps in enumerate(self.routes):
            queue = [self.target]
            while queue:
                product = queue.pop(0)
                if not self._expands(product, steps, seen.stock):
                    continue
                reactants = steps[product]
                if validate:
                    for option in self.alternatives.get(product, []) if route_index == 0 else []:
                        if (product, option.reactants) not in seen.validated:
                            return route_index, product, option.reactants, option
                    if (product, reactants) not in seen.validated:
                        return route_index, product, reactants, None
                elif not all(r in seen.stock for r in reactants):
                    return route_index, product, reactants, None
                queue.extend(reactants)
        return None

    def _analogue(self, seen: _Observed, family: str, phrase: str) -> dict[str, Any] | None:
        """A precedent made by the same, confidently named reaction, if the search returned one."""
        if phrase in UNCERTAIN:
            return None
        for result in seen.precedents or []:
            theirs = step_family(result["reactants"], result["product_smiles"])
            if theirs == family and reaction_phrase(theirs, result["reactants"], result["product_smiles"]) == phrase:
                return result
        return None

    def _summary(self, seen: _Observed) -> str:
        """One or two sentences on what the previous turn's tool results showed."""
        parts: list[str] = []
        for name, _, result in seen.last:
            if name == "inspect_molecule" and result.get("valid", True):
                rings, centres = result.get("rings", 0), result.get("chiral_centres", 0)
                stereo = "" if not centres else ", one stereocentre" if centres == 1 else f", {centres} stereocentres"
                parts.append(f"{result.get('formula')}, {rings} ring{'' if rings == 1 else 's'}{stereo}.")
            elif name == "reaction_precedent_search":
                results = result.get("results") or []
                if not results:
                    parts.append("No close precedent among the training reactions.")
                    continue
                best = results[0]
                family = step_family(best["reactants"], best["product_smiles"])
                phrase = reaction_phrase(family, best["reactants"], best["product_smiles"])
                literature = best.get("literature") or []
                where = f"{literature[0]['identifier']}, " if literature else ""
                made = "" if phrase in UNCERTAIN else f" was made by {phrase}"
                parts.append(f"Closest training precedent ({where}Tanimoto {best['similarity']:.2f}){made}.")
        verdicts = [result for name, _, result in seen.last if name == "validate_disconnection"]
        lookups = [
            (canonicalize_smiles(args["query"]), bool(result.get("results")))
            for name, args, result in seen.last
            if name == "stock_retrieve"
        ]
        missing = [smiles for smiles, found in lookups if not found]
        if verdicts and not verdicts[0].get("supported"):
            parts.append(
                self._pick(
                    [
                        "No support for that cut.",
                        "That cut has no supporting evidence.",
                        "The evidence does not support that cut.",
                    ],
                    f"reject:{len(seen.validated)}",
                )
            )
            if missing:
                parts.append(
                    "One of its pieces is not in stock either."
                    if len(missing) == 1
                    else "Its pieces are not in stock either."
                )
            return " ".join(parts)
        if verdicts:
            parts.append(
                self._pick(["Supported.", "The cut is supported.", "That cut holds."], f"accept:{len(seen.validated)}")
            )
        if lookups:
            if not missing:
                parts.append(
                    "Both pieces are in stock."
                    if len(lookups) == 2
                    else "It is in stock."
                    if len(lookups) == 1
                    else "All pieces are in stock."
                )
            elif len(missing) == len(lookups):
                parts.append(f"Not in stock: {', '.join(missing)}.")
            else:
                # Name what is in stock by the role it plays in the cut just checked.
                cut = next(
                    (
                        (args["product_smiles"], args["reactants"])
                        for name, args, _ in seen.last
                        if name == "validate_disconnection"
                    ),
                    None,
                )
                family = step_family(cut[1], cut[0]) if cut else None
                found = [f"the {role(smiles, family)}" for smiles, ok in lookups if ok]
                parts.append(f"In stock: {', '.join(found)}. Not in stock: {', '.join(missing)}.")
        return " ".join(parts)

    def _proposal(
        self,
        seen: _Observed,
        route_index: int,
        product: str,
        reactants: tuple[str, ...],
        option: Disconnection | None,
        validate: bool,
        new: list[str],
    ) -> str:
        family = option.family if option else step_family(reactants, product)
        phrase = reaction_phrase(family, reactants, product)
        what = describe(family, reactants, product)
        tried = [key for key, ok in seen.validated.items() if key[0] == product and not ok]
        key = f"{product}:{len(seen.validated)}"
        previous = (
            next((o for o in self.alternatives.get(product, []) if (product, o.reactants) == tried[-1]), None)
            if tried
            else None
        )
        # Same bond label and a shared fragment: the two proposals cut the same bond.
        if (
            option is not None
            and previous is not None
            and previous.bond == option.bond
            and set(previous.reactants) & set(option.reactants)
        ):
            lead = f"Or make the same bond by {what}."
        elif option is not None and tried:
            lead = f"Another candidate is the {option.bond}: {what}."
        elif option is not None:
            lead = self._pick(
                [
                    f"Try the {option.bond} first: {what}.",
                    f"One candidate is the {option.bond}: {what}.",
                    f"Start with the {option.bond}: {what}.",
                ],
                key,
            )
        elif product == self.target and route_index > 0:
            lead = f"A second route needs a different first cut: {what}."
        elif tried:
            lead = self._pick([f"Next candidate: {what}.", f"Try {what} instead.", f"Fall back to {what}."], key)
        elif product == self.target:
            lead = self._pick(
                [
                    f"Disconnect the target by {what}.",
                    f"The target reads as the product of {what}.",
                    f"The key disconnection is {what}.",
                ],
                key,
            )
        else:
            mentioned = [
                smiles
                for name, args, result in seen.last
                if name == "stock_retrieve"
                for smiles in [canonicalize_smiles(args["query"])]
                if not result.get("results")
            ]
            if mentioned == [product]:
                lead = self._pick([f"It needs its own disconnection: {what}.", f"Disconnect it by {what}."], key)
            else:
                lead = f"Disconnect {product} by {what}."
        if option is None and product == self.target and route_index == 0 and self._analogue(seen, family, phrase):
            lead += " That matches the closest precedent."
        if not validate:
            tail = "Look up the new pieces." if new else ""
        elif new:
            tail = self._pick(
                [
                    "Check the cut and look up the new pieces.",
                    "Validate it and check the stock.",
                    "Check it against the evidence and look up each piece.",
                ],
                f"tail:{key}",
            )
        else:
            tail = "Check the cut; its pieces were already looked up."
        return f"{lead} {tail}".strip()

    def _tree(self, smiles: str, steps: dict[str, tuple[str, ...]], seen: _Observed) -> dict[str, Any]:
        if not self._expands(smiles, steps, seen.stock):
            return {"type": "mol", "smiles": smiles, "in_stock": seen.stock.get(smiles, False), "children": []}
        reactants = steps[smiles]
        family = step_family(reactants, smiles)
        phrase = reaction_phrase(family, reactants, smiles)
        supported = seen.validated.get((smiles, reactants))
        analogue = self._analogue(seen, family, phrase) if smiles == self.target else None
        what = describe(family, reactants, smiles)
        explanation = f"{what[0].upper()}{what[1:]}."
        if supported:
            explanation += " Supported by validate_disconnection."
        if analogue:
            literature = analogue.get("literature") or []
            where = f" {literature[0]['identifier']}" if literature else ""
            explanation += (
                f" Training precedent{where} (Tanimoto {analogue['similarity']:.2f}) uses the same reaction type."
            )
        metadata = {
            "explanation": explanation,
            "reaction_class": family,
            "confidence": 0.9 if supported and analogue else 0.85 if supported else 0.6,
            "literature": list(analogue.get("literature") or [])[:1] if analogue else [],
            "precursor_roles": {reactant: role(reactant, family) for reactant in reactants},
        }
        return {
            "type": "mol",
            "smiles": smiles,
            "in_stock": False,
            "children": [
                {
                    "type": "reaction",
                    "is_reaction": True,
                    "metadata": metadata,
                    "children": [self._tree(reactant, steps, seen) for reactant in reactants],
                }
            ],
        }

    def _emit(self, seen: _Observed) -> tuple[str, list]:
        routes = [self._tree(self.target, steps, seen) for steps in self.routes]

        def leaves(node: dict[str, Any]) -> list[dict[str, Any]]:
            if not node["children"]:
                return [node]
            return [leaf for child in node["children"][0]["children"] for leaf in leaves(child)]

        def forward(node: dict[str, Any]) -> list[str]:
            """Reaction names in the order a chemist would run them: deepest first."""
            if not node["children"]:
                return []
            reaction = node["children"][0]
            earlier = [phrase for child in reaction["children"] for phrase in forward(child)]
            reactants = [child["smiles"] for child in reaction["children"]]
            return earlier + [reaction_phrase(reaction["metadata"]["reaction_class"], reactants, node["smiles"])]

        missing = sum(not leaf["in_stock"] for route in routes for leaf in leaves(route))
        steps = forward(routes[0])
        if missing:
            lead = f"{missing} leaf molecule(s) are still not in stock; submitting the best route found."
        else:
            lead = self._pick(["The route is complete.", "Every leaf is now bought.", "That closes the route."], "done")
        plan = f" Forward: {' → '.join(steps)}." if len(steps) > 1 else ""
        close = (
            " Submitting both routes."
            if len(routes) == 2
            else f" Submitting {len(routes)} routes."
            if len(routes) > 2
            else ""
        )
        content = " ".join(part for part in (self._summary(seen), f"{lead}{plan}{close}") if part)
        return content, [("emit_routes", {"submission": {"routes": routes}})]

    def _turn(self, messages: list[dict[str, Any]], exposed: set[str]) -> tuple[str, list]:
        seen = _observe(messages)
        if exposed == {"emit_routes"}:
            return self._emit(seen)
        validate = "validate_disconnection" in exposed
        if self.research and not seen.researched:
            calls: list[tuple[str, dict[str, Any]]] = []
            if self.inspect and "inspect_molecule" in exposed:
                calls.append(("inspect_molecule", {"smiles": self.target}))
            if self.precedent and "reaction_precedent_search" in exposed:
                calls.append(("reaction_precedent_search", {"product_smiles": self.target, "limit": 3}))
            if calls:
                names = {name for name, _ in calls}
                if names == {"inspect_molecule"}:
                    text = "Look at the target's structure before choosing a cut."
                elif names == {"reaction_precedent_search"}:
                    text = self._pick(
                        [
                            "Search the training reactions for close analogues first.",
                            "First look for precedents with similar products.",
                        ],
                        "research",
                    )
                else:
                    text = self._pick(
                        [
                            "Look at the target and its closest training precedents first.",
                            "Start by inspecting the target and searching for analogous reactions.",
                            "Before cutting anything, check the structure and look for precedents.",
                        ],
                        "research",
                    )
                return text, calls
        pending = self._next(seen, validate)
        if pending is None:
            return self._emit(seen)
        route_index, product, reactants, option = pending
        new = [reactant for reactant in dict.fromkeys(reactants) if reactant not in seen.stock]
        calls = []
        if validate:
            calls.append(("validate_disconnection", {"product_smiles": product, "reactants": list(reactants)}))
        calls.extend(("stock_retrieve", {"query": reactant, "mode": "exact", "limit": 1}) for reactant in new)
        text = " ".join(
            part
            for part in (
                self._summary(seen),
                self._proposal(seen, route_index, product, reactants, option, validate, new),
            )
            if part
        )
        return text, calls

    def create(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], **_: Any) -> Any:
        exposed = {tool["function"]["name"] for tool in tools}
        content, calls = self._turn(messages, exposed)
        turn = sum(message["role"] == "assistant" for message in messages) + 1
        tool_calls = [
            _ToolCall(f"call_{turn}_{k}", _Function(name, json.dumps(arguments)))
            for k, (name, arguments) in enumerate(calls)
        ]
        message = SimpleNamespace(content=content, tool_calls=tool_calls)
        usage = SimpleNamespace(prompt_tokens=0, completion_tokens=0, model_extra={})
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], model=MODEL, usage=usage)


class ReferenceTasks:
    """Private tasks by ID, so ``run_episode`` can build each episode's expert."""

    def __init__(self, benchmark_dir: str | Path):
        self.benchmark = load_benchmark(str(Path(benchmark_dir).resolve()))
        self.tasks = {task.task_id: task for task in self.benchmark.store.iter_all()}

    def expert(self, task_id: str, seed: str | None = None) -> ReferenceExpert:
        try:
            task = self.tasks[task_id]
            stock = self.benchmark.store.stock(task.stock_id)
            return ReferenceExpert(task, self.benchmark.tool_library, stock, seed)
        except KeyError:
            raise KeyError(f"{task_id} is not in the private tasks given to the reference expert") from None


def run_episode(
    llm: ReferenceTasks, env: RetroEnvClient, opening: dict[str, Any], config: agent.AgentConfig
) -> dict[str, Any]:
    # The episode ID differs per attempt, so a second attempt explores differently.
    return agent.run_episode(llm.expert(opening["task_id"], opening.get("episode_id")), env, opening, config)
