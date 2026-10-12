from __future__ import annotations

from dataset.pipeline.solver import GraphReaction, RouteGraph, is_convergent, steps_depth, steps_leaves
from dataset.pipeline.splits import DEPTH_QUOTAS, _quotas


def reaction(product, *reactants, cls="amide coupling", popularity=1):
    return GraphReaction(product, tuple(sorted(reactants)), cls, popularity, ("US1",))


# T <- A + B (amide); A <- a1 (nitro reduction); A <- a2 + a3 (SNAr); B in stock; T <- C (Boc deprotection); C <- c1 + c2
GRAPH = [
    reaction("T", "A", "B"),
    reaction("A", "a1", cls="nitro reduction", popularity=5),
    reaction("A", "a2", "a3", cls="SNAr"),
    reaction("T", "C", cls="Boc deprotection"),
    reaction("C", "c1", "c2", cls="amide coupling"),
]
STOCK = frozenset({"B", "a1", "a2", "a3", "c1", "c2"})


def test_shortest_depth_and_witness():
    graph = RouteGraph(GRAPH, STOCK)
    assert graph.depth["T"] == 2 and graph.depth["A"] == 1
    steps = graph.witness("T")
    assert steps_depth("T", steps) == 2 and steps_leaves("T", steps) <= STOCK
    # the more popular of two equally short reactions for A is chosen
    assert ("A", ("a1",)) in steps or ("C", ("c1", "c2")) in steps


def test_constrained_solves_route_around_the_constraint():
    graph = RouteGraph(GRAPH, STOCK)
    # Both ways to T need an amide coupling.
    assert graph.solve("T", forbidden=frozenset({"amide coupling"})) is None
    depth, steps = graph.solve("T", forbidden=frozenset({"nitro reduction"}))
    assert depth == 2 and ("A", ("a1",)) not in steps
    depth, steps = graph.solve("T", excluded=frozenset({"a1"}))
    assert depth == 2 and "a1" not in steps_leaves("T", steps)
    assert graph.solve("T", excluded=frozenset({"B", "c1"})) is None


def test_first_cuts_are_all_solvable_first_reactions():
    graph = RouteGraph(GRAPH, STOCK)
    cuts = graph.first_cuts("T", max_depth=3)
    assert {graph.reactions[i].reactants for _, i in cuts} == {("A", "B"), ("C",)}


def test_an_in_stock_intermediate_shortens_the_route():
    graph = RouteGraph(GRAPH, STOCK | {"A"})
    assert graph.depth["T"] == 1


def test_convergence_needs_two_made_precursors():
    assert is_convergent([("T", ("A", "C")), ("A", ("a1",)), ("C", ("c1", "c2"))], STOCK)
    assert not is_convergent([("T", ("A", "B")), ("A", ("a1",))], STOCK)


def test_quotas_sum_to_the_split_size():
    for shares in DEPTH_QUOTAS.values():
        assert sum(_quotas(shares, 1000).values()) == 1000
