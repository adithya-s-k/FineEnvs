from __future__ import annotations

import copy

from conftest import AMINO_ESTER, ESTER, NITRO_ESTER, STOCK, make_task, step
from retroenv.graph import routes_to_submission
from retroenv.models import ReferenceRoute
from retroenv.verifier import FALSE_STOCK_CAP, OFF_TARGET_CAP, RouteVerifier

KNOWN = (step(ESTER, "CCO", "CC(=O)O"),)
ACYL_CHLORIDE = (step(ESTER, "CCO", "CC(=O)Cl"),)
TWO_STEP = (
    step(AMINO_ESTER, NITRO_ESTER),
    step(NITRO_ESTER, "CCO", "O=C(O)c1ccc([N+](=O)[O-])cc1"),
)


def submit(task, *routes, stock=STOCK):
    return routes_to_submission(
        task.target_smiles, [ReferenceRoute(f"r{i}", steps) for i, steps in enumerate(routes)], stock
    )


def test_the_known_route_scores_one(library):
    task = make_task()
    score = RouteVerifier(library).score_submission(task, submit(task, KNOWN), STOCK)
    assert score.valid and score.reward == 1.0
    assert score.metrics["reference_match"] is True


def test_a_valid_route_the_patent_did_not_use_passes_on_chemistry_alone(library):
    task = make_task()
    score = RouteVerifier(library).score_submission(task, submit(task, ACYL_CHLORIDE), STOCK)
    assert score.valid
    assert score.metrics["reference_match"] is False
    assert score.step_results[0]["support"] == "template"
    assert score.reward >= 0.8


def test_an_unsupported_step_fails_the_route(library):
    task = make_task()
    bogus = (step(ESTER, "CCO", "CN"),)
    score = RouteVerifier(library).score_submission(task, submit(task, bogus, stock=STOCK), STOCK)
    assert not score.valid
    assert score.components["steps"] == 0.0


def test_a_leaf_missing_from_stock_fails_the_route(library):
    task = make_task()
    score = RouteVerifier(library).score_submission(task, submit(task, KNOWN), STOCK - {"CCO"})
    assert not score.valid
    assert score.metrics["routes"][0]["missing_stock"] == ["CCO"]


def test_a_false_stock_claim_is_capped(library):
    task = make_task()
    stock = STOCK - {"CCO"}
    submission = submit(task, KNOWN, stock=STOCK)  # claims CCO is in stock
    score = RouteVerifier(library).score_submission(task, submission, stock)
    assert score.reward <= FALSE_STOCK_CAP


def test_duplicated_routes_score_lower_than_one_route(library):
    task = make_task()
    verifier = RouteVerifier(library)
    single = verifier.score_submission(task, submit(task, KNOWN), STOCK)
    padded = verifier.score_submission(task, submit(task, KNOWN, KNOWN, KNOWN), STOCK)
    assert padded.reward < single.reward - 0.1
    assert padded.metrics["routes"][1]["duplicate_of_earlier_route"]


def test_an_off_target_submission_is_capped(library):
    task = make_task(target=AMINO_ESTER, routes=(TWO_STEP,))
    score = RouteVerifier(library).score_submission(task, submit(make_task(), KNOWN), STOCK)
    assert score.reward <= OFF_TARGET_CAP


def test_a_two_step_route_needs_depth_budget(library):
    verifier = RouteVerifier(library)
    deep = make_task(target=AMINO_ESTER, routes=(TWO_STEP,), max_depth=2)
    assert verifier.score_submission(deep, submit(deep, TWO_STEP), STOCK).valid
    shallow = make_task(target=AMINO_ESTER, routes=(TWO_STEP,), max_depth=1, variant="max_depth")
    score = verifier.score_submission(shallow, submit(shallow, TWO_STEP), STOCK)
    assert not score.valid
    assert any("exceeds max_depth" in v for v in score.metrics["routes"][0]["constraint_violations"])


def test_a_forbidden_class_and_an_excluded_building_block_are_enforced(library):
    verifier = RouteVerifier(library)
    forbidden = make_task(forbidden=("esterification",), variant="forbidden_class")
    assert not verifier.score_submission(forbidden, submit(forbidden, KNOWN), STOCK).valid
    restricted = make_task(excluded=("CC(=O)O",), variant="restricted_stock")
    assert not verifier.score_submission(restricted, submit(restricted, KNOWN), STOCK).valid
    assert verifier.score_submission(restricted, submit(restricted, ACYL_CHLORIDE), STOCK).valid


def test_diversity_needs_distinct_first_disconnections(library):
    task = make_task(min_routes=2, variant="diversity", routes=(KNOWN, ACYL_CHLORIDE))
    verifier = RouteVerifier(library)
    assert not verifier.score_submission(task, submit(task, KNOWN), STOCK).valid
    assert verifier.score_submission(task, submit(task, KNOWN, ACYL_CHLORIDE), STOCK).valid


def test_missing_metadata_costs_structure_but_not_validity(library):
    task = make_task()
    submission = submit(task, KNOWN)
    stripped = copy.deepcopy(submission)
    stripped["routes"][0]["children"][0]["metadata"] = {}
    full = RouteVerifier(library).score_submission(task, submission, STOCK)
    bare = RouteVerifier(library).score_submission(task, stripped, STOCK)
    assert bare.valid
    assert bare.components["structure"] < full.components["structure"]


def test_unparseable_and_empty_submissions_score_zero(library):
    task = make_task()
    verifier = RouteVerifier(library)
    assert verifier.score_submission(task, "{not json", STOCK).reward == 0.0
    assert verifier.score_submission(task, {"routes": []}, STOCK).reward == 0.0
