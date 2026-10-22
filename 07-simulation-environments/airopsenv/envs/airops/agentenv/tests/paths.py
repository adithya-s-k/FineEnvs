"""Where the tests' days are: the plugin's fixture days (core/tests/fixtures/tasks.jsonl), four days of b6-v1, three
of b6-v2 and two of b6-v3, a recorded run of each env and the task files the published runs played, all committed, so
no test needs the fetched packs."""

from pathlib import Path

TESTS = Path(__file__).resolve().parent
ENV = TESTS.parents[1]
FIXTURES = ENV / "core" / "tests" / "fixtures" / "tasks.jsonl"
PACK = TESTS / "fixtures" / "b6-v1"
OCC_PACK = ENV / "core" / "tests" / "fixtures" / "b6-v2"
OCC3_PACK = ENV / "core" / "tests" / "fixtures" / "b6-v3"
RUNS = TESTS / "fixtures" / "runs"
BUNDLES = TESTS / "fixtures" / "bundles"
AGENTS = TESTS.parent / "agents"
