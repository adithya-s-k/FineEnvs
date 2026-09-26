# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the watercolour environment.

Nothing here touches the network. The vision judge is exercised through a stub
client, which is also the only way to test the paths that matter most: a judge
that raises, and a judge whose two presentation orders disagree.

The tests that need a browser are marked and skipped when Playwright has no
Chromium installed, so the source, judging and reward logic stays testable
without a 100MB download.
"""

from __future__ import annotations

import asyncio
import pathlib
import re

import pytest

import sys

_ENV_ROOT = str(pathlib.Path(__file__).resolve().parents[1])
_OPENENV = str(pathlib.Path(__file__).resolve().parents[1] / "openenv")
for _p in (_OPENENV, _ENV_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

pytest.importorskip("playwright", reason="playwright is not installed")

from models import WatercolourAction
from core.gate import MIN_PAINT_FRACTION, run_gate
from core.pairwise_judge import (
    Comparison,
    JudgeReport,
    load_pool,
    PairwiseJudge,
)
from core.prompt import system_prompt as _prompt
from core.render import SketchRenderer
from rubric import build_rubric
from core.scoring import (
    QUALITY_WEIGHT,
    Evaluation,
    GATE_WEIGHT,
    JUDGE_WEIGHT,
    length_score,
    LENGTH_WEIGHT,
    MIN_LENGTH_TOKENS,
    RUNAWAY_LENGTH_TOKENS,
    TARGET_LENGTH_TOKENS,
)
from core.sketch_source import (
    KNOWN_CALLS,
    extract_sketch,
    inspect_source,
    SourceError,
)
from core.tasks import make_task, sample_task, SUBJECTS
from watercolour_environment import WatercolourEnvironment

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# Fixtures the gate should let through, and the violation each of the rest
# should be rejected for. `throws_in_draw` is admitted on purpose: it painted
# something before it died, and how good that something is belongs to the judge.
ADMISSIBLE = (
    "good_painting",
    "weak_painting",
    "throws_in_draw",
    "setup_only",
    "allowlist_only",
)
REJECTED = {
    "cheat_bare_p5": "bare_primitives",
    "cheat_external_image": "external_access",
    "cheat_text_label": "text_label",
    "not_webgl": "not_webgl",
    "blank_canvas": "no_painting_calls",
    "truncated": "truncated",
}


def fixture(name: str) -> str:
    """Return the source of a fixture by stem."""
    for suffix in (".js", ".txt"):
        path = FIXTURES / f"{name}{suffix}"
        if path.exists():
            return path.read_text()
    raise FileNotFoundError(name)


class TestExtraction:
    """Getting a sketch out of a reply."""

    def test_plain_source(self):
        assert "createCanvas" in extract_sketch(fixture("good_painting"))

    def test_fenced_source(self):
        reply = "Here you go:\n```js\nfunction setup(){}\n```\nEnjoy."
        assert extract_sketch(reply).strip() == "function setup(){}"

    def test_refusal_is_not_a_sketch(self):
        # "Sorry, I cannot draw images" mentions `draw` without defining it,
        # and an earlier substring check accepted it.
        with pytest.raises(SourceError):
            extract_sketch("Sorry, I cannot draw images.")

    def test_prose_is_not_a_sketch(self):
        with pytest.raises(SourceError):
            extract_sketch("A watercolour is made by letting pigment bleed.")


class TestUnknownBrushMethods:
    """A method that is not on `brush` throws, so it is caught before rendering.

    It used to be caught after: the throw stopped `draw`, the render came back
    with whatever had been laid down first, and the gate reported `blank_canvas`.
    That misnames the cause. Measured on a twelve-sample probe of a 35B, the four
    lowest-coverage submissions were exactly the four that threw, and two of them
    cleared the gate at 0.7% coverage and were paid for a crash.
    """

    def test_an_invented_method_is_caught(self):
        report = inspect_source(
            "async function setup(){ createCanvas(600,600,WEBGL); }\n"
            "function draw(){ brush.lineWidth(3); brush.circle(300,300,20,1); }"
        )
        assert report.unknown_calls == ["lineWidth"]

    def test_helpers_and_globals_are_not_brush_methods(self):
        # The check only reads names called on `brush`, which is why it is safe to
        # act on where the bare-primitive check was not.
        report = inspect_source(
            "async function setup(){ createCanvas(600,600,WEBGL); angleMode(DEGREES); }\n"
            "function petal(a, r){ return Math.cos(a) * r; }\n"
            "function draw(){ translate(0,0); background('#fff'); petal(1,2);"
            " brush.circle(300,300,20,1); }"
        )
        assert report.unknown_calls == []

    def test_every_method_the_prompt_lists_is_known(self):
        # If the prompt offers a method the inventory does not know, the gate
        # would reject the very sketches it asked for.
        listed = set(re.findall(r"brush\.(\w+)\(", _prompt()))
        assert listed, "no brush methods found in the prompt"
        assert not (listed - KNOWN_CALLS), listed - KNOWN_CALLS

    def test_no_reference_source_is_rejected(self):
        from core.pool import pool_dir, pool_sources

        sources = [
            pool_sources() / f"{png.stem}.js"
            for png in pool_dir().glob("*.png")
            if (pool_sources() / f"{png.stem}.js").exists()
        ]
        assert sources, "no reference sources on disk"
        flagged = {
            s.name: inspect_source(s.read_text()).unknown_calls
            for s in sources
            if inspect_source(s.read_text()).unknown_calls
        }
        assert not flagged, flagged


class TestCommentsAreNotCode:
    """The inventory reads code, and prose in a comment is not code.

    Comments were not stripped and the primitive check allows whitespace before
    the paren, so an English sentence counted as a call. Measured on a twelve
    sample probe of a 35B: four rollouts of twelve rejected as `bare_primitives`
    for `// the center point (300, 300)` and `// Outer curve (bulging out)`. A
    third of the run thrown away for being commented, and the penalty grew with
    the model, because a bigger one comments more.
    """

    def test_prose_in_a_comment_is_not_a_primitive_call(self):
        for comment in (
            "// Define the center point (300, 300)",
            "// Outer curve (bulging out)",
            "// Tip point (wide)",
            "/* draws a line (from the centre) */",
        ):
            report = inspect_source(
                "async function setup(){ createCanvas(600,600,WEBGL); }\n"
                f"function draw(){{ {comment}\n brush.circle(300,300,20,1); }}"
            )
            assert report.bare_primitives == [], comment

    def test_a_real_bare_call_is_still_caught(self):
        report = inspect_source(
            "async function setup(){ createCanvas(600,600,WEBGL); }\n"
            "function draw(){ ellipse(300, 300, 50, 50); }"
        )
        assert "ellipse" in report.bare_primitives

    def test_a_bare_call_hiding_behind_a_comment_is_still_caught(self):
        report = inspect_source(
            "async function setup(){ createCanvas(600,600,WEBGL); }\n"
            "function draw(){ // a point (the centre)\n vertex(1, 2); }"
        )
        assert "vertex" in report.bare_primitives

    def test_a_block_comment_does_not_hide_the_code_after_it(self):
        report = inspect_source(
            "async function setup(){ createCanvas(600,600,WEBGL); }\n"
            "function draw(){ /* curve (soft) */ rect(0,0,10,10); }"
        )
        assert "rect" in report.bare_primitives


class TestSourceInventory:
    """What the source says about itself, without running it."""

    def test_counts_real_api_calls(self):
        report = inspect_source(fixture("good_painting"))
        assert "polygon" in report.painting_calls
        assert "fillBleed" in report.supporting_calls
        assert report.unknown_calls == []
        assert report.webgl and report.has_setup and report.has_draw

    def test_brush_calls_are_not_bare_primitives(self):
        # Every painting call shares its name with a p5 primitive, so a naive
        # check flags `brush.rect(...)` as drawing without the library.
        assert inspect_source(fixture("good_painting")).bare_primitives == []

    def test_flags_bare_primitives(self):
        assert "ellipse" in inspect_source(fixture("cheat_bare_p5")).bare_primitives

    def test_flags_invented_calls(self):
        # `brush.noLoop` does not exist. It has its own fixture rather than
        # sharing one with `throws_in_draw`, because under the restricted
        # allowlist a sketch cannot both clear the gate and invent a method.
        assert "noLoop" in inspect_source(fixture("invented_call")).unknown_calls

    def test_flags_external_access(self):
        assert (
            "loadImage"
            in inspect_source(fixture("cheat_external_image")).external_access
        )

    def test_flags_unbalanced_braces(self):
        assert not inspect_source(fixture("truncated")).balanced


class TestTasks:
    """Subject selection."""

    def test_prompt_names_the_subject(self):
        task = make_task("two ripe plums")
        assert "two ripe plums" in task.prompt
        assert task.task_id == "two_ripe_plums"

    def test_sampling_is_seeded(self):
        assert sample_task(seed=3).subject == sample_task(seed=3).subject

    def test_every_subject_makes_a_task(self):
        assert len({make_task(s).task_id for s in SUBJECTS}) == len(SUBJECTS)


class TestAllowlist:
    """The system prompt is a closed list, and it has to be able to paint."""

    def test_prompt_lists_only_methods_that_exist(self):
        import re

        bundle = (FIXTURES.parent / "vendor/p5.brush.js").read_text()
        real = set(re.findall(r"\bt\.([A-Za-z][A-Za-z0-9_]*)\s*=", bundle))
        listed = set(re.findall(r"brush\.(\w+)\(", _prompt()))
        assert listed <= real, listed - real

    def test_prompt_carries_the_custom_shape_path(self):
        # All three published sketches from the trained model use these, and a
        # list without them paints triangles where the task wants petals.
        for name in ("beginShape", "vertex", "endShape"):
            assert f"brush.{name}" in _prompt()

    def test_the_restricted_list_gives_up_hatching(self):
        # This is a cost, recorded rather than hidden. Frames from the posted
        # video show dense hatching carrying half the visual language of the
        # finished paintings, and the restricted list cannot produce any:
        # `brush.hatchStyle` takes a brush name, and `brush.hatch` on its own
        # throws "No brush or color set". Naming a brush is what the whole
        # restriction exists to avoid, since ten of the twenty-one errors in
        # the census were invented brush and field names. Hatching goes so that
        # the error family cannot be written.
        for name in ("hatch", "hatchStyle", "noHatch", "set", "field"):
            assert f"brush.{name}" not in _prompt()

    def test_prompt_uses_their_coordinate_convention(self):
        # Their sketches translate to a top-left origin and work in degrees. A
        # model told the centred convention reasons about space in a different
        # frame than the one the reward was built on.
        assert "translate(-width / 2, -height / 2)" in _prompt()
        assert "angleMode(DEGREES)" in _prompt()
        assert "-300" not in _prompt()

    def test_prompt_describes_the_pool_not_the_trained_output(self):
        # This asserted "whole canvas" and that a dark ground was allowed, both
        # read off the video of the *trained* model's output. The reward points at
        # the reference pool, and the pool is 117 single flowers on pale paper, so
        # the prompt was describing a target the judge does not reward. Asserting
        # the corrected intent so the wrong one cannot come back.
        text = _prompt()
        assert "pale cream" in text
        assert "edge to edge" in text
        assert "do not paint on a dark ground" in text
        assert "whole canvas" not in text

    def test_prompt_has_no_worked_example(self):
        import re

        # The prompt carries a skeleton, because their sketches share a specific
        # structure and there is no way to convey structure without showing it.
        # What it must not carry is a worked painting: their finding is about
        # documentation and examples that show the library off, and an earlier
        # version of this test banned the skeleton itself, which measured the
        # wrong thing.
        skeleton = _prompt()[_prompt().index("async function setup") :]
        skeleton = skeleton[: skeleton.index("So: angles are in degrees")]
        painting = set(re.findall(r"brush\.(\w+)\(", skeleton)) - {"scaleBrushes"}
        assert painting == set(), painting
        assert "```" not in _prompt()

    def test_a_sketch_using_only_the_allowlist_is_admissible(self):
        import re
        from core.sketch_source import inspect_source

        listed = set(re.findall(r"brush\.(\w+)\(", _prompt()))
        report = inspect_source(fixture("allowlist_only"))
        used = set(report.painting_calls) | set(report.supporting_calls)
        assert used <= listed, used - listed
        assert report.unknown_calls == []
        assert report.bare_primitives == []


class TestReferencePool:
    """The pool is the reward function, so it has to be there."""

    def test_pool_is_not_empty(self):
        assert len(load_pool()) >= 2

    def test_references_are_pngs(self):
        assert all(png[:8] == b"\x89PNG\r\n\x1a\n" for _, png in load_pool())


class StubVisionClient:
    """A judge that always names a fixed slot, and counts its calls."""

    model = "stub"

    def __init__(self, winner: str = "A"):
        self._winner = winner
        self.calls = 0

    async def compare(self, prompt: str, first: bytes, second: bytes) -> str:
        self.calls += 1
        return '{"winner": "%s", "reason": "stub"}' % self._winner


class _RaisingClient(StubVisionClient):
    """A judge that cannot answer."""

    async def compare(self, prompt: str, first: bytes, second: bytes) -> str:
        raise RuntimeError("provider down")


class _PositionBiasedClient(StubVisionClient):
    """A judge with no opinion that always picks whichever came first.

    This is what both real judges did on a pair of equally poor paintings, and
    the reason every comparison runs in both orders.
    """

    async def compare(self, prompt: str, first: bytes, second: bytes) -> str:
        self.calls += 1
        return '{"winner": "A", "reason": "first"}'


class TestPairwiseJudge:
    """Comparative scoring."""

    def _pool(self):
        return [("ref_a.png", b"a"), ("ref_b.png", b"b")]

    def test_winning_both_orders_scores_one(self):
        # The stub names slot A always, so the submission wins when shown
        # first and loses when shown second. That is the biased case.
        judge = PairwiseJudge(StubVisionClient("A"), pool=self._pool(), references=2)
        report = asyncio.run(judge.score(b"png", seed=0))
        assert report.score == 0.5

    def test_position_bias_becomes_a_tie(self):
        client = _PositionBiasedClient()
        judge = PairwiseJudge(client, pool=self._pool(), references=2)
        report = asyncio.run(judge.score(b"png", seed=0))
        assert report.score == 0.5
        assert all(c.score == 0.5 for c in report.comparisons)

    def test_two_calls_per_reference(self):
        client = StubVisionClient("A")
        judge = PairwiseJudge(client, pool=self._pool(), references=2)
        asyncio.run(judge.score(b"png", seed=0))
        assert client.calls == 4

    def test_unavailable_judge_is_not_a_zero(self):
        judge = PairwiseJudge(_RaisingClient(), pool=self._pool(), references=2)
        report = asyncio.run(judge.score(b"png", seed=0))
        assert report.score == 0.0
        assert not report.available

    def test_empty_pool_is_unavailable(self):
        judge = PairwiseJudge(StubVisionClient(), pool=[], references=2)
        report = asyncio.run(judge.score(b"png"))
        assert not report.available

    def test_reference_sampling_is_seeded(self):
        judge = PairwiseJudge(StubVisionClient(), pool=self._pool(), references=1)
        first = asyncio.run(judge.score(b"png", seed=7)).comparisons[0].reference
        second = asyncio.run(judge.score(b"png", seed=7)).comparisons[0].reference
        assert first == second


class TestTheTwoRewardPaths:
    """The rubric tree and the observation's own reward must agree.

    They are computed twice from the same evaluation: `Evaluation.reward` in
    `scoring`, and the `Rubric` tree in `rubric`, which is the one that lands on
    the observation and therefore the one a trainer optimises. Adding a dense
    term to `scoring` alone left them 0.30 apart on the same painting, with the
    feedback string quoting a reward the policy was never given, and nothing
    caught it until a probe printed both numbers side by side.
    """

    def test_the_tree_carries_every_term_the_reward_does(self):
        import inspect

        import rubric as rubric_mod
        from core.scoring import Evaluation

        tree = inspect.getsource(rubric_mod.build_rubric)
        # Every weight `Evaluation.reward` multiplies by has to appear in the tree.
        for weight in (
            "GATE_WEIGHT",
            "LENGTH_WEIGHT",
            "JUDGE_WEIGHT",
            "QUALITY_WEIGHT",
        ):
            assert weight in inspect.getsource(Evaluation), weight
            assert weight in tree, f"{weight} is in the reward but not in the tree"

    def test_they_agree_on_the_same_evaluation(self):
        from core.gate import GateResult
        from rubric import build_rubric
        from core.scoring import Evaluation
        from core.tasks import Task

        class FakeRender:
            paint_fraction = 0.24
            finished = True
            errors: list[str] = []
            png = b""

        class FakeSource:
            source = "x" * 1200  # part way up the length ramp

        class FakeQuality:
            score = 0.7
            available = True

        class FakeJudge:
            score = 0.5
            available = True
            comparisons: list[object] = []

            def to_dict(self):
                return {}

        evaluation = Evaluation(
            task=Task(subject="a peach hibiscus", prompt="p", task_id="t"),
            gate=GateResult(passed=True, source=FakeSource(), render=FakeRender()),
            judge=FakeJudge(),
            judge_enabled=True,
            quality=FakeQuality(),
        )

        class Observation:
            gate_passed = True
            length_score = evaluation.length_score
            judge_score = evaluation.judge_score
            quality_score = evaluation.quality_score

        assert build_rubric()(None, Observation()) == pytest.approx(evaluation.reward)


class TestRevisions:
    """The multi-turn episode, and the guarantee that it changes nothing by default."""

    def test_reset_can_set_zero_references_and_revisions(self):
        env = WatercolourEnvironment(
            subject="two ripe plums",
            enable_judge=False,
            references=4,
            revisions=2,
        )
        env.reset(references=0, revisions=0)
        assert env._episode_references == 0
        assert env._episode_revisions == 0

    def test_the_default_episode_is_still_one_shot(self):
        # Six runs trained against a single-shot episode and the reward was only
        # ever measured against that shape. Revisions are opt-in so none of that
        # has to be re-established.
        env = WatercolourEnvironment(subject="two ripe plums", enable_judge=False)
        env.reset(subject="two ripe plums")
        observation = env.step(WatercolourAction(response=fixture("good_painting")))
        assert observation.done
        assert observation.revisions_left == 0
        assert env.state.submitted
        asyncio.run(env.close())

    def test_a_budget_keeps_the_episode_open(self):
        env = WatercolourEnvironment(
            subject="two ripe plums", enable_judge=False, revisions=2
        )
        env.reset(subject="two ripe plums", revisions=2)
        seen = []
        for _ in range(3):
            obs = env.step(WatercolourAction(response=fixture("good_painting")))
            seen.append((obs.done, obs.revisions_left))
        assert seen == [(False, 2), (False, 1), (True, 0)]
        asyncio.run(env.close())

    def test_the_critique_never_names_the_references_or_the_score(self):
        # The line that separates a harder task from a rigged one. With a shared
        # seed the whole group faces the same eight references, so telling the
        # policy how many it beat tells it who they were, and it would learn to
        # beat a draw instead of to paint.
        env = WatercolourEnvironment(
            subject="two ripe plums", enable_judge=False, revisions=1
        )
        env.reset(subject="two ripe plums", revisions=1)
        obs = env.step(WatercolourAction(response=fixture("good_painting")))
        assert obs.critique
        for leak in ("beat", "reward", "judge", "love_", "okay_", "meh_"):
            assert leak not in obs.critique.lower(), leak
        asyncio.run(env.close())


class TestReferenceSampling:
    """The draw has to be reproducible and it has to return what was asked for."""

    def _judge(self):
        from core.pairwise_judge import (
            DEFAULT_MIX,
            PairwiseJudge,
            load_pool,
        )

        judge = PairwiseJudge.__new__(PairwiseJudge)
        judge._pool = load_pool()
        judge._mix = DEFAULT_MIX
        return judge

    @pytest.mark.parametrize("wanted", [1, 2, 3, 4, 5, 8, 12])
    def test_the_draw_returns_what_was_asked_for(self, wanted):
        # Rounding each tier's share independently silently dropped references:
        # asking for two returned one, because 0.5 rounds down in all three tiers.
        import random

        drawn = self._judge()._sample(random.Random(0), wanted)
        assert len(drawn) == wanted

    def test_the_same_seed_draws_the_same_references(self):
        # This is what makes a GRPO group comparable. Without it every rollout
        # faces its own opponents, and measured on one painting scored six times
        # the draw alone moves the judge score from 0.250 to 0.625, which GRPO
        # reads as one rollout being better than another.
        import random

        judge = self._judge()
        first = judge._sample(random.Random(7), 8)
        again = judge._sample(random.Random(7), 8)
        other = judge._sample(random.Random(8), 8)
        assert [n for n, _ in first] == [n for n, _ in again]
        assert [n for n, _ in first] != [n for n, _ in other]

    def test_the_seed_arrives_through_reset_not_step(self):
        # OpenEnv's client accepts `**kwargs` on `step` and documents them as
        # ignored, so they never reach the server. The environment therefore has
        # to take the seed on `reset` and remember it.
        from watercolour_environment import (
            WatercolourEnvironment,
        )

        env = WatercolourEnvironment(subject="two ripe plums", enable_judge=False)
        env.reset(subject="two ripe plums", seed=41)
        assert env._episode_seed == 41
        asyncio.run(env.close())


class TestDomains:
    """A domain that cannot be trained on must not be selectable."""

    def test_every_selectable_domain_has_a_pool(self):
        # The jellyfish domain pointed at a directory that did not exist, and
        # `load_pool` globbed it to an empty list, so the judge scored zero on
        # every comparison and 0.60 of the reward was dead with nothing said. This
        # is the test that was missing, not the fix.
        from core.domains import DOMAINS
        from core.pairwise_judge import load_pool

        for name, domain in DOMAINS.items():
            assert load_pool(domain.pool), name

    def test_an_empty_pool_is_an_error(self, tmp_path):
        from core.pairwise_judge import load_pool

        with pytest.raises(FileNotFoundError):
            load_pool(tmp_path)


class TestQuality:
    """The dense term that fills the slot HPSv3 holds in the write-up."""

    def test_a_mark_is_rescaled_to_the_unit_interval(self):
        # The rubric sums terms in [0, 1] at their weights, so a mark out of ten
        # cannot go in raw: a 10 would contribute 3.0 on a 0.30 weight.
        from core.quality import QualityReport

        assert QualityReport(score=0.0, mark=1, available=True).score == 0.0
        assert QualityReport(score=1.0, mark=10, available=True).score == 1.0

    def test_an_unscored_painting_is_not_a_bad_painting(self):
        # The distinction that cost three steps of a run to notice was missing on
        # the pairwise side. A model that could not answer leaves `available`
        # false, and a zero from that is not a verdict about the painting.
        from core.quality import QualityReport

        blank = QualityReport()
        assert blank.score == 0.0
        assert not blank.available

    def test_the_scorer_rescales_and_clamps(self):
        # The reply is model-generated text against a schema that says "integer",
        # which does not say "between one and ten".
        from core.quality import QualityScorer

        class Client:
            def __init__(self, reply):
                self.reply = reply

            async def describe(self, prompt, png, schema):
                return self.reply

        for reply, expected in (
            ('{"score": 1, "reason": ""}', 0.0),
            ('{"score": 10, "reason": ""}', 1.0),
            ('{"score": 99, "reason": ""}', 1.0),
            ('{"score": -4, "reason": ""}', 0.0),
        ):
            report = asyncio.run(
                QualityScorer(Client(reply), criteria="c").score(b"png")
            )
            assert report.score == pytest.approx(expected), reply
            assert report.available

    def test_a_broken_reply_leaves_the_term_unscored(self):
        from core.quality import QualityScorer

        class Client:
            async def describe(self, prompt, png, schema):
                return "not json at all"

        report = asyncio.run(QualityScorer(Client(), criteria="c").score(b"png"))
        assert not report.available
        assert report.score == 0.0

    def test_the_prompt_carries_the_domain_criteria(self):
        # The mark and the pairwise verdict have to be answering the same
        # question, or the two vision terms pull in different directions.
        from core.quality import QualityScorer

        scorer = QualityScorer(object(), criteria="petals around a centre")
        assert "petals around a centre" in scorer._prompt

    def test_the_anchors_name_what_each_mark_looks_like(self):
        # An unanchored "score from 1 to 10" compresses everything into 7-9, which
        # is the dead signal this term exists to fix, in a different disguise.
        # Measured with the anchors: pool love 9.0, okay 8.4, meh 7.4, and an
        # untrained policy's own output 3.4 where the pairwise judge gave it zero.
        from core.quality import _RUBRIC

        for anchor in ("1 is", "4 is", "7 is", "10 is"):
            assert anchor in _RUBRIC, anchor


class TestLengthRamp:
    """The ramp that replaced the binary length band."""

    def test_a_stub_scores_nothing(self):
        assert length_score("function setup(){}") == 0.0

    def test_a_runaway_scores_nothing(self):
        # Their 13,500-token sketches were a problem to be compressed, not a goal.
        assert length_score("x" * (RUNAWAY_LENGTH_TOKENS * 4 + 40)) == 0.0

    def test_writing_more_always_pays_up_to_the_target(self):
        # The point of the change. The band this replaced returned one for
        # everything between 150 and 1200 tokens, and measured output sits inside
        # it: small models write 570 to 1256 tokens and a VLM 700 to 1300. A term that is
        # one for every rollout adds nothing to a GRPO group, so the only signal
        # about elaboration in the rubric was doing no work at all.
        marks = [length_score("x" * 4 * n) for n in (200, 600, 1200, 2000, 3000)]
        assert marks == sorted(marks)
        assert marks[0] < marks[-1]
        assert marks[-1] == 1.0

    def test_the_target_is_theirs(self):
        # "a code length ramp targeting around 3,000 tokens", from the write-up.
        assert TARGET_LENGTH_TOKENS == 3000
        assert MIN_LENGTH_TOKENS < TARGET_LENGTH_TOKENS < RUNAWAY_LENGTH_TOKENS

    def test_the_reference_pool_is_far_below_the_target(self):
        """The known tension in this term, recorded rather than hidden.

        The test this replaces asserted every reference scored 1.0, on the argument
        that a term must admit the things the reward points at. Under a ramp that
        is no longer the right property, and asserting it would pin the target to
        our own pool. The references run 202 to 315 tokens because they were
        hand-authored; theirs came from frontier models writing rich sketches. So
        the ramp pulls towards elaboration that our own pool does not exemplify,
        and the judge pulls towards paintings that look like the pool. They agree
        in intent, since more elaborate code makes a richer painting, but a sketch
        can be padded without painting better, and at 0.05 that is what the farming
        is worth. Revisit with the pool work, not by moving the target.
        """
        from core.pool import pool_dir, pool_sources

        sources = [
            pool_sources() / f"{png.stem}.js"
            for png in pool_dir().glob("*.png")
            if (pool_sources() / f"{png.stem}.js").exists()
        ]
        assert sources, "no reference sources on disk"
        marks = [length_score(s.read_text()) for s in sources]
        assert max(marks) < 1.0, "a reference already reaches the target"
        assert all(m > 0.0 for m in marks), "a reference scores nothing at all"


def free_credit(name: str = "good_painting") -> float:
    """Return what a real sketch earns before any vision call.

    Was `GATE_WEIGHT + LENGTH_WEIGHT`, which held while the length check was a
    band returning one for anything a small model writes. It is a ramp towards
    3,000 tokens now, so a real sketch collects a fraction of that weight and the
    assertions have to compute it rather than assume it.

    Measured on the extracted sketch, not the fixture, because that is what the
    gate carries and what the reward scores. The fence around the code block is
    only a few characters, which put the two four parts in a million apart and
    failed the tolerance rather than the eye.
    """
    return GATE_WEIGHT + LENGTH_WEIGHT * length_score(extract_sketch(fixture(name)))


class TestReward:
    """How the layers combine."""

    def _evaluation(self, gate_passed: bool, judge: JudgeReport | None):
        from core.gate import GateResult
        from core.sketch_source import inspect_source

        return Evaluation(
            task=make_task("two ripe plums"),
            gate=GateResult(
                passed=gate_passed,
                violations=[] if gate_passed else ["x"],
                # Extracted first, the way `run_gate` does it. Scoring the
                # fixture instead measures the fence around the code block as
                # part of the sketch.
                source=inspect_source(extract_sketch(fixture("good_painting"))),
            ),
            judge=judge,
            judge_enabled=judge is not None,
        )

    def test_gate_failure_zeroes_a_good_judge_score(self):
        report = JudgeReport(score=1.0, comparisons=[], available=True)
        assert self._evaluation(False, report).reward == 0.0

    def test_clearing_the_gate_earns_something_on_its_own(self):
        # The point of the weighted sum over a hard gate: a policy that
        # paints legibly but beats nothing still has a reward to climb from,
        # so a GRPO group has variance on day one.
        report = JudgeReport(score=0.0, comparisons=[], available=True)
        assert self._evaluation(True, report).reward == pytest.approx(free_credit())

    def test_judge_carries_most_of_the_weight(self):
        report = JudgeReport(score=1.0, comparisons=[], available=True)
        assert self._evaluation(True, report).reward == pytest.approx(
            free_credit() + JUDGE_WEIGHT
        )
        assert JUDGE_WEIGHT > GATE_WEIGHT + LENGTH_WEIGHT

    def test_a_perfect_judge_alone_does_not_reach_one(self):
        # Renamed from test_hpsv3_absence_caps_the_reward_below_one, which asserted
        # 0.70 because HPSv3's 0.30 was unimplemented. That slot now holds an
        # absolute mark at their 0.30 and the weights sum to one, so the cap it
        # described is gone. What remains true is that the judge alone cannot reach
        # 1.0: this fixture was never marked, so that term pays nothing.
        report = JudgeReport(score=1.0, comparisons=[], available=True)
        expected = free_credit() + JUDGE_WEIGHT
        assert self._evaluation(True, report).reward == pytest.approx(expected)
        assert expected < 1.0

    def test_no_judge_leaves_only_the_free_components(self):
        assert self._evaluation(True, None).reward == pytest.approx(free_credit())

    def test_unavailable_judge_does_not_inflate_the_reward(self):
        report = JudgeReport(score=0.0, comparisons=[], available=False)
        evaluation = self._evaluation(True, report)
        assert evaluation.reward == pytest.approx(free_credit())
        assert not evaluation.judged

    def test_feedback_counts_only_resolved_comparisons(self):
        comparisons = [
            Comparison("ref_a.png", "A", "B", 1.0, []),
            Comparison("ref_b.png", None, None, 0.0, []),
        ]
        report = JudgeReport(score=1.0, comparisons=comparisons, available=True)
        assert "beat 1 of 1" in self._evaluation(True, report).feedback

    def test_rubric_sums_its_terms_at_their_weights(self):
        # Renamed from test_rubric_agrees_with_the_evaluation, which it did not
        # test: it compares the tree against a restatement of the same formula, so
        # it passes whenever the tree is self-consistent and stayed green while the
        # tree and `Evaluation.reward` were 0.30 apart. The comparison against the
        # other implementation is in `TestTheTwoRewardPaths`.
        rubric = build_rubric()

        class Observation:
            gate_passed = True
            length_score = 1.0
            judge_score = 0.5
            quality_score = 1.0

        expected = GATE_WEIGHT + LENGTH_WEIGHT + JUDGE_WEIGHT * 0.5 + QUALITY_WEIGHT
        assert rubric(None, Observation()) == pytest.approx(expected)

    def test_the_weights_still_sum_to_one(self):
        # WeightedSum requires it. The absolute mark holds the weight HPSv3 carries
        # in the write-up, at their number, so the arithmetic is theirs exactly.
        assert GATE_WEIGHT + LENGTH_WEIGHT + JUDGE_WEIGHT + QUALITY_WEIGHT == 1.0

    def test_rubric_gate_is_absolute(self):
        rubric = build_rubric()

        class Observation:
            gate_passed = False
            length_score = 1.0
            judge_score = 1.0
            paint_fraction = 0.24

        assert rubric(None, Observation()) == 0.0


browser = pytest.mark.skipif(
    not list((pathlib.Path.home() / "Library/Caches/ms-playwright").glob("chromium*"))
    and not list(pathlib.Path("/ms-playwright").glob("chromium*")),
    reason="no Chromium build for Playwright",
)


@browser
class TestGateWithBrowser:
    """The checks that need a real render."""

    @pytest.fixture(scope="class")
    def renderer(self):
        renderer = SketchRenderer()
        yield renderer
        asyncio.run(renderer.close())

    @pytest.mark.parametrize("name", ADMISSIBLE)
    def test_admissible_fixtures_pass(self, renderer, name):
        result = asyncio.run(run_gate(fixture(name), renderer))
        assert result.passed, result.violations
        assert result.render.paint_fraction >= MIN_PAINT_FRACTION

    @pytest.mark.parametrize("name", sorted(REJECTED))
    def test_rejected_fixtures_are_caught(self, renderer, name):
        result = asyncio.run(run_gate(fixture(name), renderer))
        assert not result.passed
        assert REJECTED[name] in result.violations

    def test_a_refusal_is_rejected_before_rendering(self, renderer):
        result = asyncio.run(run_gate(fixture("no_sketch"), renderer))
        assert result.violations == ["no_sketch_in_response"]
        assert result.render is None

    def test_a_setup_only_sketch_paints_and_is_admitted(self, renderer):
        # It does all its work in `setup` and defines no `draw`, which paints
        # the same picture. Requiring both entry points rejected these, and
        # waiting for a non-existent `draw` to stop looping burned the whole
        # render deadline on a painting that was finished in seconds.
        result = asyncio.run(run_gate(fixture("setup_only"), renderer))
        assert result.passed, result.violations
        assert result.render.finished
        assert result.render.elapsed_ms < 30_000

    def test_a_sketch_that_throws_is_still_scored(self, renderer):
        # It never reaches its own `noLoop()`, so it is cut off rather than
        # finished, and it is admitted on whatever it painted first.
        result = asyncio.run(run_gate(fixture("throws_in_draw"), renderer))
        assert result.passed
        assert not result.render.finished
        assert result.render.errors


@browser
class TestEnvironment:
    """The episode contract."""

    def test_reset_hands_over_the_api_reference(self):
        env = WatercolourEnvironment(subject="two ripe plums", enable_judge=False)
        observation = env.reset()
        assert "two ripe plums" in observation.prompt
        # Without this the model has no way to know the library's surface, and
        # no model tested up to 30B emitted a real call without it.
        assert "brush.fillBleed" in observation.system_prompt
        assert observation.reward is None and not observation.done
        asyncio.run(env.close())

    def test_step_before_reset_is_an_error(self):
        env = WatercolourEnvironment(enable_judge=False)
        with pytest.raises(RuntimeError):
            env.step(WatercolourAction(response=fixture("good_painting")))

    def test_episode_ends_on_the_one_submission(self):
        env = WatercolourEnvironment(subject="two ripe plums", enable_judge=False)
        env.reset()
        observation = env.step(WatercolourAction(response=fixture("good_painting")))
        assert observation.done and observation.gate_passed
        assert observation.reward == pytest.approx(
            free_credit() + QUALITY_WEIGHT * observation.quality_score
        )
        assert env.state.submitted
        asyncio.run(env.close())

    def test_stub_judge_drives_the_reward(self):
        judge = PairwiseJudge(
            StubVisionClient("A"), pool=[("ref.png", b"x")], references=1
        )
        env = WatercolourEnvironment(subject="two ripe plums", judge=judge)
        env.reset()
        observation = env.step(WatercolourAction(response=fixture("good_painting")))
        # Slot A always wins, so the submission takes one order and drops the
        # other: a tie, scoring half.
        assert observation.judge_score == 0.5
        assert observation.reward == pytest.approx(
            free_credit()
            + JUDGE_WEIGHT * 0.5
            + QUALITY_WEIGHT * observation.quality_score
        )
        assert observation.judged
        asyncio.run(env.close())

    def test_returned_image_is_optional(self):
        env = WatercolourEnvironment(
            subject="two ripe plums", enable_judge=False, return_image=True
        )
        env.reset()
        observation = env.step(WatercolourAction(response=fixture("good_painting")))
        assert observation.image_png_base64
        asyncio.run(env.close())

    def test_reset_can_ask_for_the_image(self):
        # The client cannot reach the constructor of a deployed environment, and
        # `step` drops its kwargs by design, so `reset` is the only way in.
        env = WatercolourEnvironment(subject="two ripe plums", enable_judge=False)
        env.reset(return_image=True)
        observation = env.step(WatercolourAction(response=fixture("good_painting")))
        assert observation.image_png_base64
        asyncio.run(env.close())

    def test_episode_overrides_do_not_persist(self):
        env = WatercolourEnvironment(subject="two ripe plums", enable_judge=False)
        env.reset(return_image=True)
        env.reset()
        observation = env.step(WatercolourAction(response=fixture("good_painting")))
        assert observation.image_png_base64 is None
        asyncio.run(env.close())

    def test_reset_can_lower_the_reference_count(self):
        client = StubVisionClient("A")
        judge = PairwiseJudge(
            client, pool=[("a.png", b"a"), ("b.png", b"b")], references=2
        )
        env = WatercolourEnvironment(subject="two ripe plums", judge=judge)
        env.reset(references=1)
        env.step(WatercolourAction(response=fixture("good_painting")))
        # One reference in both orders is two calls, not four. This is the knob
        # that halves the cost of a training run.
        assert client.calls == 2
        asyncio.run(env.close())
        asyncio.run(env.close())
