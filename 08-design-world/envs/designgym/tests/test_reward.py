"""Reward invariants over every task in the pack.

If any of these fail, training against the environment is meaningless:

- replaying the expert trajectory scores exactly 1.0, in both modes;
- the HTML surface expresses the ground truth losslessly;
- an empty canvas scores 0;
- the score falls monotonically as a reconstruction gets worse, along every
  axis we care about (position, size, font, colour, missing and extra elements).
"""

from __future__ import annotations

import random

import pytest

from designgym.core import expert, ops
from designgym.core.doc import DesignDoc
from designgym.core.html_profile import doc_to_html, html_to_doc

from .conftest import needs_pack

pytestmark = needs_pack


def _score(pack, task, doc, mode="reference"):
    return pack.grade(task, doc, mode)


def _replay(pack, task):
    gt = pack.gt_doc(task)
    ctx = ops.OpContext(pack.fonts, pack.asset_ctx(task))
    doc = DesignDoc(width=gt.width, height=gt.height)
    for call in expert.tool_calls(gt):
        if call["name"] != "submit":
            doc, _ = ops.apply(doc, call["name"], call["arguments"], ctx)
    return doc, ctx


def _all_tasks(pack):
    return [pack.at(s, i) for s in pack.splits() for i in range(pack.count(s))]


def test_expert_scores_one_on_every_task(pack):
    bad = []
    for task in _all_tasks(pack):
        doc, _ = _replay(pack, task)
        for mode in ("reference", "description"):
            g = _score(pack, task, doc, mode)
            if g.total < 0.9999:
                bad.append((task["task_id"], mode, round(g.total, 5)))
    assert not bad, bad[:10]


def test_html_surface_is_lossless_on_every_task(pack):
    bad = []
    for task in _all_tasks(pack):
        gt = pack.gt_doc(task)
        ctx = ops.OpContext(pack.fonts, pack.asset_ctx(task))
        parsed, warnings = html_to_doc(doc_to_html(gt), gt.width, gt.height)
        doc, _ = ops.apply(DesignDoc(width=gt.width, height=gt.height), "replace_document",
                           {"document": parsed}, ctx)
        g = _score(pack, task, doc)
        if g.total < 0.9999 or warnings:
            bad.append((task["task_id"], round(g.total, 5), warnings[:1]))
    assert not bad, bad[:10]


def test_blank_canvas_scores_zero(pack):
    for task in _all_tasks(pack)[::7]:
        gt = pack.gt_doc(task)
        assert _score(pack, task, DesignDoc(width=gt.width, height=gt.height)).total == 0.0


def _jitter(doc, px, rng):
    d = doc.copy_deep()
    for el in d.elements:
        el.x += rng.choice([-1, 1]) * px
        el.y += rng.choice([-1, 1]) * px
    return d


def test_score_falls_with_position_error(pack):
    for task in _all_tasks(pack)[::11]:
        doc, _ = _replay(pack, task)
        scale = max(doc.width, doc.height) / 1000
        scores = [_score(pack, task, _jitter(doc, px * scale, random.Random(0))).total
                  for px in (0, 5, 20, 60, 150)]
        assert all(a > b for a, b in zip(scores, scores[1:], strict=False)), (task["task_id"], scores)
        assert scores[0] == pytest.approx(1.0)


def test_score_falls_with_each_kind_of_mistake(pack):
    checked = 0
    for task in _all_tasks(pack)[::5]:
        doc, ctx = _replay(pack, task)
        texts = [e for e in doc.elements if e.kind == "text"]
        images = [e for e in doc.elements if e.kind == "image"]
        if not texts or not images:
            continue
        base = _score(pack, task, doc).total
        t = texts[0]

        def with_update(doc=doc, t=t, ctx=ctx, task=task, **props):
            d, _ = ops.apply(doc, "update_element", {"element_id": t.id, **props}, ctx)
            return _score(pack, task, d).total

        other_font = next(f for f in pack.fonts.families() if f != t.font_family)
        assert with_update(font_family=other_font) < base
        assert with_update(font_size=t.font_size * 1.5) < base
        assert with_update(color="#00ff00" if (t.color or "") != "#00ff00" else "#ff0000") < base
        assert with_update(text=t.text + " extra words") < base

        missing, _ = ops.apply(doc, "delete_element", {"element_id": images[0].id}, ctx)
        assert _score(pack, task, missing).total < base

        spam = doc
        for _ in range(5):
            spam, _ = ops.apply(spam, "add_shape", {"x": 0, "y": 0, "width": 10, "height": 10}, ctx)
        assert _score(pack, task, spam).total < base
        checked += 1
    assert checked >= 10


def test_wrong_asset_is_penalised(pack):
    for task in _all_tasks(pack)[::9]:
        doc, ctx = _replay(pack, task)
        imgs = [e for e in doc.elements if e.kind == "image"]
        ids = [a["id"] for a in task["assets"]]
        if not imgs or len(ids) < 2:
            continue
        el = imgs[0]
        other = next(a for a in ids if a != el.asset_id)
        d, _ = ops.apply(doc, "update_element", {"element_id": el.id, "asset_id": other}, ctx)
        assert _score(pack, task, d).structure < _score(pack, task, doc).structure


def test_description_mode_is_more_tolerant(pack):
    for task in _all_tasks(pack)[::13]:
        doc, _ = _replay(pack, task)
        off = _jitter(doc, 0.05 * max(doc.width, doc.height), random.Random(1))
        assert _score(pack, task, off, "description").total > _score(pack, task, off, "reference").total


def test_descriptions_mention_every_asset_and_text(pack):
    for task in _all_tasks(pack)[::3]:
        desc = task["description"]
        gt = pack.gt_doc(task)
        for el in gt.elements:
            if el.kind == "image":
                assert f"asset {el.asset_id}" in desc
            elif el.kind == "text":
                assert el.display_text().replace("\n", " / ") in desc


def test_invisible_text_box_width_is_not_scored(pack):
    """Widening a left-aligned text box changes nothing on screen, so nothing in the score."""
    checked = 0
    for task in _all_tasks(pack)[::4]:
        doc, ctx = _replay(pack, task)
        t = next((e for e in doc.elements if e.kind == "text" and e.align == "left" and not e.angle), None)
        if t is None:
            continue
        wide, _ = ops.apply(doc, "update_element", {"element_id": t.id, "width": t.width * 1.6}, ctx)
        assert _score(pack, task, wide).total == pytest.approx(1.0, abs=1e-3), task["task_id"]
        checked += 1
    assert checked >= 10


def test_right_aligned_width_does_move_text(pack):
    for task in _all_tasks(pack):
        doc, ctx = _replay(pack, task)
        t = next((e for e in doc.elements if e.kind == "text" and e.align == "right"), None)
        if t is None:
            continue
        wide, _ = ops.apply(doc, "update_element", {"element_id": t.id, "width": t.width * 1.6}, ctx)
        assert _score(pack, task, wide).total < 0.999
        return
    pytest.skip("no right-aligned text in the pack")


def test_lookalike_asset_beats_unrelated_asset(pack):
    checked = 0
    for task in _all_tasks(pack):
        ids = [a["id"] for a in task["assets"]]
        if len(ids) < 3:
            continue
        doc, ctx = _replay(pack, task)
        sim = pack.asset_similarity(task)
        img = next((e for e in doc.elements if e.kind == "image"), None)
        others = sorted((sim(img.asset_id, o), o) for o in ids if o != img.asset_id)
        if others[-1][0] - others[0][0] < 0.2:
            continue
        near, _ = ops.apply(doc, "update_element", {"element_id": img.id, "asset_id": others[-1][1]}, ctx)
        far, _ = ops.apply(doc, "update_element", {"element_id": img.id, "asset_id": others[0][1]}, ctx)
        assert _score(pack, task, near).structure > _score(pack, task, far).structure
        checked += 1
        if checked >= 8:
            break
    assert checked >= 3


def test_small_offsets_are_tolerated_large_ones_are_not(pack):
    for task in _all_tasks(pack)[::17]:
        doc, _ = _replay(pack, task)
        scale = max(doc.width, doc.height) / 1000
        small = _score(pack, task, _jitter(doc, 3 * scale, random.Random(2))).total
        large = _score(pack, task, _jitter(doc, 40 * scale, random.Random(2))).total
        assert small > 0.85, (task["task_id"], small)
        assert large < 0.7, (task["task_id"], large)


def _split_multiline(pack, doc):
    """The same render, built as one text box per line."""
    from designgym.core.text import layout

    out = []
    for e in doc.elements:
        if e.kind == "text" and len(e.lines) > 1 and not e.angle and not e.runs:
            lay = layout(e, pack.fonts)
            for k, (ln, text) in enumerate(zip(lay.lines, e.lines, strict=False)):
                out.append(e.model_copy(update={"id": f"{e.id}_{k}", "text": text, "y": e.y + ln.top}))
        else:
            out.append(e)
    return doc.model_copy(update={"elements": out})


def test_v2_scores_text_line_by_line(pack):
    """Reward v2: one multi-line box and a stack of one-line boxes that look the same score the same."""
    checked = 0
    for task in _all_tasks(pack):
        gt = pack.gt_doc(task)
        split = _split_multiline(pack, gt)
        if len(split.elements) == len(gt.elements):
            continue
        checked += 1
        assert pack.grade(task, gt, version=2).total == pytest.approx(1.0)
        assert pack.grade(task, split, version=2).total > 0.99
        assert pack.grade(task, split, version=1).total < 0.99  # v1 is kept as it was
        if checked >= 5:
            break
    assert checked, "no multi-line text in the pack"


def test_v2_still_penalises_mistakes(pack):
    task = next(t for t in _all_tasks(pack) if any(e.kind == "text" for e in pack.gt_doc(t).elements))
    gt = pack.gt_doc(task)
    no_text = gt.model_copy(update={"elements": [e for e in gt.elements if e.kind != "text"]})
    wrong = gt.model_copy(deep=True)
    for e in wrong.elements:
        if e.kind == "text":
            e.color, e.runs = "#ff00ff", None
    assert pack.grade(task, no_text, version=2).total < 0.8
    assert pack.grade(task, wrong, version=2).total < pack.grade(task, gt, version=2).total - 0.02


def _colour_blobs(pack, task, gt, k=16):
    """k x k rectangles painted with the target's average colours: looks right from afar, rebuilds nothing."""
    from designgym.core.doc import Element

    px = pack.target_pixels(task, 256)
    h, w = px.shape[:2]
    els = []
    for i in range(k):
        for j in range(k):
            cell = px[i * h // k:(i + 1) * h // k, j * w // k:(j + 1) * w // k].reshape(-1, px.shape[2]).mean(0)
            fill = "#" + "".join(f"{int(c):02x}" for c in cell[:3])
            els.append(Element(id=f"b{i}_{j}", kind="shape", shape="rect", fill=fill, x=j * gt.width / k,
                               y=i * gt.height / k, width=gt.width / k + 1, height=gt.height / k + 1))
    return gt.model_copy(update={"elements": els})


def _dump(gt, kind, rng):
    """Every element of one kind, the right content scattered anywhere."""
    els = [e.model_copy(update={"x": rng.uniform(0, gt.width * 0.7), "y": rng.uniform(0, gt.height * 0.7)})
           for e in gt.elements if e.kind == kind]
    return gt.model_copy(update={"elements": els})


def test_v3_expert_scores_one_and_blank_scores_zero(pack):
    for task in _all_tasks(pack)[::3]:
        doc, _ = _replay(pack, task)
        assert pack.grade(task, doc, version=3).total == pytest.approx(1.0), task["task_id"]
        gt = pack.gt_doc(task)
        assert pack.grade(task, DesignDoc(width=gt.width, height=gt.height), version=3).total == 0.0


def test_v3_closes_the_exploits_v1_pays_for(pack):
    """Colour patches, scattered copy and scattered assets earned ~0.25 under v1, as much as a weak honest
    attempt. Under v3 the pixel terms only count once the structure is there."""
    rng = random.Random(0)
    v1, v3 = [], []
    for task in _all_tasks(pack)[::13]:
        gt = pack.gt_doc(task)
        for doc in (_colour_blobs(pack, task, gt), _dump(gt, "text", rng), _dump(gt, "image", rng)):
            v1.append(pack.grade(task, doc, version=1).total)
            v3.append(pack.grade(task, doc, version=3).total)
    assert sum(v1) / len(v1) > 0.15  # the hole v3 closes
    assert sum(v3) / len(v3) < 0.08, sum(v3) / len(v3)
    assert max(v3) < 0.3, max(v3)


def test_v3_falls_with_position_error_and_credits_rough_placement(pack):
    """Coarse-to-fine geometry: still strictly falling with error, but everything ~10% off (a hand-placed
    GUI attempt) keeps real credit instead of collapsing toward zero."""
    for task in _all_tasks(pack)[::11]:
        doc, _ = _replay(pack, task)
        scale = max(doc.width, doc.height) / 1000
        scores = [pack.grade(task, _jitter(doc, px * scale, random.Random(0)), version=3).total
                  for px in (0, 5, 20, 60, 150)]
        assert all(a > b for a, b in zip(scores, scores[1:], strict=False)), (task["task_id"], scores)
        assert scores[0] == pytest.approx(1.0)
        assert scores[3] > 0.2, (task["task_id"], scores)


def test_v3_wrong_font_and_colour_costs_more_than_v1(pack):
    checked = 0
    for task in _all_tasks(pack)[::7]:
        gt = pack.gt_doc(task)
        if not any(e.kind == "text" for e in gt.elements):
            continue
        wrong = gt.model_copy(deep=True)
        for e in wrong.elements:
            if e.kind == "text":
                e.font_family, e.color, e.runs = "Roboto", "#000000", None
        drop1 = 1 - pack.grade(task, wrong, version=1).total
        drop3 = 1 - pack.grade(task, wrong, version=3).total
        assert drop3 >= drop1 - 1e-9, (task["task_id"], drop1, drop3)
        checked += 1
    assert checked >= 3


def test_v3_leaves_description_mode_alone(pack):
    task = _all_tasks(pack)[5]
    doc = _jitter(pack.gt_doc(task), 30, random.Random(1))
    assert pack.grade(task, doc, "description", version=3).total == pack.grade(task, doc, "description", version=1).total


def test_attempt_on_another_canvas_size_is_graded(pack):
    """A real app can read back a canvas of another size (a resized board): grading renders the attempt on
    the target's canvas instead of failing on mismatched renders."""
    task = _all_tasks(pack)[0]
    gt = pack.gt_doc(task)
    resized = gt.model_copy(update={"width": gt.width + 37, "height": gt.height - 11})
    assert _score(pack, task, resized).total > 0.5
