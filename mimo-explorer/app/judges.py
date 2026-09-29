"""The judge prompts, as the judges receive them, with English translations to read them by.

Both prompts are in Chinese and are sent unchanged, so scores stay comparable with Xiaomi's. A
translation is tied to the SHA-256 of the exact original it translates: if the original ever changes
(a new dataset revision, a new vendored rubric), the page shows the original alone rather than a
translation of something else.

Nothing here carries an answer. The General prompt is sent once per check with that check's
expected answer (`pass_anchor`) in it; every place an answer would go is shown as WITHHELD.
"""

from __future__ import annotations

import ast
import hashlib

WITHHELD = "‹expected answer withheld›"
_sha = lambda s: hashlib.sha256(s.encode()).hexdigest()   # noqa: E731


# ── General: one yes/no question per check, from the task's own verify.py ────
GENERAL_SHA = "079dc5191345db36bd0e566ed8e1dfdb6c73855907e3d4b9511f8942d3a99235"   # all 925 tasks ship this one
GENERAL_EN = """You are checking whether a work deliverable meets one **objective checkpoint** (it has one right answer; this is not about taste).
Below are the text extracted from the deliverable's files, and the checkpoint. Judge strictly: met = 1, not met = 0.
**Score only 0 or 1, no partial credit** (partly met = not met). **Rely only on the text evidence given**; if you can't see the evidence, score 0. Don't guess.
The evidence may contain [truncated…] markers or a "⚠ content not extracted" list: if the checkpoint's target content falls exactly in a truncated or unextracted part, score 0 and write "evidence truncated" or "evidence not extracted" in why (so audits can tell these apart; still don't guess a score).

[Deliverable text]
{evidence}

[Checkpoint]
{items}

Output only one JSON object (it may be wrapped in a json code block):
{"results": {"checkpoint id": {"score": 0, "why": "reason, at most 25 characters"}}}"""


def _constant(py: str, name: str) -> str | None:
    try:
        tree = ast.parse(py)
    except SyntaxError:
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == name for t in node.targets) \
                and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            return node.value.value
    return None


def general(verify_py: str, check: dict | None, files: list[str]) -> dict | None:
    """The prompt the judge gets for one model-judged check, filled in the way `_run_llm_sc` fills it,
    with the deliverable's text described and the expected answer withheld."""
    original = _constant(verify_py, "_LLM_PROMPT")
    if not original:
        return None
    names = ", ".join(files) or "answer.md"
    evidence = f"‹the text extracted from {names}, up to 20,000 characters per file›"
    cid, q = (check or {}).get("id", "check_id"), (check or {}).get("question", "")
    fill = lambda p, item: p.replace("{evidence}", evidence).replace("{items}", item)   # noqa: E731
    return {
        "original": fill(original, f"- [{cid}] {q}(满足=1:{WITHHELD})"),
        "english": fill(GENERAL_EN, f"- [{cid}] {q} (met = 1: {WITHHELD})") if _sha(original) == GENERAL_SHA else None,
        "check": cid,
    }


# ── Webdev: one vision call, seven continuous scores ─────────────────────────
WEBDEV_SHA = "a4d3be63029e8fb28b469bf3d188816a7fa238b415749d7ff4aad1ca360b2997"   # eval_rubric.build_prompt(), RUBRIC_ID rva1
WEBDEV_RULES_EN = [
    "Score only what is visible in the full-page screenshot.",
    "Judge each criterion on its own: a strong or weak showing on one must not move the others.",
    "Give each criterion a continuous score from 0 to 1: first pick the band its description matches, then a value "
    "inside the band by how severe, how many and how damaging the problems are. Not just band edges or fixed steps.",
    "In a borderline case, choose the neighbouring band that best matches the visible evidence.",
    "Asset quality may not reach 0.6 without premium assets.",
    "A blank page, a page that badly failed to load, or one that shows nothing scores 0 on every criterion.",
    "Reply with JSON only: the seven scores and a short reason.",
]
WEBDEV_BANDS_EN = {
    "layout_integrity": [
        "Badly broken: core content widely overlaps, is cut off, overflows or can't be read",
        "Several obvious layout errors that significantly hurt use of the main content",
        "The main body is usable, but with one serious or several obvious alignment, overlap or overflow problems",
        "Complete overall, with only a few minor alignment, spacing or edge problems",
        "Complete and stable: no perceptible overlap, misalignment, truncation or overflow",
    ],
    "typography_hierarchy": [
        "Text is generally unreadable; sizes and hierarchy are out of control",
        "Heading and body hierarchy is confused; many clearly wrong sizes, line heights or weights",
        "Readable, but the hierarchy or typographic rhythm has clear defects",
        "A clear hierarchy that reads smoothly, with only a few minor problems",
        "Headings, body and secondary text are distinct, with mature proportions, line heights and weights",
    ],
    "color_harmony": [
        "Colours clash badly, or key text is almost illegible",
        "Many uncoordinated colours or weak contrast, clearly hurting the look or readability",
        "Usable, but the palette, contrast or consistency has clear problems",
        "Harmonious and readable, with only a few minor inconsistencies or contrast problems",
        "A unified, mature palette with clear contrast and no obvious problems",
    ],
    "whitespace": [
        "Extremely crowded or empty; the organisation of the information has basically failed",
        "Many areas clearly too dense or too empty, seriously breaking the rhythm of browsing",
        "Browsable overall, but local density, spacing or section rhythm has clear problems",
        "Whitespace and density are reasonable overall, with only a few minor problems",
        "Whitespace, spacing and section rhythm are natural and consistent; density is well controlled",
    ],
    "content_richness": [
        "Nearly an empty shell that doesn't fill one screen: a title, an empty frame or a little placeholder text",
        "Only one or two simple sections; content is repetitive, vague or mostly placeholder",
        "Some real sections, but too few or too shallow to carry a complete website",
        "Fairly complete sections with concrete content; only a few parts are thin",
        "About four or more fully formed sections with concrete, substantial content, enough for a complete site",
    ],
    "query_fulfillment": [
        "What the screenshot shows basically doesn't implement the brief, or the topic is wrong",
        "Only a few surface elements; most core requirements are missing",
        "The main direction is there, but several key sections, elements or functional views are clearly missing",
        "The main requirements are implemented, with only a few minor omissions or deviations",
        "Within the visible page, the brief's topic, sections, elements and functional views are complete and accurate",
    ],
    "premium_assets": [
        "No real visual assets, or mostly crude emoji, bare placeholders, or low-quality or blurry images",
        "Ordinary or rough assets with no unified design; a little decoration doesn't add up to a finish",
        "Clear, roughly coordinated assets, but mostly ordinary pictures or pure-CSS visuals; nothing premium",
        "High-quality real photos, artistic images, a consistent illustration set or refined mockups; small flaws remain",
        "Outstanding premium assets, a complete system, closely integrated with the layout: mature professional work",
    ],
}
WEBDEV_WHY = [
    "The three parts weigh the same. An earlier version gave visual quality 0.6 and added a static check of the code "
    "(0.1). That check turned out to reward not referencing external assets: the pages with the fewest premium assets "
    "scored highest on it, so it was dropped.",
    "Asset quality separates models most sharply (0.417 against 0.619 between two arms of Xiaomi's comparison, where visual quality "
    "differed by 0.042 and brief fulfilment by 0.054), so it gets a full third instead of being drowned out by a "
    "0.6-weighted visual score.",
]
WEBDEV_TRAINING = ("Training used a different, group-relative reward. A grading service compared the screenshots of "
                   "sibling rollouts on the same brief (in groups of up to 8) and picked among them, then subtracted a "
                   "deduction when a page missed the brief. Delivering nothing was a real 0; a page that failed to render "
                   "was left out of the group. That number only means something next to its siblings, so a single "
                   "rollout here is graded with Xiaomi's evaluation rubric instead. The grading service is not part of "
                   "the release.")


def webdev(query: str) -> dict:
    from .vendor.webdev.eval_rubric import build_prompt

    template = build_prompt()
    current = _sha(template) == WEBDEV_SHA
    return {
        "original": template.format(query=query[:1500]),   # QUERY_CAP: the judge sees the first 1,500 characters
        "query_cap": 1500,
        "rules": WEBDEV_RULES_EN if current else None,
        "bands": WEBDEV_BANDS_EN if current else None,
        "edges": ["0.0–0.2", "0.2–0.4", "0.4–0.6", "0.6–0.8", "0.8–1.0"],
        "why": WEBDEV_WHY,
        "training": WEBDEV_TRAINING,
    }
