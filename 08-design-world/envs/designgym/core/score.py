"""The reward: how close a submitted document is to the ground truth.

Two signals, because each is blind where the other sees:

- **Structure** matches submitted elements to ground-truth elements one-to-one
  (Hungarian assignment) and scores every pair on content, geometry and, for
  text, typography. It is dense, it is readable off a trajectory, and it tells
  you *which* element is wrong. Unmatched elements on either side count against
  the score, so spamming elements does not buy coverage.
- **Visual** compares `render(submission)` with `render(ground_truth)` through
  the same renderer: multi-scale SSIM plus colour-layout error, weighted toward
  coarse scales and normalised so that an empty canvas scores zero rather than
  whatever fraction of the target happens to be white.

Reference mode (the model sees the target image) uses tight tolerances.
Description mode (the model only reads a description) loosens geometry, since
"top left, large" cannot pin a box to the pixel, and adds a coverage term: were
all the assets and all the words actually used?

Every constant below is a tolerance a person can reason about: `pos_sigma` is a
fraction of the canvas diagonal, `size_sigma` a log-ratio.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment

from .colors import delta_e
from .doc import DesignDoc, Element


@dataclass(frozen=True)
class ScoreConfig:
    mode: str
    pos_sigma: float
    size_sigma: float
    pixel_size: int
    w_structure: float
    w_visual: float
    w_coverage: float
    w_order: float
    w_background: float
    # (pyramid level, term, weight); level 0 is the 256 px render, each level
    # halves it by area averaging, so level 3 is 32 px.
    visual_terms: tuple = ()


# Precision lives in the structure term. The visual term answers "does it look
# like the target", so it is weighted toward coarse scales, where a shift of a
# few pixels is invisible, as it is to a person looking at the design.
REFERENCE = ScoreConfig("reference", 0.025, 0.12, 256, 0.60, 0.30, 0.00, 0.05, 0.05,
                        ((1, "ssim", 0.15), (2, "ssim", 0.30), (3, "ssim", 0.25),
                         (3, "color", 0.30)))
DESCRIPTION = ScoreConfig("description", 0.12, 0.45, 256, 0.50, 0.15, 0.25, 0.05, 0.05,
                          ((3, "ssim", 0.30), (4, "ssim", 0.30), (4, "color", 0.40)))
CONFIGS = {"reference": REFERENCE, "description": DESCRIPTION}
IOU_PAD = 0.004  # fraction of the canvas diagonal

# Version 3 (reference mode), calibrated against a pairwise visual judge on frontier and Qwen attempts over all
# three surfaces (results/reward-design). Geometry is coarse-to-fine: half the tight kernel above, half a loose
# one, so a hand-placed element in the right region earns credit while exact placement is still needed for 1.0.
# Pixel-level terms only count once the structure is there, which closes colour-patch and dump exploits.
V3_COARSE_MIX = 0.5
V3_COARSE_POS = 0.15     # fraction of the canvas diagonal
V3_COARSE_SIZE = 0.6     # log-ratio
V3_TEXT_WEIGHT = 3.0     # a piece of copy counts three times an image or shape in structure
V3_GATE = 0.3            # visual, order and background scale with min(1, structure / V3_GATE)


@dataclass
class PairScore:
    gt_id: str
    matched_id: str | None
    score: float
    content: float = 0.0
    geometry: float = 0.0
    style: float | None = None
    note: str = ""


@dataclass
class Grade:
    total: float
    structure: float
    visual: float
    coverage: float
    order: float
    background: float
    mode: str
    n_gt: int
    n_submitted: int
    n_matched: int
    elements: list[PairScore] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = asdict(self)
        for key in ("total", "structure", "visual", "coverage", "order", "background"):
            d[key] = round(d[key], 4)
        for e in d["elements"]:
            for k in ("score", "content", "geometry"):
                e[k] = round(e[k], 4)
            if e["style"] is not None:
                e["style"] = round(e["style"], 4)
        return d

    def report(self) -> str:
        head = (
            f"reward {self.total:.3f} = structure {self.structure:.3f}, visual {self.visual:.3f}"
            + (f", coverage {self.coverage:.3f}" if self.mode == "description" else "")
            + f", order {self.order:.3f}, background {self.background:.3f}"
        )
        lines = [head, f"matched {self.n_matched}/{self.n_gt} target elements, "
                       f"{self.n_submitted} submitted"]
        worst = sorted(self.elements, key=lambda p: p.score)[:4]
        for p in worst:
            if p.score < 0.9:
                lines.append(f"  {p.gt_id}: {p.score:.2f} {p.note}".rstrip())
        if self.unmatched:
            lines.append(f"  extra elements: {', '.join(self.unmatched)}")
        return "\n".join(lines)


# -------------------------------------------------------------- helpers


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def text_similarity(a: str, b: str) -> float:
    na = " ".join(a.split()).casefold()
    nb = " ".join(b.split()).casefold()
    if not na and not nb:
        return 1.0
    return max(0.0, 1.0 - levenshtein(na, nb) / max(len(na), len(nb)))


def _box(el: Element) -> tuple[float, float, float, float]:
    return el.x, el.y, el.x + el.width, el.y + el.height


def iou(a: Element, b: Element) -> float:
    ax0, ay0, ax1, ay1 = _box(a)
    bx0, by0, bx1, by1 = _box(b)
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0


def _angle_gap(a: float, b: float) -> float:
    d = abs(a - b) % 360
    return min(d, 360 - d)


def _iou_box(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def geometry(a: Element, g: Element, diag: float, cfg: ScoreConfig, boxes=None) -> float:
    """Position, size and overlap of what is drawn.

    `boxes` maps element ids to the box that is actually rendered; for text
    that is the extent of the glyphs, not the declared box, whose width is
    invisible for left-aligned copy and should not be scored.
    """
    ba = (boxes or {}).get(("a", a.id)) or (a.x, a.y, a.x + a.width, a.y + a.height)
    bg = (boxes or {}).get(("g", g.id)) or (g.x, g.y, g.x + g.width, g.y + g.height)
    wa, ha = max(1e-3, ba[2] - ba[0]), max(1e-3, ba[3] - ba[1])
    wg, hg = max(1e-3, bg[2] - bg[0]), max(1e-3, bg[3] - bg[1])
    dist = math.dist(((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2), ((bg[0] + bg[2]) / 2, (bg[1] + bg[3]) / 2))
    pos = math.exp(-dist / (cfg.pos_sigma * diag))
    size = math.exp(-(abs(math.log(wa / wg)) + abs(math.log(ha / hg))) / (2 * cfg.size_sigma))
    rot = math.exp(-_angle_gap(a.angle, g.angle) / 10.0)
    # IoU on boxes padded by a sliver of the canvas: plain IoU punishes a
    # four-pixel nudge of a four-pixel dot as hard as a wholesale miss.
    pad = IOU_PAD * diag
    pa = (ba[0] - pad, ba[1] - pad, ba[2] + pad, ba[3] + pad)
    pg = (bg[0] - pad, bg[1] - pad, bg[2] + pad, bg[3] + pad)
    base = (_iou_box(pa, pg) + pos + size) / 3
    return base * (0.85 + 0.15 * rot)


def geometry_v3(a: Element, g: Element, diag: float, cfg: ScoreConfig, boxes=None) -> float:
    """Coarse-to-fine geometry, scaled by the square root of the (coarse-to-fine) position match: an element in
    the wrong place is the wrong element however well it is sized."""
    m = V3_COARSE_MIX
    coarse = replace(cfg, pos_sigma=V3_COARSE_POS, size_sigma=V3_COARSE_SIZE)
    base = (1 - m) * geometry(a, g, diag, cfg, boxes) + m * geometry(a, g, diag, coarse, boxes)
    ba = (boxes or {}).get(("a", a.id)) or (a.x, a.y, a.x + a.width, a.y + a.height)
    bg = (boxes or {}).get(("g", g.id)) or (g.x, g.y, g.x + g.width, g.y + g.height)
    dist = math.dist(((ba[0] + ba[2]) / 2, (ba[1] + ba[3]) / 2), ((bg[0] + bg[2]) / 2, (bg[1] + bg[3]) / 2))
    pos = (1 - m) * math.exp(-dist / (cfg.pos_sigma * diag)) + m * math.exp(-dist / (V3_COARSE_POS * diag))
    return base * math.sqrt(pos)


def text_box(el: Element, fonts) -> tuple[float, float, float, float]:
    from .text import layout

    lay = layout(el, fonts)
    x0 = min((ln.x for ln in lay.lines if ln.width > 0), default=0.0)
    x1 = max((ln.x + ln.width for ln in lay.lines if ln.width > 0), default=el.width)
    return el.x + x0, el.y, el.x + max(x1, x0 + 1.0), el.y + lay.height


def _dominant_color(el: Element) -> str:
    runs = el.styled_runs()
    weights: dict[str, int] = {}
    for r in runs:
        weights[r.color] = weights.get(r.color, 0) + len(r.text.strip())
    return max(weights.items(), key=lambda kv: kv[1])[0] if weights else (el.color or "#000000")


def typography(a: Element, g: Element, salient: bool = False) -> tuple[float, list[str]]:
    notes = []
    font = 1.0 if (a.font_family or "").lower() == (g.font_family or "").lower() else 0.0
    if font < 1:
        notes.append(f"font {a.font_family} vs {g.font_family}")
    size = math.exp(-abs(math.log(a.font_size / g.font_size)) / 0.1)
    if size < 0.7:
        notes.append(f"size {a.font_size:.0f} vs {g.font_size:.0f}")
    de = delta_e(_dominant_color(a), _dominant_color(g))
    color = math.exp(-de / 25.0)
    if color < 0.7:
        notes.append(f"color {_dominant_color(a)} vs {_dominant_color(g)}")
    align = 1.0 if a.align == g.align else 0.0
    weight = 1.0 - 0.5 * ((a.bold != g.bold) + (a.italic != g.italic))
    case = 1.0 if a.uppercase == g.uppercase or a.display_text() == g.display_text() else 0.5
    spacing = math.exp(-abs(a.letter_spacing - g.letter_spacing) / max(1.0, 0.05 * g.font_size))
    lines = 1.0
    if len(g.lines) > 1:
        lines = math.exp(-abs(a.line_height - g.line_height) / 0.15)
    rest = (0.22 * size + 0.08 * align + 0.06 * weight + 0.04 * case + 0.05 * spacing + 0.05 * lines) / 0.50
    if salient:  # font and colour are what a reader notices first: they scale everything else
        return (0.5 + 0.5 * font) * (0.4 + 0.6 * color) * (0.5 + 0.5 * rest), notes
    return 0.28 * font + 0.22 * color + 0.50 * rest, notes


def pair_score(a: Element, g: Element, diag: float, cfg: ScoreConfig, boxes=None,
               asset_sim=None, style_weight: float = 0.38, version: int = 1) -> PairScore:
    geometry_fn = geometry_v3 if version >= 3 else geometry
    if a.kind == "text" and g.kind == "text":
        content = text_similarity(a.display_text(), g.display_text())
        if content < 0.25:
            return PairScore(g.id, a.id, 0.0, content, 0.0, None, "text differs")
        geo = geometry_fn(a, g, diag, cfg, boxes)
        style, notes = typography(a, g, salient=version >= 3)
        opacity = math.exp(-abs(a.opacity - g.opacity) / 0.2)
        if version >= 3:  # well-set copy in the wrong place is not the copy the target has
            score = content * geo * ((1 - style_weight) + style_weight * style) * (0.9 + 0.1 * opacity)
        else:
            score = content * ((1 - style_weight) * geo + style_weight * style) * (0.9 + 0.1 * opacity)
        if content < 0.95:
            notes.insert(0, f'text "{a.text[:20]}" vs "{g.text[:20]}"')
        return PairScore(g.id, a.id, score, content, geo, style, "; ".join(notes))
    if a.kind == "text" or g.kind == "text":
        return PairScore(g.id, a.id, 0.0, 0.0, 0.0, None, "kind mismatch")
    if a.kind == "image" and g.kind == "image":
        if a.asset_id == g.asset_id:
            content = 1.0
        else:
            # A look-alike asset (two near-identical clouds) is a small mistake,
            # an unrelated one a large one.
            sim = asset_sim(a.asset_id, g.asset_id) if asset_sim else 0.0
            content = 0.15 + 0.6 * sim
    else:
        content = 0.2  # a drawn shape standing in for a rendered graphic
    geo = geometry_fn(a, g, diag, cfg, boxes)
    opacity = math.exp(-abs(a.opacity - g.opacity) / 0.2)
    note = "" if content == 1.0 else f"asset {a.asset_id or a.shape} vs {g.asset_id}"
    return PairScore(g.id, a.id, content * geo * (0.9 + 0.1 * opacity), content, geo, None, note)


# ---------------------------------------------------------------- pixels


def _gray(rgb: np.ndarray) -> np.ndarray:
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def ssim(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> float:
    x, y = _gray(a.astype(np.float64)), _gray(b.astype(np.float64))
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mx, my = gaussian_filter(x, sigma), gaussian_filter(y, sigma)
    sxx = gaussian_filter(x * x, sigma) - mx * mx
    syy = gaussian_filter(y * y, sigma) - my * my
    sxy = gaussian_filter(x * y, sigma) - mx * my
    num = (2 * mx * my + c1) * (2 * sxy + c2)
    den = (mx * mx + my * my + c1) * (sxx + syy + c2)
    return float(np.clip((num / den).mean(), 0.0, 1.0))


def pyramid(img: np.ndarray, levels: int) -> list[np.ndarray]:
    """Area-averaged pyramid. Rendering small directly aliases (a one-pixel
    canvas shift can flip a 32 px thumbnail's pixels by 150 levels); halving a
    256 px render does not."""
    out = [img.astype(np.float64)]
    for _ in range(levels):
        x = out[-1]
        h, w = (x.shape[0] // 2) * 2, (x.shape[1] // 2) * 2
        if h < 4 or w < 4:
            break
        x = x[:h, :w]
        out.append((x[0::2, 0::2] + x[1::2, 0::2] + x[0::2, 1::2] + x[1::2, 1::2]) / 4)
    return out


def _term(kind: str, a: np.ndarray, b: np.ndarray) -> float:
    if kind == "ssim":
        return ssim(a, b)
    mae = float(np.abs(a - b).mean() / 255.0)
    return math.exp(-mae / 0.12)


def visual_score(attempt: np.ndarray, target: np.ndarray, blank: np.ndarray,
                 terms=REFERENCE.visual_terms) -> float:
    """Multi-scale visual similarity, rescaled so an empty canvas scores 0.

    Each term is normalised against the blank canvas on its own scale, then
    the terms are averaged with their weights.
    """
    if attempt.shape != target.shape:
        raise ValueError("renders differ in size")
    depth = max(level for level, _, _ in terms)
    pa, pt, pb = pyramid(attempt, depth), pyramid(target, depth), pyramid(blank, depth)
    total = weight = 0.0
    for level, kind, w in terms:
        level = min(level, len(pt) - 1)
        v = _term(kind, pa[level], pt[level])
        v0 = _term(kind, pb[level], pt[level])
        score = (1.0 if v >= 0.999 else 0.0) if v0 >= 0.999 else (v - v0) / (1.0 - v0)
        total += w * float(np.clip(score, 0.0, 1.0))
        weight += w
    return total / weight if weight else 0.0


# ----------------------------------------------------------------- grade


def _overlap(a: Element, b: Element) -> bool:
    ax0, ay0, ax1, ay1 = _box(a)
    bx0, by0, bx1, by1 = _box(b)
    return min(ax1, bx1) > max(ax0, bx0) and min(ay1, by1) > max(ay0, by0)


def _split_runs(el: Element) -> list[list]:
    """The element's runs cut at its line breaks: one run list per line."""
    per_line: list[list] = [[]]
    for r in el.runs or []:
        parts = r.text.split("\n")
        for k, part in enumerate(parts):
            if k:
                per_line.append([])
            if part:
                per_line[-1].append(r.model_copy(update={"text": part}))
    return per_line


def explode_lines(elements: list[Element], fonts) -> list[Element]:
    """Reward v2 scores text line by line: a three-line box and three stacked one-line boxes that
    render identically are the same design, and building it either way should score the same."""
    from .text import layout

    out = []
    for e in elements:
        if e.kind != "text" or e.angle or len(e.lines) < 2 or fonts is None:
            out.append(e)
            continue
        lay = layout(e, fonts)
        runs = _split_runs(e) if e.runs else None
        for k, (line, text) in enumerate(zip(lay.lines, e.lines, strict=False)):
            if not text.strip():
                continue
            out.append(e.model_copy(update={
                "id": f"{e.id}#l{k}", "text": text, "y": e.y + line.top, "height": lay.line_box,
                "runs": (runs[k] or None) if runs and k < len(runs) else None}))
    return out


def grade(
    attempt: DesignDoc,
    target: DesignDoc,
    mode: str = "reference",
    attempt_pixels: np.ndarray | None = None,
    target_pixels: np.ndarray | None = None,
    blank_pixels: np.ndarray | None = None,
    fonts=None,
    asset_sim=None,
    version: int = 1,
) -> Grade:
    cfg = CONFIGS[mode]
    if version >= 3 and mode != "reference":
        version = 1  # v3 is calibrated for reference mode only; description mode keeps v1
    # a cropped picture is compared as the whole picture it shows part of (Element.uncropped)
    sub = [e.uncropped() for e in attempt.elements if e.visible and e.opacity > 0.01]
    gts = [e.uncropped() for e in target.elements if e.visible and e.opacity > 0.01]
    if version == 2:
        sub, gts = explode_lines(sub, fonts), explode_lines(gts, fonts)
    diag = math.hypot(target.width, target.height)
    boxes = {}
    if fonts is not None:
        # Only unrotated text: a rotated box's width moves its pivot, so there
        # the declared box is what determines where the glyphs land.
        boxes.update({("a", e.id): text_box(e, fonts) for e in sub if e.kind == "text" and not e.angle})
        boxes.update({("g", e.id): text_box(e, fonts) for e in gts if e.kind == "text" and not e.angle})

    if not sub:
        return Grade(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mode, len(gts), 0, 0,
                     [PairScore(g.id, None, 0.0, note="missing") for g in gts], [])

    matrix = np.zeros((len(sub), len(gts)))
    pairs: dict[tuple[int, int], PairScore] = {}
    for i, a in enumerate(sub):
        for j, g in enumerate(gts):
            # v2 weighs typography as much as placement: wrong-colour or wrong-font copy is as
            # visible to a reader as misplaced copy.
            ps = pair_score(a, g, diag, cfg, boxes, asset_sim, style_weight=0.5 if version == 2 else 0.38,
                            version=version)
            matrix[i, j] = ps.score
            pairs[(i, j)] = ps
    rows, cols = linear_sum_assignment(-matrix) if gts else ([], [])
    matched = {}
    for i, j in zip(rows, cols, strict=True):
        if matrix[i, j] > 0:
            matched[j] = i
    elements = []
    for j, g in enumerate(gts):
        if j in matched:
            elements.append(pairs[(matched[j], j)])
        else:
            elements.append(PairScore(g.id, None, 0.0, note="missing"))
    used = set(matched.values())
    unmatched = [a.id for i, a in enumerate(sub) if i not in used]
    if version >= 3:
        # Copy weighs more; every unmatched submission counts against the score like an average element.
        weights = [V3_TEXT_WEIGHT if g.kind == "text" else 1.0 for g in gts]
        mean_w = sum(weights) / len(weights) if weights else 1.0
        denom = sum(weights) + len(unmatched) * mean_w
        structure = sum(w * p.score for w, p in zip(weights, elements, strict=True)) / denom if denom else 1.0
    else:
        denom = max(len(gts), len(sub))
        structure = sum(p.score for p in elements) / denom if denom else 1.0

    # Stacking order only matters where elements overlap.
    agree = total = 0
    gt_index = {g.id: k for k, g in enumerate(gts)}
    sub_index = {a.id: k for k, a in enumerate(sub)}
    matched_pairs = [(gts[j], sub[i]) for j, i in matched.items()]
    for p in range(len(matched_pairs)):
        for q in range(p + 1, len(matched_pairs)):
            g1, a1 = matched_pairs[p]
            g2, a2 = matched_pairs[q]
            if not _overlap(g1, g2):
                continue
            total += 1
            same = (gt_index[g1.id] < gt_index[g2.id]) == (sub_index[a1.id] < sub_index[a2.id])
            agree += same
    order = (agree / total if total else 1.0) if matched else 0.0

    background = math.exp(-delta_e(attempt.background, target.background) / 20.0)

    visual = 0.0
    if attempt_pixels is not None and target_pixels is not None and blank_pixels is not None:
        visual = visual_score(attempt_pixels, target_pixels, blank_pixels, cfg.visual_terms)

    coverage = 0.0
    if mode == "description":
        gt_assets = {g.asset_id for g in gts if g.kind == "image"}
        sub_assets = {a.asset_id for a in sub if a.kind == "image"}
        asset_cov = len(gt_assets & sub_assets) / len(gt_assets) if gt_assets else 1.0
        gt_texts = [g.display_text() for g in gts if g.kind == "text"]
        sub_texts = [a.display_text() for a in sub if a.kind == "text"]
        hits = sum(
            1 for t in gt_texts if any(text_similarity(t, s) >= 0.8 for s in sub_texts)
        )
        text_cov = hits / len(gt_texts) if gt_texts else 1.0
        coverage = 0.5 * asset_cov + 0.5 * text_cov

    gate = min(1.0, structure / V3_GATE) if version >= 3 else 1.0
    total_score = (
        cfg.w_structure * structure
        + gate * (cfg.w_visual * visual + cfg.w_order * order + cfg.w_background * background)
        + cfg.w_coverage * coverage
    )
    return Grade(
        total=float(min(1.0, max(0.0, total_score))),
        structure=float(structure),
        visual=float(visual),
        coverage=float(coverage),
        order=float(order),
        background=float(background),
        mode=mode,
        n_gt=len(gts),
        n_submitted=len(sub),
        n_matched=len(matched),
        elements=elements,
        unmatched=unmatched,
    )
