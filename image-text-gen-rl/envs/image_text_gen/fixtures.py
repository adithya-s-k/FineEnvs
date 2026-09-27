"""Synthetic, labelled text renders for tests, smoke runs and verifier calibration.

Each variant knows exactly what a faithful literal reader should report, so a verifier
can be scored against ground truth: a broken glyph must come back as MALFORMED, a typo
must stay a typo, and injected gibberish must appear as extra text.
"""

import csv
import io
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

from .server.scoring import MALFORMED

VARIANTS = (
    "clean", "typo_swap", "missing_char", "doubled_char", "wrong_case",
    "malformed_glyph", "extra_text", "blank",
)
FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Supplemental/Georgia.ttf",
    "/System/Library/Fonts/Supplemental/Courier New Bold.ttf",
    "/System/Library/Fonts/Supplemental/Impact.ttf",
    "/System/Library/Fonts/Supplemental/Brush Script.ttf",
    "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
)
PALETTES = (
    ((250, 240, 220), (25, 25, 25)),
    ((20, 30, 60), (245, 210, 80)),
    ((235, 245, 235), (160, 30, 40)),
    ((40, 40, 40), (250, 250, 250)),
)
GIBBERISH = ("QRTVN", "LOREM SIPUM", "Xhq Vwrp", "ZAKKO")


def fonts():
    return [path for path in FONT_CANDIDATES if Path(path).is_file()]


def _font(path, size):
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def _editable_positions(text):
    letters = [i for i, c in enumerate(text) if c.isalpha()]
    return letters or [i for i, c in enumerate(text) if not c.isspace()]


def variant_text(target, variant, rng):
    """(text to draw, what a literal reader should transcribe, index of a broken glyph)."""
    if variant in {"clean", "blank"}:
        return target, ("" if variant == "blank" else target), None
    if variant == "extra_text":
        return target, target + "\n" + rng.choice(GIBBERISH), None
    positions = _editable_positions(target)
    if not positions:
        return target, target, None
    i = rng.choice(positions)
    if variant == "typo_swap":
        j = next((k for k in (i + 1, i - 1) if 0 <= k < len(target) and target[k] != target[i]), None)
        if j is None:
            return target, target, None
        a, b = sorted((i, j))
        drawn = target[:a] + target[b] + target[a] + target[b + 1:]
        return drawn, drawn, None
    if variant == "missing_char":
        drawn = target[:i] + target[i + 1:]
        return drawn, drawn, None
    if variant == "doubled_char":
        drawn = target[:i] + target[i] + target[i:]
        return drawn, drawn, None
    if variant == "wrong_case":
        drawn = target.swapcase()
        return drawn, drawn, None
    if variant == "malformed_glyph":
        return target, target[:i] + MALFORMED + target[i + 1:], i
    raise ValueError(f"Unknown variant {variant!r}")


def _wrap(draw, text, font, width):
    words, lines, line = text.split(" "), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def render(target, variant="clean", seed=0, size=(1024, 640), font_path=None, font_size=None):
    """Return (PIL image, expected literal transcription)."""
    rng = random.Random(f"{seed}:{variant}:{target}")
    background, ink = rng.choice(PALETTES)
    image = Image.new("RGB", size, background)
    drawn, expected, broken = variant_text(target, variant, rng)
    if variant == "blank":
        return image, expected
    draw = ImageDraw.Draw(image)
    font_path = font_path if font_path is not None else (rng.choice(fonts()) if fonts() else None)
    font_size = font_size or max(40, min(120, int(size[0] * 1.6 / max(8, len(drawn)))))
    font = _font(font_path, font_size)
    lines = _wrap(draw, drawn, font, size[0] * 0.86)
    while len(lines) * font_size * 1.3 > size[1] * 0.7 and font_size > 28:
        font_size = int(font_size * 0.85)
        font = _font(font_path, font_size)
        lines = _wrap(draw, drawn, font, size[0] * 0.86)
    y = (size[1] - len(lines) * font_size * 1.3) / 2
    offset = 0  # character offset of the current line within `drawn`
    for line in lines:
        x = (size[0] - draw.textlength(line, font=font)) / 2
        draw.text((x, y), line, fill=ink, font=font)
        if broken is not None and offset <= broken < offset + len(line):
            k = broken - offset
            left = x + draw.textlength(line[:k], font=font)
            right = x + draw.textlength(line[: k + 1], font=font)
            top, bottom = y + font_size * 0.15, y + font_size * 1.05
            # Erase the glyph and scrawl strokes that are not any letter.
            draw.rectangle((left, top, right, bottom), fill=background)
            w = max(4, font_size // 9)
            draw.line((left, top, right, bottom), fill=ink, width=w)
            draw.arc((left - w, top, right + w, bottom), 200, 340, fill=ink, width=w)
        offset += len(line) + 1
        y += font_size * 1.3
    if variant == "extra_text":
        noise = expected.split("\n", 1)[1]
        small = _font(font_path, max(28, font_size // 2))
        draw.text((size[0] * 0.06, size[1] * 0.84), noise, fill=ink, font=small)
    return image, expected


def png_bytes(image, readings=None):
    """PNG bytes; `readings` are stored for FixtureVerifier (one string per reader)."""
    info = PngImagePlugin.PngInfo()
    if readings is not None:
        info.add_text("fixture_readings", "\x1f".join(readings))
    buffer = io.BytesIO()
    image.save(buffer, "PNG", pnginfo=info)
    return buffer.getvalue()


FIXTURE_ROWS = (
    ("Happy Birthday", 'A cake with "Happy Birthday" written in icing.'),
    ("OPEN 24 HOURS", 'A neon sign in a diner window reading "OPEN 24 HOURS".'),
    ("Why not?", 'A chalkboard with the question "Why not?" in white chalk.'),
    ("bon appétit", 'A menu card that says "bon appétit" in cursive.'),
    ("Keep Calm", 'A red poster with the words "Keep Calm" in bold.'),
    ("Around the World", 'A globe with the phrase "Around the World" written above.'),
)


def write_source(directory, train_repeat=4):
    """Write leffff-format CSVs so the real catalog code runs offline."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    columns = ["", "prompt", "text", "text_len", "id"] + [
        f"v{i}_qwen_ocr_levenstein_score" for i in range(1, 6)
    ]

    def rows(start, count, suffix):
        for n in range(count):
            text, prompt = FIXTURE_ROWS[n % len(FIXTURE_ROWS)]
            row_id = start + n
            yield [row_id, f"{prompt} Scene {suffix}{n}.", text, len(text), row_id] + [0.9] * 5

    for name, start, count, suffix in (
        ("data_with_ocr_reward_train.csv", 0, len(FIXTURE_ROWS) * train_repeat, "t"),
        ("data_with_ocr_reward_test.csv", 1000, len(FIXTURE_ROWS), "x"),
    ):
        with open(directory / name, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            writer.writerows(rows(start, count, suffix))
    return directory
