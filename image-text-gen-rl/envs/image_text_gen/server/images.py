"""Decode, bound and normalise a submitted image before any verifier sees it.

Hosted Qwen VL endpoints misread text in small images and invent plausible words
instead (a 768x384 render of "Happy Birthday!" came back as "Hilary Pichler"), while
the same image at 2x was transcribed exactly. Every image is therefore resized so its
long side is TARGET_LONG_SIDE, which also makes the verifier's input independent of
the policy's output resolution.
"""

import base64
import binascii
import hashlib
import io

from PIL import Image, ImageStat, UnidentifiedImageError

from ..models import MAX_IMAGE_BYTES

TARGET_LONG_SIDE = 1536
MAX_SOURCE_PIXELS = 4096 * 4096
MIN_SIDE = 64
FORMATS = {"PNG", "JPEG", "WEBP"}


class InvalidImage(ValueError):
    pass


def decode_payload(payload):
    """Base64 (optionally a data URL) to raw bytes, bounded before decoding."""
    text = (payload or "").strip()
    if text.startswith("data:"):
        header, _, text = text.partition(",")
        if ";base64" not in header:
            raise InvalidImage("Data URL must be base64-encoded")
    if not text:
        raise InvalidImage("No image submitted")
    if len(text) > (MAX_IMAGE_BYTES * 4) // 3 + 4:
        raise InvalidImage("Image exceeds the size limit")
    try:
        raw = base64.b64decode(text, validate=True)
    except (binascii.Error, ValueError) as error:
        raise InvalidImage("Image is not valid base64") from error
    if len(raw) > MAX_IMAGE_BYTES:
        raise InvalidImage("Image exceeds the size limit")
    return raw


def load_image(raw):
    """Raw bytes to an RGB PIL image, rejecting unsupported or oversized inputs."""
    try:
        with Image.open(io.BytesIO(raw)) as probe:
            if probe.format not in FORMATS:
                raise InvalidImage(f"Unsupported image format {probe.format}")
            width, height = probe.size
            if width * height > MAX_SOURCE_PIXELS:
                raise InvalidImage("Image has too many pixels")
            if min(width, height) < MIN_SIDE:
                raise InvalidImage(f"Image sides must be at least {MIN_SIDE}px")
            probe.verify()
        image = Image.open(io.BytesIO(raw))
        image.load()
        text_info = {k: v for k, v in image.info.items() if k == "fixture_readings"}
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise InvalidImage("Image could not be decoded") from error
    if image.mode in ("RGBA", "LA", "P"):
        background = Image.new("RGB", image.size, (255, 255, 255))
        rgba = image.convert("RGBA")
        background.paste(rgba, mask=rgba.getchannel("A"))
        background.info.update(text_info)
        return background
    converted = image.convert("RGB")
    converted.info.update(text_info)
    return converted


def normalise(image, long_side=TARGET_LONG_SIDE):
    scale = long_side / max(image.size)
    if scale == 1:
        return image
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.resize(size, Image.LANCZOS)


def prepare(payload, long_side=TARGET_LONG_SIDE):
    """Return (png_bytes, info) for the verifier. `info` is safe to show the caller."""
    raw = decode_payload(payload)
    image = load_image(raw)
    # Only FixtureVerifier reads this; model verifiers receive re-encoded pixels alone.
    fixture_readings = image.info.get("fixture_readings")
    original = image.size
    blank = is_blank(image)
    image = normalise(image, long_side)
    buffer = io.BytesIO()
    image.save(buffer, "PNG", optimize=False)
    png = buffer.getvalue()
    return png, {
        "image_sha256": hashlib.sha256(raw).hexdigest(),
        "width": original[0],
        "height": original[1],
        "verifier_width": image.width,
        "verifier_height": image.height,
        "blank": blank,
        "fixture_readings": fixture_readings,
    }


def is_blank(image, threshold=2.0):
    """True when the image is effectively one flat colour, so no text can be present."""
    return ImageStat.Stat(image.convert("L")).stddev[0] < threshold
