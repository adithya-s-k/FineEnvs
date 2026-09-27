import base64
import io

import pytest
from image_text_gen.client import encode_image
from image_text_gen.fixtures import png_bytes, render
from image_text_gen.models import MAX_IMAGE_BYTES
from image_text_gen.server.images import TARGET_LONG_SIDE, InvalidImage, prepare
from PIL import Image


def test_images_are_normalised_to_a_fixed_long_side():
    image, _ = render("Keep Calm", size=(768, 384))
    png, info = prepare(encode_image(image))
    with Image.open(io.BytesIO(png)) as out:
        assert out.size == (TARGET_LONG_SIDE, TARGET_LONG_SIDE // 2)
        assert "fixture_readings" not in out.info
    assert (info["width"], info["height"]) == (768, 384) and info["blank"] is False


def test_data_urls_jpeg_and_transparency_are_accepted():
    image, _ = render("Keep Calm", size=(300, 200))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG")
    payload = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()
    assert prepare(payload)[1]["width"] == 300
    rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    assert prepare(encode_image(rgba))[1]["blank"] is True


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "not base64!",
        base64.b64encode(b"GIF89a-not-really").decode(),
        "data:image/png,rawbytes",
        encode_image(Image.new("RGB", (32, 32), "white")),
        "A" * ((MAX_IMAGE_BYTES * 4) // 3 + 8),
    ],
)
def test_invalid_submissions_are_rejected_before_any_verifier_call(payload):
    with pytest.raises(InvalidImage):
        prepare(payload)


def test_gif_is_not_an_accepted_format():
    buffer = io.BytesIO()
    Image.new("RGB", (100, 100), "red").save(buffer, "GIF")
    with pytest.raises(InvalidImage, match="Unsupported"):
        prepare(encode_image(buffer.getvalue()))


def test_fixture_labels_travel_in_info_not_pixels():
    image, expected = render("Keep Calm", size=(400, 200))
    _, info = prepare(encode_image(png_bytes(image, [expected, "other"])))
    assert info["fixture_readings"] == f"{expected}\x1fother"
