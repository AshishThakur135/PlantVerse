import pytest

import image_utils
from image_utils import ImageValidationError, validate_image
from tests.conftest import make_image


def test_valid_png_and_jpeg(png_bytes, jpeg_bytes):
    assert validate_image(png_bytes).mode == "RGB"
    assert validate_image(jpeg_bytes).size == (128, 128)


def test_rgba_and_grayscale_converted_to_rgb():
    assert validate_image(make_image("PNG", mode="RGBA")).mode == "RGB"
    assert validate_image(make_image("PNG", mode="L")).mode == "RGB"


@pytest.mark.parametrize("bad", [None, b""])
def test_missing_or_empty(bad):
    with pytest.raises(ImageValidationError):
        validate_image(bad)


def test_corrupt_image():
    with pytest.raises(ImageValidationError, match="not a readable image"):
        validate_image(b"definitely not an image")


def test_truncated_image(png_bytes):
    with pytest.raises(ImageValidationError):
        validate_image(png_bytes[: len(png_bytes) // 2])


def test_unsupported_format():
    with pytest.raises(ImageValidationError, match="Unsupported image type"):
        validate_image(make_image("GIF"))


def test_too_large_file(monkeypatch, png_bytes):
    monkeypatch.setattr(image_utils, "MAX_IMAGE_BYTES", 100)
    with pytest.raises(ImageValidationError, match="limit"):
        validate_image(png_bytes)


def test_dimensions(monkeypatch):
    with pytest.raises(ImageValidationError, match="too small"):
        validate_image(make_image("PNG", size=(10, 10)))
    monkeypatch.setattr(image_utils, "MAX_SIDE_PX", 100)
    with pytest.raises(ImageValidationError, match="too large"):
        validate_image(make_image("PNG", size=(200, 200)))


def test_file_like_object(png_bytes):
    import io

    assert validate_image(io.BytesIO(png_bytes)).size == (128, 128)
