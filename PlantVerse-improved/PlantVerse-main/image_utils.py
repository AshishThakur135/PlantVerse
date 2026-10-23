"""Shared image validation used by the Streamlit app and the ML API."""

from __future__ import annotations

import io
import os
from typing import Any

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_SIZE_BYTES", str(10 * 1024 * 1024)))
MIN_SIDE_PX = int(os.getenv("MIN_IMAGE_SIDE_PX", "64"))
MAX_SIDE_PX = int(os.getenv("MAX_IMAGE_SIDE_PX", "8000"))
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}


class ImageValidationError(ValueError):
    """Raised with a user-friendly message when an image is not acceptable."""


def read_bytes(source: Any) -> bytes:
    """Return raw bytes from bytes, a file-like object (Streamlit upload) or a path."""
    if source is None:
        raise ImageValidationError("No image provided. Please upload a photo.")
    if isinstance(source, (bytes, bytearray)):
        return bytes(source)
    if hasattr(source, "getvalue"):
        return source.getvalue()
    if hasattr(source, "read"):
        if hasattr(source, "seek"):
            source.seek(0)
        data = source.read()
        if hasattr(source, "seek"):
            source.seek(0)
        return data
    if isinstance(source, (str, os.PathLike)):
        try:
            if os.path.getsize(source) > MAX_IMAGE_BYTES:
                raise ImageValidationError(_too_big())
            with open(source, "rb") as fh:
                return fh.read()
        except OSError:
            raise ImageValidationError("The image file could not be read.")
    raise ImageValidationError("Unsupported image input.")


def _too_big() -> str:
    return f"The image is larger than the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit."


def validate_image(source: Any) -> Image.Image:
    """Validate type, size, readability and dimensions; return a loaded RGB image."""
    if isinstance(source, Image.Image):
        img = source
        data = None
    else:
        data = read_bytes(source)
        if not data:
            raise ImageValidationError("The uploaded file is empty.")
        if len(data) > MAX_IMAGE_BYTES:
            raise ImageValidationError(_too_big())
        try:
            probe = Image.open(io.BytesIO(data))
            fmt = (probe.format or "").upper()
            probe.verify()  # integrity check; invalidates `probe`
            img = Image.open(io.BytesIO(data))
            img.load()
        except (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError):
            raise ImageValidationError("The file is not a readable image (it may be corrupt or unsupported).")
        if fmt not in ALLOWED_FORMATS:
            raise ImageValidationError("Unsupported image type. Please upload a JPG, PNG or WEBP image.")

    width, height = img.size
    if min(width, height) < MIN_SIDE_PX:
        raise ImageValidationError(f"The image is too small ({width}×{height}px). Minimum side is {MIN_SIDE_PX}px.")
    if max(width, height) > MAX_SIDE_PX:
        raise ImageValidationError(f"The image is too large ({width}×{height}px). Maximum side is {MAX_SIDE_PX}px.")

    img = ImageOps.exif_transpose(img)  # honour phone-camera orientation
    return img.convert("RGB")


def to_jpeg_bytes(image: Image.Image, quality: int = 88) -> bytes:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
