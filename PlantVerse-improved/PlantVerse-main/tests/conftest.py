import io
import os
import sys

import pytest
from PIL import Image

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)  # st.logo / static paths and AppTest resolve relative to the project root
os.environ.setdefault("PLANTVERSE_PERSIST_HISTORY", "0")


def make_image(fmt="PNG", size=(128, 128), mode="RGB") -> bytes:
    buf = io.BytesIO()
    Image.new(mode, size, (30, 140, 40) if mode == "RGB" else 128).save(buf, format=fmt)
    return buf.getvalue()


@pytest.fixture
def png_bytes():
    return make_image("PNG")


@pytest.fixture
def jpeg_bytes():
    return make_image("JPEG")


@pytest.fixture
def pil_image():
    return Image.new("RGB", (128, 128), (30, 140, 40))
