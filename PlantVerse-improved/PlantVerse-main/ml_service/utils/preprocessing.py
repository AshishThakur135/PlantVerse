"""API image validation (delegates to the shared validator)."""

import os
import sys
from typing import Tuple

from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from image_utils import MAX_IMAGE_BYTES as MAX_UPLOAD_SIZE  # noqa: E402,F401
from image_utils import ImageValidationError, validate_image  # noqa: E402


def validate_and_load_image(data: bytes) -> Image.Image:
    """Validate raw bytes; raises ValueError (ImageValidationError) with a safe message."""
    return validate_image(data)


def prepare_tensor_input(image: Image.Image, target_size: Tuple[int, int] = (224, 224)) -> Image.Image:
    return image.resize(target_size, Image.Resampling.BILINEAR)
