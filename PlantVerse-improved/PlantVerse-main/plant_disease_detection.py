"""PlantVerse - plant disease detection engine.

Backends (first usable wins): optional external API (PLANT_DISEASE_API_URL),
optional local Keras model (PLANT_DISEASE_KERAS_MODEL_PATH, must be set
explicitly), then the Khawajaa EfficientNet-B4 PlantVillage model.

Every call returns a structured dict with an explicit ``status``:
``success`` | ``uncertain`` | ``unsupported`` | ``unavailable`` | ``error``.
A prediction is never fabricated: if no model can run, status is
``unavailable`` and no label or confidence is returned.
"""

from __future__ import annotations

import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from disease_recommendations import RAW_LABEL_MAP, get_disease_info, normalize_disease_label
from image_utils import ImageValidationError, validate_image

log = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = os.getenv(
    "PLANT_DISEASE_MODEL_NAME", "Khawajaa/plant-disease-detector"
)
CONFIDENCE_THRESHOLD = float(os.getenv("DISEASE_CONFIDENCE_THRESHOLD", "0.50"))
EXTERNAL_API_URL = os.getenv("PLANT_DISEASE_API_URL", "").strip()
EXTERNAL_API_TIMEOUT = float(os.getenv("PLANT_DISEASE_API_TIMEOUT", "12"))
# Keras is opt-in: the model file is not shipped, so it is only used when configured.
LOCAL_KERAS_PATH = os.getenv("PLANT_DISEASE_KERAS_MODEL_PATH", "").strip()
# Preprocessing is model specific; default matches the common PlantVillage Keras
# tutorial models (raw 0-255 pixels). Set to 0.00392156 (=1/255) if yours was trained on scaled input.
KERAS_RESCALE = float(os.getenv("PLANT_DISEASE_KERAS_RESCALE", "1.0"))
# Both supported model formats use the same standard alphabetical class order.
DISEASE_CLASSES: List[str] = list(RAW_LABEL_MAP.keys())
KERAS_CLASSES = DISEASE_CLASSES

_models: Dict[str, Any] = {}  # only successful loads are cached, so failures can be retried
_lock = threading.Lock()


def reset_model_cache() -> None:
    _models.clear()


def _load_backend() -> Tuple[Optional[Tuple[str, Any]], Optional[str]]:
    """Return ((kind, model), None) or (None, reason). Cached after success."""
    with _lock:
        if "backend" in _models:
            return _models["backend"], None
        reason = None
        if LOCAL_KERAS_PATH:
            if not os.path.exists(LOCAL_KERAS_PATH):
                reason = "Configured Keras model file was not found."
            else:
                try:
                    import tensorflow as tf  # optional dependency

                    _models["backend"] = ("keras", tf.keras.models.load_model(LOCAL_KERAS_PATH))
                    return _models["backend"], None
                except Exception as exc:  # noqa: BLE001
                    log.warning("Keras model load failed: %s", exc)
                    reason = "The local Keras model could not be loaded."
        try:
            import torch
            from huggingface_hub import hf_hub_download
            from torchvision import models

            checkpoint_path = hf_hub_download(
                repo_id=DEFAULT_MODEL_NAME,
                filename="best_model.pth",
            )

            class EfficientNetB4Classifier(torch.nn.Module):
                def __init__(self) -> None:
                    super().__init__()
                    backbone = models.efficientnet_b4(weights=None)
                    self.features = backbone.features
                    self.avgpool = backbone.avgpool
                    self.classifier = torch.nn.Sequential(
                        torch.nn.Dropout(p=0.4),
                        torch.nn.Linear(1792, 512),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(p=0.2),
                        torch.nn.Linear(512, len(DISEASE_CLASSES)),
                    )

                def forward(self, tensor):
                    tensor = self.features(tensor)
                    tensor = self.avgpool(tensor)
                    tensor = torch.flatten(tensor, 1)
                    return self.classifier(tensor)

            model = EfficientNetB4Classifier()
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            state_dict = checkpoint.get("model_state_dict") if isinstance(checkpoint, dict) else None
            if not isinstance(state_dict, dict):
                raise ValueError("The disease checkpoint has no model state dictionary.")
            model.load_state_dict(state_dict)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            model.eval()
            _models["backend"] = ("torch", model)
            return _models["backend"], None
        except Exception as exc:  # noqa: BLE001
            log.warning("Hugging Face disease model load failed: %s", exc)
            return None, (reason + " " if reason else "") + (
                "The disease model could not be loaded. Check your internet connection for the "
                "first-time model download and the installed dependencies."
            )


def _valid_pairs(pairs: Any) -> List[Tuple[str, float]]:
    """Keep only (str label, finite score in [0,1]) pairs, sorted best-first."""
    out = []
    for label, score in pairs or []:
        try:
            score = float(score)
        except (TypeError, ValueError):
            continue
        if isinstance(label, str) and label.strip() and math.isfinite(score) and 0.0 <= score <= 1.0:
            out.append((label, score))
    return sorted(out, key=lambda p: p[1], reverse=True)


def _run_keras(model: Any, image: Image.Image) -> List[Tuple[str, float]]:
    import numpy as np

    shape = getattr(model, "input_shape", None) or (None, 128, 128, 3)
    height, width = (shape[1] or 128), (shape[2] or 128)
    arr = np.asarray(image.resize((int(width), int(height))), dtype=np.float32) * KERAS_RESCALE
    probs = np.asarray(model.predict(arr[None, ...], verbose=0)[0], dtype=np.float64)
    if probs.shape[0] != len(KERAS_CLASSES):
        raise ValueError(f"Keras model returned {probs.shape[0]} classes, expected {len(KERAS_CLASSES)}.")
    return list(zip(KERAS_CLASSES, probs.tolist()))


def _run_torch(model: Any, image: Image.Image) -> List[Tuple[str, float]]:
    """Run Khawajaa's model; returned scores are softmax scores, not calibrated probabilities."""
    import numpy as np
    import torch

    image = image.convert("RGB")
    width, height = image.size
    scale = 256 / min(width, height)
    resized = image.resize((round(width * scale), round(height * scale)), Image.Resampling.BICUBIC)
    left = (resized.width - 224) // 2
    top = (resized.height - 224) // 2
    cropped = resized.crop((left, top, left + 224, top + 224))

    pixels = np.asarray(cropped, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1).copy())
    mean = torch.tensor((0.485, 0.456, 0.406), dtype=tensor.dtype)[:, None, None]
    std = torch.tensor((0.229, 0.224, 0.225), dtype=tensor.dtype)[:, None, None]
    tensor = ((tensor - mean) / std).unsqueeze(0).to(next(model.parameters()).device)
    with torch.inference_mode():
        logits = model(tensor)
        if logits.ndim != 2 or logits.shape[0] != 1 or logits.shape[1] != len(DISEASE_CLASSES):
            raise ValueError("The disease model returned an unexpected output shape.")
        scores = torch.softmax(logits[0], dim=0)
    top_scores, top_indices = torch.topk(scores, k=min(5, len(DISEASE_CLASSES)))
    return [
        (DISEASE_CLASSES[index], float(score))
        for index, score in zip(top_indices.cpu().tolist(), top_scores.cpu().tolist())
    ]


def _build_result(pairs: Any, backend: str, threshold: float) -> Dict[str, Any]:
    pairs = _valid_pairs(pairs)
    if not pairs:
        return _fail("error", "The model returned no valid prediction.", backend)
    raw_label, score = pairs[0]
    info = get_disease_info(raw_label)
    candidates = [
        {"raw_label": lb, "label": normalize_disease_label(lb), "score": sc, "confidence_pct": round(sc * 100, 1)}
        for lb, sc in pairs[:5]
    ]
    base = {
        "backend": backend,
        "raw_label": raw_label,
        "confidence": score,
        "confidence_pct": round(score * 100, 1),
        "top_candidates": candidates,
        "threshold": threshold,
    }
    if not info["supported"]:
        return {**base, "success": False, "status": "unsupported", "supported": False,
                "message": info["message"], "recommendations": [], "preventive_measures": []}

    crop, healthy = info.get("crop", ""), info.get("health_status") == "Healthy"
    condition = info.get("condition", "")
    result = {
        **base,
        "supported": True,
        "plant": crop,
        "disease": condition,
        "disease_name": f"Healthy {crop}" if healthy else (condition if crop in condition else f"{crop} {condition}"),
        "health_status": info.get("health_status"),
        "pathogen": info.get("pathogen"),
        "possible_causes": info.get("possible_causes"),
        "severity": info.get("severity"),
        "symptoms": info.get("symptoms"),
        "recommendations": info.get("recommendations", []),
        "preventive_measures": info.get("preventive_measures", []),
        "references": info.get("references", []),
        "expert_note": info["expert_note"],
        "is_uncertain": score < threshold,
    }
    if score < threshold:
        result.update(
            success=False, status="uncertain",
            message=(f"Low confidence ({score * 100:.1f}% < {threshold * 100:.0f}%). Treat this as a "
                     "possibility only and retake a clearer, close-up photo of a single leaf."),
        )
    else:
        result.update(success=True, status="success", message="")
    return result


def _fail(status: str, message: str, backend: str = "none") -> Dict[str, Any]:
    return {"success": False, "status": status, "message": message, "backend": backend,
            "recommendations": [], "preventive_measures": []}


def _call_external(image_bytes: bytes) -> Tuple[Optional[List[Tuple[str, float]]], Optional[str]]:
    import requests

    try:
        resp = requests.post(EXTERNAL_API_URL, files={"image": ("leaf.jpg", image_bytes, "image/jpeg")},
                             timeout=EXTERNAL_API_TIMEOUT)
    except requests.Timeout:
        return None, "The disease service timed out."
    except requests.RequestException:
        return None, "The disease service could not be reached."
    if resp.status_code != 200:
        return None, f"The disease service returned HTTP {resp.status_code}."
    try:
        data = resp.json()
        label = data["rawLabel"] if "rawLabel" in data else data["raw_label"]
        if "confidenceScore" in data:
            score = float(data["confidenceScore"])
        else:
            score = float(data["confidence"])
            score = score / 100 if score > 1 else score
        pairs = [(label, score)]
    except (ValueError, KeyError, TypeError):
        return None, "The disease service returned an invalid response."
    return pairs, None


def analyze_disease(image: Image.Image, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Run disease detection on an already validated RGB image. Never raises."""
    threshold = CONFIDENCE_THRESHOLD if threshold is None else threshold
    notes: List[str] = []
    try:
        if EXTERNAL_API_URL:
            from image_utils import to_jpeg_bytes

            pairs, err = _call_external(to_jpeg_bytes(image))
            if pairs:
                return _build_result(pairs, "external-api", threshold)
            notes.append(err or "")
        backend, reason = _load_backend()
        if backend is None:
            return _fail("unavailable", " ".join(n for n in notes + [reason or ""] if n))
        kind, model = backend
        if kind == "keras":
            pairs = _run_keras(model, image)
        elif kind == "torch":
            pairs = _run_torch(model, image)
        else:
            pairs = [(p.get("label"), p.get("score")) for p in model(image, top_k=5)]
        return _build_result(pairs, kind, threshold)
    except Exception as exc:  # noqa: BLE001 - inference must never crash the page
        log.exception("Disease inference failed")
        return _fail("error", f"Disease analysis failed ({type(exc).__name__}). Please try again.")


def predict_disease(image_input: Any, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Validate ``image_input`` (bytes/file/path/PIL) and analyse it. Never raises."""
    try:
        image = validate_image(image_input)
    except ImageValidationError as exc:
        return _fail("error", str(exc))
    return analyze_disease(image, threshold)
