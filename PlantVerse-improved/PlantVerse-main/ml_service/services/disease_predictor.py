"""ML service predictor: thin wrapper over the shared engine (model loaded once)."""

from __future__ import annotations

import os
import sys
import threading
from typing import Any, Dict

from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import plant_disease_detection as engine  # noqa: E402
from treatment_service import get_treatment  # noqa: E402


class DiseasePredictor:
    def __init__(self) -> None:
        self.threshold = engine.CONFIDENCE_THRESHOLD
        self.backend = "uninitialized"
        self.last_error: str = ""
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.backend in ("torch", "transformers", "keras")

    def load_model(self) -> bool:
        """Load once; safe to call repeatedly. Returns readiness."""
        with self._lock:
            if self.is_loaded:
                return True
            backend, reason = engine._load_backend()
            if backend:
                self.backend, self.last_error = backend[0], ""
            else:
                self.backend, self.last_error = "unavailable", reason or "Model could not be loaded."
            return self.is_loaded

    def predict(self, image: Image.Image) -> Dict[str, Any]:
        """Return camelCase API payload. Confidence is a percentage in `confidence`
        (backward compatible) and a 0-1 fraction in `confidenceScore`."""
        r = engine.analyze_disease(image, self.threshold)
        if r["status"] in ("unavailable", "error"):
            return {"success": False, "status": r["status"], "error": r["message"]}
        out: Dict[str, Any] = {
            "success": r["status"] == "success",
            "status": r["status"],
            "backend": r.get("backend"),
            "rawLabel": r.get("raw_label"),
            "confidence": r.get("confidence_pct"),
            "confidenceScore": r.get("confidence"),
            "isUncertain": r["status"] != "success",
            "message": r.get("message") or None,
            "topCandidates": [{"label": c["label"], "rawLabel": c["raw_label"], "confidence": c["score"],
                               "confidencePct": c["confidence_pct"]} for c in r.get("top_candidates", [])],
        }
        treatment = get_treatment(r)
        out["treatment"] = treatment
        if r["status"] == "unsupported":
            return out
        out.update({
            "plant": r["plant"], "disease": r["disease"], "diseaseName": r["disease_name"],
            "healthStatus": r["health_status"], "pathogen": r["pathogen"], "severity": r["severity"],
            "symptoms": treatment["symptoms"],
            "recommendations": treatment["immediate_actions"] + treatment["non_chemical_management"],
            "preventiveMeasures": treatment["prevention"],
            "uncertaintyMessage": r.get("message") or None,
        })
        return out


predictor = DiseasePredictor()
