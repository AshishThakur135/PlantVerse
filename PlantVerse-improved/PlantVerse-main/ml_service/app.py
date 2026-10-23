"""
PlantVerse — Plant Disease Detection ML Microservice
===================================================
Production-ready ASGI / REST API for plant leaf disease inference.

Endpoints:
- GET  /health               : Service health check & model metadata
- POST /api/disease/predict  : Run neural disease prediction on uploaded image
- POST /api/disease/diagnosis: Save or validate diagnosis record
"""

import asyncio
import logging
import os
import sys
import time
from contextlib import asynccontextmanager

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from image_utils import MAX_IMAGE_BYTES  # noqa: E402
from ml_service.services.disease_predictor import predictor  # noqa: E402
from ml_service.utils.preprocessing import validate_and_load_image  # noqa: E402

log = logging.getLogger("plantverse.api")
START_TIME = time.time()
PREDICT_TIMEOUT = float(os.getenv("PREDICT_TIMEOUT_SECONDS", "30"))
# Comma-separated approved origins. Empty = no cross-origin access (same-origin / server-to-server only).
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]


def err(message: str, status: int, code: str) -> JSONResponse:
    return JSONResponse({"success": False, "status": "error", "code": code, "error": message}, status_code=status)


async def health(request: Request) -> JSONResponse:
    """Liveness (`status`) is separate from model readiness (`modelReady`)."""
    ready = predictor.is_loaded
    return JSONResponse({
        "status": "ok",
        "service": "PlantVerse Plant Disease Prediction ML API",
        "version": "1.1.0",
        "uptimeSeconds": round(time.time() - START_TIME, 1),
        "modelReady": ready,
        "modelLoaded": ready,  # deprecated alias of modelReady
        "modelBackend": predictor.backend,
        "threshold": predictor.threshold,
        **({} if ready else {"modelMessage": "Model not loaded yet or unavailable."}),
    })


async def ready(request: Request) -> JSONResponse:
    ok = predictor.is_loaded
    return JSONResponse({"ready": ok, "modelBackend": predictor.backend}, status_code=200 if ok else 503)


async def predict(request: Request) -> JSONResponse:
    """POST multipart/form-data with an `image` (or `file`) field."""
    declared = request.headers.get("content-length")
    if declared and declared.isdigit() and int(declared) > MAX_IMAGE_BYTES + 64 * 1024:
        return err(f"Upload exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.", 413, "too_large")
    try:
        form = await request.form()
    except Exception:  # noqa: BLE001
        return err("Request must be multipart/form-data with an 'image' field.", 400, "bad_request")
    field = form.get("image") or form.get("file")
    if field is None or not hasattr(field, "read"):
        return err("No image uploaded. Send a file in the 'image' field.", 400, "missing_image")
    content = await field.read()
    if not content:
        return err("Uploaded image file is empty.", 400, "empty_image")
    if len(content) > MAX_IMAGE_BYTES:
        return err(f"Upload exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.", 413, "too_large")
    try:
        image = validate_and_load_image(content)
    except ValueError as exc:
        return err(str(exc), 422, "invalid_image")
    if not predictor.load_model():
        return err("The prediction model is not ready.", 503, "model_unavailable")
    try:
        result = await asyncio.wait_for(asyncio.to_thread(predictor.predict, image), timeout=PREDICT_TIMEOUT)
    except asyncio.TimeoutError:
        return err("Prediction timed out.", 504, "timeout")
    except Exception:  # noqa: BLE001
        log.exception("Prediction failed")
        return err("Internal prediction error.", 500, "internal_error")
    if result.get("status") in ("unavailable", "error"):
        return err(result.get("error", "Prediction failed."), 503 if result["status"] == "unavailable" else 500,
                   "model_unavailable" if result["status"] == "unavailable" else "internal_error")
    return JSONResponse(result)


async def save_diagnosis(request: Request) -> JSONResponse:
    """Validate a diagnosis payload. This API has no storage or auth, so nothing is saved."""
    try:
        body = await request.json()
    except Exception:  # noqa: BLE001
        return err("Body must be valid JSON.", 400, "bad_request")
    if not isinstance(body, dict):
        return err("Body must be a JSON object.", 400, "bad_request")
    for field in ("plant", "disease", "confidence"):
        if field not in body:
            return err(f"Missing required diagnosis field: {field}", 422, "missing_field")
    if not isinstance(body["confidence"], (int, float)) or isinstance(body["confidence"], bool) \
            or not 0 <= body["confidence"] <= 100:
        return err("'confidence' must be a number between 0 and 100.", 422, "invalid_field")
    return JSONResponse({
        "success": True,
        "saved": False,
        "message": "Diagnosis validated but NOT stored: this API does not persist records.",
        "record": {k: body.get(k) for k in ("plant", "disease", "healthStatus", "confidence", "timestamp")},
    })


@asynccontextmanager
async def lifespan(app):
    # Load in a worker thread so the service starts (and /health answers) while the model loads.
    task = asyncio.create_task(asyncio.to_thread(predictor.load_model))
    yield
    task.cancel()


routes = [
    Route("/health", health, methods=["GET"]),
    Route("/ready", ready, methods=["GET"]),
    Route("/api/disease/predict", predict, methods=["POST"]),
    Route("/api/disease/diagnosis", save_diagnosis, methods=["POST"]),
]

middleware = [Middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"],
                         allow_headers=["Content-Type"])] if ALLOWED_ORIGINS else []

app = Starlette(debug=False, routes=routes, middleware=middleware, lifespan=lifespan)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("ml_service.app:app", host=os.getenv("HOST", "127.0.0.1"), port=port, reload=False)
