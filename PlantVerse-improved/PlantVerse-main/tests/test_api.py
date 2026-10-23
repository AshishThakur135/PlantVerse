import pytest
from starlette.testclient import TestClient

import ml_service.app as api
from ml_service.services.disease_predictor import predictor
import plant_disease_detection as eng
from tests.conftest import make_image


class Pipe:
    def __init__(self, preds=None, exc=None):
        self.preds, self.exc = preds, exc

    def __call__(self, image, top_k=5):
        if self.exc:
            raise self.exc
        return self.preds


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(eng, "EXTERNAL_API_URL", "")
    # skip model loading in lifespan; individual tests set the backend
    monkeypatch.setattr(predictor, "backend", "uninitialized")
    monkeypatch.setattr(eng, "_load_backend", lambda: (None, "no model"))
    with TestClient(api.app) as c:
        yield c


def ready_model(monkeypatch, preds=None, exc=None):
    monkeypatch.setattr(eng, "_load_backend", lambda: (("transformers", Pipe(preds, exc)), None))
    monkeypatch.setattr(predictor, "backend", "uninitialized")


def post_image(client, data, name="image"):
    return client.post("/api/disease/predict", files={name: ("leaf.png", data, "image/png")})


def test_health_separates_liveness_from_readiness(client):
    body = client.get("/health").json()
    assert client.get("/health").status_code == 200 and body["status"] == "ok"
    assert body["modelReady"] is False and body["modelLoaded"] is False
    assert client.get("/ready").status_code == 503


def test_ready_after_load(client, monkeypatch):
    ready_model(monkeypatch, [{"label": "Tomato___healthy", "score": 0.9}])
    predictor.load_model()
    assert client.get("/ready").status_code == 200 and client.get("/health").json()["modelReady"] is True


def test_predict_success(client, monkeypatch):
    ready_model(monkeypatch, [{"label": "Tomato___Early_blight", "score": 0.92}])
    r = post_image(client, make_image("PNG"))
    body = r.json()
    assert r.status_code == 200 and body["success"] and body["plant"] == "Tomato"
    assert body["confidence"] == 92.0 and body["confidenceScore"] == pytest.approx(0.92)
    assert body["treatment"]["status"] == "identity_unconfirmed"
    assert body["recommendations"] == []


def test_predict_uncertain_and_unsupported(client, monkeypatch):
    ready_model(monkeypatch, [{"label": "Tomato___healthy", "score": 0.2}])
    assert post_image(client, make_image("PNG")).json()["status"] == "uncertain"
    ready_model(monkeypatch, [{"label": "Strange", "score": 0.9}])
    body = post_image(client, make_image("PNG")).json()
    assert body["status"] == "unsupported" and "plant" not in body


def test_missing_upload(client):
    r = client.post("/api/disease/predict", files={"other": ("x.txt", b"x", "text/plain")})
    assert r.status_code == 400 and r.json()["code"] == "missing_image"


def test_not_multipart(client):
    assert client.post("/api/disease/predict", json={"a": 1}).status_code == 400


def test_empty_file(client):
    assert post_image(client, b"").status_code == 400


def test_invalid_image(client):
    r = post_image(client, b"not an image")
    assert r.status_code == 422 and r.json()["code"] == "invalid_image"


def test_too_large(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_IMAGE_BYTES", 50)
    assert post_image(client, make_image("PNG")).status_code == 413


def test_model_unavailable_503(client):
    r = post_image(client, make_image("PNG"))
    assert r.status_code == 503 and r.json()["code"] == "model_unavailable"


def test_inference_error_hides_internals(client, monkeypatch):
    ready_model(monkeypatch, exc=RuntimeError("secret /srv/model.bin"))
    r = post_image(client, make_image("PNG"))
    assert r.status_code == 500 and "secret" not in r.text and "/srv" not in r.text


def test_timeout(client, monkeypatch):
    import time

    ready_model(monkeypatch, [{"label": "Tomato___healthy", "score": 0.9}])
    predictor.load_model()
    monkeypatch.setattr(api, "PREDICT_TIMEOUT", 0.05)
    monkeypatch.setattr(predictor, "predict", lambda image: time.sleep(0.5))
    assert post_image(client, make_image("PNG")).status_code == 504


def test_diagnosis_endpoint_does_not_claim_saved(client):
    ok = client.post("/api/disease/diagnosis", json={"plant": "Tomato", "disease": "x", "confidence": 80})
    assert ok.status_code == 200 and ok.json()["saved"] is False
    assert client.post("/api/disease/diagnosis", json={"plant": "Tomato"}).status_code == 422
    assert client.post("/api/disease/diagnosis", json={"plant": "a", "disease": "b", "confidence": 500}).status_code == 422
    assert client.post("/api/disease/diagnosis", content=b"{bad").status_code == 400
    assert client.post("/api/disease/diagnosis", json=[1]).status_code == 400


def test_cors_closed_by_default(client):
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
