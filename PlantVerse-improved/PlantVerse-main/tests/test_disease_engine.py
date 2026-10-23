import math

import pytest
import requests

import plant_disease_detection as eng


class FakePipe:
    def __init__(self, preds=None, exc=None):
        self.preds, self.exc = preds, exc

    def __call__(self, image, top_k=5):
        if self.exc:
            raise self.exc
        return self.preds


def use_backend(monkeypatch, preds=None, exc=None):
    monkeypatch.setattr(eng, "EXTERNAL_API_URL", "")
    monkeypatch.setattr(eng, "_load_backend", lambda: (("transformers", FakePipe(preds, exc)), None))


def test_success(monkeypatch, png_bytes):
    use_backend(monkeypatch, [{"label": "Tomato___Early_blight", "score": 0.93},
                              {"label": "Tomato___Late_blight", "score": 0.04}])
    r = eng.predict_disease(png_bytes)
    assert r["status"] == "success" and r["success"] is True
    assert r["plant"] == "Tomato" and r["health_status"] == "Diseased"
    assert r["confidence"] == pytest.approx(0.93)
    assert r["recommendations"] == [] and r["expert_note"]


def test_khawajaa_torch_model_returns_confidence_and_treatment(monkeypatch, pil_image):
    torch = pytest.importorskip("torch")

    class TorchModel:
        def parameters(self):
            return iter((torch.nn.Parameter(torch.zeros(1)),))

        def __call__(self, image_tensor):
            assert tuple(image_tensor.shape) == (1, 3, 224, 224)
            expected = torch.tensor([
                (30 / 255 - 0.485) / 0.229,
                (140 / 255 - 0.456) / 0.224,
                (40 / 255 - 0.406) / 0.225,
            ])
            assert torch.allclose(image_tensor[0, :, 112, 112], expected, atol=1e-5)
            logits = torch.full((1, len(eng.DISEASE_CLASSES)), -5.0)
            logits[0, eng.DISEASE_CLASSES.index("Tomato___Early_blight")] = 5.0
            return logits

    assert eng.DISEASE_CLASSES == sorted(eng.RAW_LABEL_MAP)
    monkeypatch.setattr(eng, "EXTERNAL_API_URL", "")
    monkeypatch.setattr(eng, "_load_backend", lambda: (("torch", TorchModel()), None))

    result = eng.analyze_disease(pil_image)

    assert result["status"] == "success"
    assert result["backend"] == "torch"
    assert result["plant"] == "Tomato"
    assert result["disease"] == "Early Blight"
    assert result["confidence_pct"] > 99
    assert result["recommendations"] == []


def test_healthy_label(monkeypatch, png_bytes):
    use_backend(monkeypatch, [{"label": "Potato___healthy", "score": 0.9}])
    r = eng.predict_disease(png_bytes)
    assert r["health_status"] == "Healthy" and r["disease_name"] == "Healthy Potato"


def test_low_confidence_is_uncertain(monkeypatch, png_bytes):
    use_backend(monkeypatch, [{"label": "Tomato___healthy", "score": 0.30}])
    r = eng.predict_disease(png_bytes)
    assert r["status"] == "uncertain" and r["success"] is False and "Low confidence" in r["message"]


def test_unknown_label_is_unsupported_not_guessed(monkeypatch, png_bytes):
    use_backend(monkeypatch, [{"label": "Totally_new_disease", "score": 0.99}])
    r = eng.predict_disease(png_bytes)
    assert r["status"] == "unsupported" and "plant" not in r and "disease_name" not in r


def test_partial_label_not_matched(monkeypatch, png_bytes):
    use_backend(monkeypatch, [{"label": "Tomato", "score": 0.99}])
    assert eng.predict_disease(png_bytes)["status"] == "unsupported"


def test_missing_model_never_fabricates_healthy(monkeypatch, png_bytes):
    monkeypatch.setattr(eng, "EXTERNAL_API_URL", "")
    monkeypatch.setattr(eng, "_load_backend", lambda: (None, "model missing"))
    r = eng.predict_disease(png_bytes)
    assert r["status"] == "unavailable" and r["success"] is False
    assert "confidence" not in r and "disease_name" not in r and "health_status" not in r


def test_inference_exception(monkeypatch, png_bytes):
    use_backend(monkeypatch, exc=RuntimeError("boom /secret/path"))
    r = eng.predict_disease(png_bytes)
    assert r["status"] == "error" and "secret" not in r["message"]


@pytest.mark.parametrize("score", [float("nan"), 1.7, -0.2, "abc", None])
def test_invalid_scores_rejected(monkeypatch, png_bytes, score):
    use_backend(monkeypatch, [{"label": "Tomato___healthy", "score": score}])
    assert eng.predict_disease(png_bytes)["status"] == "error"


def test_empty_prediction(monkeypatch, png_bytes):
    use_backend(monkeypatch, [])
    assert eng.predict_disease(png_bytes)["status"] == "error"


def test_invalid_image_returns_error():
    assert eng.predict_disease(b"junk")["status"] == "error"
    assert eng.predict_disease(None)["status"] == "error"


# ---- external API ----------------------------------------------------------
class Resp:
    def __init__(self, status=200, payload=None, bad_json=False):
        self.status_code, self.payload, self.bad_json = status, payload, bad_json

    def json(self):
        if self.bad_json:
            raise ValueError("bad json")
        return self.payload


def use_external(monkeypatch, behaviour, local=None):
    monkeypatch.setattr(eng, "EXTERNAL_API_URL", "http://svc.invalid/predict")

    def post(*a, **k):
        if isinstance(behaviour, Exception):
            raise behaviour
        return behaviour

    monkeypatch.setattr(requests, "post", post)
    monkeypatch.setattr(eng, "_load_backend", lambda: (local, "no local model") if local is None else (local, None))


def test_external_success(monkeypatch, png_bytes):
    use_external(monkeypatch, Resp(payload={"rawLabel": "Tomato___healthy", "confidenceScore": 0.88}))
    r = eng.predict_disease(png_bytes)
    assert r["status"] == "success" and r["backend"] == "external-api"


@pytest.mark.parametrize("behaviour", [
    requests.Timeout(), requests.ConnectionError(), Resp(status=500),
    Resp(bad_json=True), Resp(payload={"nope": 1}), Resp(payload={"rawLabel": "x", "confidence": "high"}),
])
def test_external_failures_do_not_crash(monkeypatch, png_bytes, behaviour):
    use_external(monkeypatch, behaviour)
    r = eng.predict_disease(png_bytes)
    assert r["status"] == "unavailable" and r["success"] is False


def test_external_failure_falls_back_to_local(monkeypatch, png_bytes):
    use_external(monkeypatch, requests.Timeout(),
                 local=("transformers", FakePipe([{"label": "Potato___healthy", "score": 0.8}])))
    assert eng.predict_disease(png_bytes)["backend"] == "transformers"
