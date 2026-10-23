import pytest
import requests

import plant_identification as pi


class Pipe:
    def __init__(self, preds=None, exc=None):
        self.preds, self.exc = preds, exc

    def __call__(self, image, top_k=5):
        if self.exc:
            raise self.exc
        return self.preds


def patch_model(monkeypatch, pipe=None, load_exc=None):
    def loader():
        if load_exc:
            raise load_exc
        return pipe

    monkeypatch.setattr(pi, "load_classifier", loader)


def test_success(monkeypatch, pil_image):
    patch_model(monkeypatch, Pipe([{"label": "ginkgo_biloba", "score": 0.91}, {"label": "oak", "score": 0.02}]))
    r = pi.identify_plant(pil_image)
    assert r["status"] == "success" and r["label"] == "ginkgo biloba" and r["score"] == pytest.approx(0.91)
    assert r["candidates"][0]["confidence_pct"] == 91.0


def test_openplants_model_is_the_default():
    assert pi.DEFAULT_PLANT_MODEL_NAME == "domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224"
    assert pi.OPENPLANTS_MODEL_NAME == "domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224"


def test_invalid_candidate_does_not_hide_valid_prediction(monkeypatch, pil_image):
    patch_model(monkeypatch, Pipe([
        {"label": "invalid", "score": None},
        {"label": "ginkgo biloba", "score": 0.82},
    ]))
    result = pi.identify_plant(pil_image)
    assert result["status"] == "success"
    assert result["label"] == "ginkgo biloba"
    assert result["score"] == pytest.approx(0.82)


def test_low_confidence_uncertain(monkeypatch, pil_image):
    patch_model(monkeypatch, Pipe([{"label": "rose", "score": 0.2}]))
    r = pi.identify_plant(pil_image)
    assert r["status"] == "uncertain" and "20.0000%" in r["message"]
    assert r["candidates"][0]["label"] == "rose"


def test_identify_each_detected_potted_plant_with_per_crop_scores(monkeypatch, pil_image):
    monkeypatch.setattr(pi, "PLANT_OBJECT_MODEL_NAME", "test-detector")
    detections = [
        {"label": "potted plant", "score": 0.91,
         "box": {"xmin": 0, "ymin": 0, "xmax": 60, "ymax": 90}},
        {"label": "potted plant", "score": 0.88,
         "box": {"xmin": 60, "ymin": 0, "xmax": 128, "ymax": 90}},
        {"label": "potted plant", "score": 0.70,
         "box": {"xmin": 1, "ymin": 1, "xmax": 59, "ymax": 89}},
        {"label": "person", "score": 0.99,
         "box": {"xmin": 0, "ymin": 0, "xmax": 128, "ymax": 128}},
    ]
    monkeypatch.setattr(pi, "load_plant_detector", lambda: lambda image, threshold: detections)
    crops = []

    def identify(crop, threshold=None):
        crops.append(crop)
        label = "Epipremnum aureum" if len(crops) == 1 else "Monstera deliciosa"
        return {
            "status": "success", "success": True, "label": label,
            "score": 0.82 if len(crops) == 1 else 0.76, "threshold": 0.5,
            "candidates": [{"label": label, "score": 0.82}],
        }

    monkeypatch.setattr(pi, "identify_plant", identify)
    result = pi.identify_plants(pil_image)
    assert result["detection_status"] == "detected"
    assert len(result["instances"]) == 2  # overlapping duplicate suppressed
    assert [instance["label"] for instance in result["instances"]] == [
        "Epipremnum aureum", "Monstera deliciosa"
    ]
    assert [instance["detection_score"] for instance in result["instances"]] == [0.91, 0.88]
    assert [instance["score"] for instance in result["instances"]] == [0.82, 0.76]
    assert all(crop.size[0] > 0 and crop.size[1] > 0 for crop in crops)


def test_no_detected_plants_falls_back_to_explicit_whole_image_prediction(monkeypatch, pil_image):
    monkeypatch.setattr(pi, "PLANT_OBJECT_MODEL_NAME", "test-detector")
    monkeypatch.setattr(pi, "load_plant_detector", lambda: lambda image, threshold: [])
    expected = {"status": "success", "label": "fern", "score": 0.8}
    monkeypatch.setattr(pi, "identify_plant", lambda image, threshold=None: dict(expected))
    result = pi.identify_plants(pil_image)
    assert result["detection_status"] == "no_instances"
    assert result["label"] == "fern"
    assert "whole-photo classification" in result["detection_message"]
    assert result["instances"] == []


def test_plant_detector_error_is_reported_and_whole_photo_fallback_kept(monkeypatch, pil_image):
    monkeypatch.setattr(pi, "PLANT_OBJECT_MODEL_NAME", "test-detector")

    def fail():
        raise OSError("model files missing")

    monkeypatch.setattr(pi, "load_plant_detector", fail)
    monkeypatch.setattr(pi, "identify_plant", lambda image, threshold=None: {
        "status": "success", "label": "fern", "score": 0.8,
    })
    result = pi.identify_plants(pil_image)
    assert result["detection_status"] == "unavailable"
    assert result["status"] == "success"
    assert "model files missing" not in result["detection_message"]


def test_optional_plant_detector_is_skipped_by_default(monkeypatch, pil_image):
    monkeypatch.setattr(pi, "PLANT_OBJECT_MODEL_NAME", "")
    expected = {"status": "success", "label": "fern", "score": 0.8}
    monkeypatch.setattr(pi, "identify_plant", lambda image, threshold=None: dict(expected))

    def detector_must_not_load():
        raise AssertionError("optional detector should be skipped")

    monkeypatch.setattr(pi, "load_plant_detector", detector_must_not_load)
    result = pi.identify_plants(pil_image)
    assert result["detection_status"] == "disabled"
    assert result["label"] == "fern"
    assert "disabled to keep model memory use lower" in result["detection_message"]


def test_model_load_failure(monkeypatch, pil_image):
    patch_model(monkeypatch, load_exc=OSError("no network"))
    r = pi.identify_plant(pil_image)
    assert r["status"] == "unavailable" and "label" not in r


def test_inference_failure(monkeypatch, pil_image):
    patch_model(monkeypatch, Pipe(exc=RuntimeError("x")))
    assert pi.identify_plant(pil_image)["status"] == "error"


@pytest.mark.parametrize("preds", [[], [{"label": "", "score": 0.9}], [{"label": "a", "score": 5}],
                                   [{"label": "a", "score": float("nan")}]])
def test_invalid_output(monkeypatch, pil_image, preds):
    patch_model(monkeypatch, Pipe(preds))
    assert pi.identify_plant(pil_image)["status"] == "error"


def test_predict_species_raises_on_failure(monkeypatch, pil_image):
    patch_model(monkeypatch, load_exc=OSError("x"))
    with pytest.raises(RuntimeError):
        pi.predict_species(pil_image)


@pytest.mark.parametrize("label,title,ok", [
    ("Ginkgo", "Ginkgo biloba", True), ("Oak", "Rosemary", False),
    ("Rose", "Rose (disambiguation)", False), ("", "Anything", False),
])
def test_title_matching(label, title, ok):
    assert pi.title_matches_label(label, title) is ok


@pytest.mark.parametrize("label,expected", [
    ("Haemanthus albiflos Jacq.", "Haemanthus albiflos"),
    ("Citrus x paradisi Macfad", "Citrus paradisi"),
    ("Ginkgo biloba", "Ginkgo biloba"),
    ("tomato", "tomato"),
])
def test_canonical_species_name_removes_authority(label, expected):
    assert pi.canonical_species_name(label) == expected


class Resp:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self.payload


def test_wikipedia_first_hit_not_trusted(monkeypatch):
    pi.resolve_wikipedia_title.clear()
    hits = {"query": {"search": [{"title": "Unrelated thing"}, {"title": "Ginkgo biloba"}]}}
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(hits))
    assert pi.resolve_wikipedia_title("Ginkgo") == "Ginkgo biloba"


def test_wikipedia_lookup_strips_taxonomic_author(monkeypatch):
    pi.resolve_wikipedia_title.clear()
    calls = {}

    def fake_get(url, params, **kwargs):
        calls.update(params)
        return Resp({"query": {"search": [{"title": "Haemanthus albiflos"}]}})

    monkeypatch.setattr(requests, "get", fake_get)
    assert pi.resolve_wikipedia_title("Haemanthus albiflos Jacq.") == "Haemanthus albiflos"
    assert calls["srsearch"] == "Haemanthus albiflos"


def test_wikipedia_no_match_returns_none_and_fallback_name(monkeypatch):
    pi.resolve_wikipedia_title.clear()
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp({"query": {"search": [{"title": "Zebra"}]}}))
    assert pi.resolve_wikipedia_title("Fern") is None
    assert pi.get_wikipedia_title("Fern") == "Fern"


@pytest.mark.parametrize("exc", [requests.Timeout(), requests.ConnectionError()])
def test_wikipedia_failures(monkeypatch, exc):
    pi.resolve_wikipedia_title.clear()

    def boom(*a, **k):
        raise exc

    monkeypatch.setattr(requests, "get", boom)
    assert pi.resolve_wikipedia_title("Tulip") is None


def test_wikipedia_malformed_json(monkeypatch):
    pi.resolve_wikipedia_title.clear()
    monkeypatch.setattr(requests, "get", lambda *a, **k: Resp(["not", "a", "dict"]))
    assert pi.resolve_wikipedia_title("Lily") is None


def test_translation_failure_preserves_text(monkeypatch):
    pi.translate_text.clear()

    def boom(*a, **k):
        raise requests.Timeout()

    monkeypatch.setattr(requests, "get", boom)
    assert pi.translate_text("Healthy leaf", "hi") == "Healthy leaf"
    assert pi.translate_text("Healthy leaf", "en") == "Healthy leaf"


def test_scientific_name_only_from_species_rank():
    assert pi.scientific_name_from_taxonomy({"Species": "Ginkgo biloba", "Genus": "Ginkgo"}) == "Ginkgo biloba"
    assert pi.scientific_name_from_taxonomy({"Genus": "Ginkgo"}) is None
    assert pi.scientific_name_from_taxonomy({"error": "x"}) is None
    assert pi.scientific_name_from_taxonomy(None) is None
