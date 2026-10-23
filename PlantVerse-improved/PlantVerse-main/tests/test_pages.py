import os
import re

import pytest
from streamlit.testing.v1 import AppTest

import plant_disease_detection as eng
import plant_identification as pi
from tests.conftest import ROOT, make_image

PLANT_OK = {"status": "success", "success": True, "label": "tomato", "score": 0.93, "threshold": 0.5,
            "message": "", "candidates": [{"label": "tomato", "score": 0.93}, {"label": "potato", "score": 0.04}]}
PLANT_FAIL = {"status": "unavailable", "success": False, "message": "Plant model could not be loaded."}
DISEASE_OK = eng._build_result([
    ("Tomato___Early_blight", 0.9),
    ("Tomato___Late_blight", 0.05),
    ("Tomato___healthy", 0.03),
], "test", 0.5)
TOMATO_LATE_BLIGHT = eng._build_result([("Tomato___Late_blight", 0.9)], "test", 0.5)
DISEASE_FAIL = {"success": False, "status": "unavailable", "message": "Disease model could not be loaded.",
                "backend": "none", "recommendations": [], "preventive_measures": []}


@pytest.fixture
def patched(monkeypatch):
    def apply(plant, disease):
        def identify(image, threshold=None):
            result = dict(plant)
            if "detection_status" not in result:
                result["detection_status"] = "detected" if result.get("label") else "unavailable"
            if "instances" not in result:
                result["instances"] = (
                    [{**result, "detection_score": 0.9, "image": None}]
                    if result.get("label") else []
                )
            return result

        monkeypatch.setattr(pi, "identify_plants", identify)
        monkeypatch.setattr(eng, "analyze_disease", lambda image, threshold=None: disease)
        monkeypatch.setattr(pi, "resolve_wikipedia_title", lambda name: None)  # no network

    return apply


def run_with_upload(data=None, name="leaf.png"):
    at = AppTest.from_file(os.path.join(ROOT, "app_pages", "analysis.py"), default_timeout=60).run()
    if data is not None:
        at.file_uploader[0].upload(name, data, "image/png")
        at.run()
    return at


def texts(at):
    return " ".join(e.value for e in list(at.markdown) + list(at.error) + list(at.warning) + list(at.info) + list(at.caption))


def test_navigation_has_no_disease_page_and_targets_exist():
    app_src = open("app.py").read()
    assert "disease_detection" not in app_src and "Plant Analysis" in app_src
    for title in ("Explore", "Plant Library", "AR Explorer", "About"):
        assert f'title="{title}"' in app_src
    for root, _, files in os.walk("."):
        if ".git" in root or "tests" in root:
            continue
        for f in files:
            if f.endswith(".py"):
                src = open(os.path.join(root, f), encoding="utf-8").read()
                for target in re.findall(r'switch_page\("([^"]+)"\)', src):
                    assert os.path.exists(target), f"{f} links to missing page {target}"


@pytest.mark.parametrize("page", ["explore", "library", "about", "ar_explorer", "analysis"])
def test_pages_render_without_exceptions(page):
    at = AppTest.from_file(os.path.join(ROOT, "app_pages", f"{page}.py"), default_timeout=60).run()
    assert not at.exception


def test_single_uploader_and_analyze_disabled_without_upload():
    at = run_with_upload()
    assert len(at.file_uploader) == 1
    assert [b.label for b in at.button] == ["Analyze Plant"] and at.button[0].disabled


def test_corrupt_upload_shows_error_and_disables_button():
    at = run_with_upload(b"not an image")
    assert any("not a readable image" in e.value for e in at.error) and at.button[0].disabled


def test_analysis_shows_both_sections(patched):
    patched(PLANT_OK, DISEASE_OK)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    assert not at.exception
    body = texts(at)
    assert "Plant Identity" in body and "Disease Detection" in body and "Tomato Early Blight" in body
    assert "Recommended Treatment" in body and "Safety Notes" in body and "References" in body
    assert "Tomato with Late Blight (5.0%)" in body
    assert "Scientific name not available" in body  # never invented


def test_optional_plant_details_do_not_delay_analysis(patched, monkeypatch):
    patched(PLANT_OK, DISEASE_OK)
    calls = []
    original_lookup = pi.resolve_wikipedia_title

    def track_lookup(name):
        calls.append(name)
        return original_lookup(name)

    monkeypatch.setattr(pi, "resolve_wikipedia_title", track_lookup)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    assert not at.exception
    assert calls == []
    at.button(key="load_plant_details").click().run()
    assert calls == ["tomato"]
    assert not at.exception


def test_treatment_requires_reliable_compatible_species(patched):
    identified = {
        **PLANT_OK,
        "label": "Solanum lycopersicum",
        "score": 0.93,
        "threshold": 0.5,
    }
    patched(identified, TOMATO_LATE_BLIGHT)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    disease = at.session_state["analysis_result"]["disease"]
    assert disease["treatment_eligible"] is True
    assert disease["recommendations"]
    assert "Cornell University" in body
    assert disease["treatment"]["status"] == "available"
    assert disease["treatment"]["chemical_options"] == []


def test_treatment_withheld_when_species_is_unreliable(patched):
    uncertain_plant = {
        **PLANT_OK,
        "label": "Solanum lycopersicum",
        "score": 0.1,
        "threshold": 0.5,
        "status": "uncertain",
    }
    patched(uncertain_plant, TOMATO_LATE_BLIGHT)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    assert "Tomato Late Blight" in body
    assert "withheld" in body
    disease = at.session_state["analysis_result"]["disease"]
    assert disease["treatment_eligible"] is False
    assert not disease["recommendations"]
    assert disease["treatment"]["status"] == "identity_unconfirmed"
    assert "Do not apply a pesticide" in body


def test_treatment_withheld_when_species_and_disease_crop_conflict(patched):
    identified = {
        **PLANT_OK,
        "label": "Haemanthus albiflos Jacq.",
        "score": 0.93,
        "threshold": 0.5,
    }
    patched(identified, TOMATO_LATE_BLIGHT)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    assert "Tomato Late Blight" in body
    assert "conflicts with the identified plant" in body
    disease = at.session_state["analysis_result"]["disease"]
    assert disease["treatment_eligible"] is False
    assert not disease["recommendations"]
    assert disease["treatment"]["status"] == "conflict"
    assert "Class prediction - confirm before treating" in body
    assert "Cornell University" in body
    assert "not confirmation for the identified plant" in body
    assert "Do not apply a pesticide" in body


def test_verified_disease_recommendation_is_visible_for_matching_plant(patched):
    tomato = {
        **PLANT_OK,
        "label": "Solanum lycopersicum",
        "score": 0.93,
        "threshold": 0.5,
    }
    patched(tomato, TOMATO_LATE_BLIGHT)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    assert "Cornell advises removing affected plants" in body
    assert "Rotate away from solanaceous crops for three years." in body
    assert "No pesticide is recommended here." in body


def test_missing_treatment_entry_shows_safe_fallback(caplog, patched):
    tomato = {
        **PLANT_OK,
        "label": "Solanum lycopersicum",
        "score": 0.93,
        "threshold": 0.5,
    }
    patched(tomato, DISEASE_OK)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    disease = at.session_state["analysis_result"]["disease"]
    assert disease["treatment"]["status"] == "unverified"
    assert "No verified treatment entry is available yet" in body
    assert "No verified treatment mapping" in caplog.text
    assert not disease["recommendations"]


def test_healthy_result_is_not_a_health_guarantee_and_shows_prevention(patched):
    tomato = {
        **PLANT_OK,
        "label": "Solanum lycopersicum",
        "score": 0.93,
        "threshold": 0.5,
    }
    healthy = eng._build_result([("Tomato___healthy", 0.9)], "test", 0.5)
    patched(tomato, healthy)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    assert "No supported disease detected" in body
    assert "not proof that the plant is healthy" in body
    assert "Inspect plants regularly" in body


def test_same_uploaded_image_is_passed_to_both_models(monkeypatch):
    observed = {}

    def identify(image, threshold=None):
        observed["plant_image"] = image
        return {**PLANT_OK, "detection_status": "detected",
                "instances": [{**PLANT_OK, "detection_score": 0.9}]}

    def disease(image, threshold=None):
        observed["disease_image"] = image
        return DISEASE_OK

    monkeypatch.setattr(pi, "identify_plants", identify)
    monkeypatch.setattr(eng, "analyze_disease", disease)
    monkeypatch.setattr(pi, "resolve_wikipedia_title", lambda name: None)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    assert observed["plant_image"] is observed["disease_image"]


def test_multiple_detected_plants_show_individual_confidence_and_withhold_treatment(patched):
    first = {
        **PLANT_OK,
        "label": "Solanum lycopersicum",
        "score": 0.91,
        "threshold": 0.5,
        "detection_score": 0.93,
    }
    second = {
        **PLANT_OK,
        "label": "Epipremnum aureum",
        "score": 0.87,
        "threshold": 0.5,
        "detection_score": 0.89,
    }
    multiple = {
        **first,
        "detection_status": "detected",
        "detection_message": "Localized 2 separate potted-plant object(s).",
        "instances": [first, second],
    }
    patched(multiple, TOMATO_LATE_BLIGHT)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    disease = at.session_state["analysis_result"]["disease"]
    assert "Identified plants (2)" in body
    assert "Solanum lycopersicum" in body and "Epipremnum aureum" in body
    metric_values = [metric.value for metric in at.metric]
    assert "91.0000%" in metric_values and "87.0000%" in metric_values
    assert disease["treatment"]["status"] == "identity_unconfirmed"
    assert "cannot attribute a result to one plant" in disease["treatment"]["message"]


def test_plant_model_failure_still_shows_disease(patched):
    patched(PLANT_FAIL, DISEASE_OK)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    assert "Plant model could not be loaded" in body and "Tomato Early Blight" in body


def test_disease_model_failure_still_shows_plant(patched):
    patched(PLANT_OK, DISEASE_FAIL)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    body = texts(at)
    assert "Disease model could not be loaded" in body and "tomato" in body.lower()
    assert "Healthy" not in body  # no fabricated healthy result
    assert "General care guidance" not in body
    assert "Do not apply a pesticide" in body


def test_save_session_only_is_honest_and_reset_works(patched):
    patched(PLANT_OK, DISEASE_OK)
    at = run_with_upload(make_image("PNG"))
    at.button(key="analyze_plant").click().run()
    at.button(key="save_diag").click().run()
    assert any("browser session only" in i.value for i in at.info)
    assert len(at.session_state["disease_diagnoses"]) == 1
    at.button(key="reset_analysis").click().run()
    assert at.session_state["analysis_result"] is None
