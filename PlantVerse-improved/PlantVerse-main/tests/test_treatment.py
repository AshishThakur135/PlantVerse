import logging

import pytest

import treatment_service
from disease_recommendations import DISEASE_DATABASE, RAW_LABEL_MAP
from plant_disease_detection import _build_result


TOMATO = {
    "status": "success",
    "label": "Solanum lycopersicum",
    "score": 0.93,
    "threshold": 0.5,
}


def test_treatment_database_covers_verified_mappings_and_fields():
    entries, error = treatment_service._load_database()
    assert error is None
    assert set(entries) == {"Tomato with Late Blight", "Potato with Late Blight"}
    for raw_label in RAW_LABEL_MAP:
        assert raw_label in RAW_LABEL_MAP
        assert RAW_LABEL_MAP[raw_label] in DISEASE_DATABASE
    for entry in entries.values():
        assert entry["source_url"].startswith("https://")
        assert entry["source_organization"] and entry["region"]
        assert entry["symptoms"] and entry["prevention"]
        assert entry["chemical_options"] == []


def test_verified_late_blight_returns_treatment_for_compatible_tomato():
    disease = _build_result([("Tomato___Late_blight", 0.91)], "test", 0.5)
    result = treatment_service.get_treatment(disease, TOMATO)
    assert result["status"] == "available"
    assert result["crop"] == "Tomato"
    assert result["references"][0]["organization"].startswith("Cornell University")
    assert result["chemical_options"] == []


def test_healthy_prediction_has_prevention_but_no_disease_treatment():
    disease = _build_result([("Tomato___healthy", 0.91)], "test", 0.5)
    result = treatment_service.get_treatment(disease, TOMATO)
    assert result["status"] == "healthy"
    assert result["immediate_actions"] == []
    assert "not proof" in result["message"]
    assert result["prevention"]


def test_uncertain_disease_never_gets_treatment():
    disease = _build_result([("Tomato___Late_blight", 0.3)], "test", 0.5)
    result = treatment_service.get_treatment(disease, TOMATO)
    assert result["status"] == "uncertain"
    assert result["immediate_actions"] == []


def test_identity_conflict_withholds_crop_specific_treatment():
    disease = _build_result([("Tomato___Late_blight", 0.91)], "test", 0.5)
    pothos = {**TOMATO, "label": "Epipremnum aureum"}
    result = treatment_service.get_treatment(disease, pothos)
    assert result["status"] == "conflict"
    assert result["immediate_actions"] == []
    assert result["chemical_options"] == []
    assert result["safety_notes"]


def test_unconfirmed_identity_withholds_crop_specific_treatment():
    disease = _build_result([("Tomato___Late_blight", 0.91)], "test", 0.5)
    result = treatment_service.get_treatment(disease, {**TOMATO, "status": "uncertain"})
    assert result["status"] == "identity_unconfirmed"
    assert result["immediate_actions"] == []


def test_missing_verified_entry_is_logged_and_has_safe_fallback(caplog):
    disease = _build_result([("Tomato___Early_blight", 0.91)], "test", 0.5)
    with caplog.at_level(logging.WARNING, logger="treatment_service"):
        result = treatment_service.get_treatment(disease, TOMATO)
    assert result["status"] == "unverified"
    assert "No verified treatment mapping" in caplog.text
    assert result["immediate_actions"] == []
    assert result["safety_notes"]


def test_missing_treatment_database_is_explicit_and_logged(monkeypatch, caplog, tmp_path):
    treatment_service._load_database.cache_clear()
    monkeypatch.setattr(treatment_service, "_DATABASE_PATH", tmp_path / "missing.json")
    disease = _build_result([("Tomato___Late_blight", 0.91)], "test", 0.5)
    with caplog.at_level(logging.ERROR, logger="treatment_service"):
        result = treatment_service.get_treatment(disease, TOMATO)
    assert result["status"] == "database_error"
    assert "could not be loaded" in result["message"]
    assert "Could not load treatment database" in caplog.text
    treatment_service._load_database.cache_clear()


@pytest.mark.parametrize("raw", list(RAW_LABEL_MAP))
def test_every_model_label_has_an_exact_canonical_mapping(raw):
    assert RAW_LABEL_MAP[raw] in DISEASE_DATABASE
