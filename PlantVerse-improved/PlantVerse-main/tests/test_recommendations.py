import pytest

from disease_recommendations import (
    DISEASE_DATABASE,
    RAW_LABEL_MAP,
    get_disease_info,
    normalize_disease_label,
)
from crop_compatibility import species_matches_crop


def test_all_raw_labels_map_to_database():
    for raw in RAW_LABEL_MAP:
        assert normalize_disease_label(raw) in DISEASE_DATABASE


@pytest.mark.parametrize("label,expected", [
    ("Tomato___Early_blight", "Tomato with Early Blight"),
    ("tomato early blight", "Tomato with Early Blight"),
    ("  Potato___healthy ", "Healthy Potato Plant"),
])
def test_exact_normalisation(label, expected):
    assert normalize_disease_label(label) == expected


@pytest.mark.parametrize("label", ["Tomato", "Early blight", "blight", "healthy", "", None, 5, "Tomato___Unknown"])
def test_no_partial_matches(label):
    assert normalize_disease_label(label) is None


def test_unknown_returns_explicit_unsupported():
    info = get_disease_info("Mystery rot")
    assert info["supported"] is False and "recommendations" not in info and "symptoms" not in info


def test_known_has_required_fields():
    info = get_disease_info("Tomato___Late_blight")
    assert info["supported"] and info["crop"] == "Tomato"
    for key in ("symptoms", "possible_causes", "preventive_measures", "next_steps", "references", "expert_note"):
        assert key in info


@pytest.mark.parametrize("species,crop,expected", [
    ("Solanum lycopersicum L.", "Tomato", True),
    ("Solanum tuberosum", "Tomato", False),
    ("Solanum tuberosum", "Potato", True),
    ("Prunus persica (L.) Batsch", "Peach", True),
    ("Citrus x paradisi Macfad", "Orange", True),
    ("Haemanthus albiflos Jacq.", "Tomato", False),
    ("Haemanthus albiflos Jacq.", "Haemanthus", False),
    (None, "Tomato", False),
])
def test_species_matches_disease_crop(species, crop, expected):
    assert species_matches_crop(species, crop) is expected
