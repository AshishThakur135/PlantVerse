import json

import diagnosis_store as store

PLANT = {"status": "success", "label": "Tomato", "score": 0.9}
DISEASE = {"status": "success", "disease_name": "Tomato with Early Blight", "confidence": 0.8,
           "health_status": "Diseased", "recommendations": ["Remove infected leaves"]}


def test_build_record_marks_failed_models():
    rec = store.build_record({"status": "unavailable"}, DISEASE, b"img")
    assert rec["plant_label"] is None and rec["plant_status"] == "unavailable"
    assert rec["disease_name"] == "Tomato with Early Blight" and len(rec["image_sha256"]) == 64


def test_save_and_retrieve(tmp_path):
    db = tmp_path / "h.db"
    rec = store.build_record(PLANT, DISEASE, b"img", "Solanum lycopersicum")
    assert store.save_record(rec, db) is True
    got = store.get_record(rec["id"], db)
    assert got["recommendations"] == ["Remove infected leaves"] and got["scientific_name"] == "Solanum lycopersicum"
    assert [r["id"] for r in store.list_records(path=db)] == [rec["id"]]
    assert store.delete_record(rec["id"], db) is True and store.list_records(path=db) == []


def test_persistence_disabled_never_claims_saved(monkeypatch):
    monkeypatch.setenv("PLANTVERSE_PERSIST_HISTORY", "0")
    assert store.save_record(store.build_record(PLANT, DISEASE)) is False
    assert store.list_records() == []


def test_persistence_enabled_writes(monkeypatch, tmp_path):
    monkeypatch.setenv("PLANTVERSE_PERSIST_HISTORY", "1")
    monkeypatch.setattr(store, "DB_PATH", tmp_path / "x.db")
    rec = store.build_record(PLANT, DISEASE)
    assert store.save_record(rec) is True and store.list_records()[0]["id"] == rec["id"]


def test_save_failure_returns_false(tmp_path):
    bad = tmp_path / "f"
    bad.mkdir()  # a directory cannot be opened as a database
    assert store.save_record(store.build_record(PLANT, DISEASE), bad) is False


def test_export_formats():
    rec = store.build_record(PLANT, DISEASE)
    assert json.loads(store.export_report(rec, "json"))["plant_label"] == "Tomato"
    txt = store.export_report(rec)
    assert "Tomato" in txt and "not a definitive diagnosis" in txt
