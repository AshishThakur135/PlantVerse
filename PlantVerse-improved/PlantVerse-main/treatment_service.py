"""Evidence-aware treatment lookup for PlantVillage disease predictions."""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from crop_compatibility import species_matches_crop
from disease_recommendations import normalize_disease_label

log = logging.getLogger(__name__)
_DATABASE_PATH = Path(__file__).resolve().parent / "data" / "treatment_database.json"
_REQUIRED_FIELDS = {
    "crop",
    "disease",
    "symptoms",
    "immediate_actions",
    "non_chemical_management",
    "prevention",
    "chemical_options",
    "source_organization",
    "source_url",
    "region",
    "safety_notes",
}
_GENERAL_CARE = [
    "Do not apply a pesticide or fungicide based only on an AI prediction.",
    "Use clean tools and avoid moving suspect leaves or soil between plants.",
    "Photograph the affected area clearly and ask a local extension service or plant pathologist to confirm the cause.",
]
_GENERAL_PREVENTION = [
    "Inspect plants regularly and keep notes or photographs of changes.",
    "Use clean tools and follow watering, light, and drainage needs for the confirmed plant species.",
]


@lru_cache(maxsize=1)
def _load_database() -> tuple[Dict[str, Dict[str, Any]], Optional[str]]:
    """Load and validate only explicitly sourced entries."""
    try:
        with _DATABASE_PATH.open(encoding="utf-8") as treatment_file:
            entries = json.load(treatment_file)
    except (OSError, json.JSONDecodeError) as exc:
        log.exception("Could not load treatment database at %s", _DATABASE_PATH)
        return {}, f"The verified treatment database could not be loaded ({type(exc).__name__})."

    if not isinstance(entries, dict):
        log.error("Treatment database root must be an object: %s", _DATABASE_PATH)
        return {}, "The verified treatment database has an invalid format."

    valid: Dict[str, Dict[str, Any]] = {}
    for label, entry in entries.items():
        if not isinstance(label, str) or normalize_disease_label(label) is None or not isinstance(entry, dict):
            log.error("Ignoring invalid treatment database mapping for %r", label)
            continue
        missing = _REQUIRED_FIELDS - entry.keys()
        if missing:
            log.error("Ignoring treatment mapping %r; missing fields: %s", label, sorted(missing))
            continue
        if not isinstance(entry["chemical_options"], list) or not isinstance(entry["source_url"], str) \
                or not entry["source_url"].startswith("https://"):
            log.error("Ignoring treatment mapping %r; source or chemical options are invalid", label)
            continue
        if any(not isinstance(entry[field], list) for field in (
            "immediate_actions", "non_chemical_management", "prevention", "safety_notes"
        )):
            log.error("Ignoring treatment mapping %r; action fields must be lists", label)
            continue
        valid[normalize_disease_label(label)] = entry
    return valid, None


def _base_result(status: str, message: str, prediction: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "status": status,
        "message": message,
        "label": normalize_disease_label(prediction.get("raw_label")),
        "crop": prediction.get("plant"),
        "disease": prediction.get("disease"),
        "confidence": prediction.get("confidence"),
        "threshold": prediction.get("threshold"),
        "symptoms": None,
        "immediate_actions": [],
        "non_chemical_management": [],
        "prevention": list(_GENERAL_PREVENTION),
        "chemical_options": [],
        "safety_notes": list(_GENERAL_CARE),
        "references": [],
    }


def get_treatment(
    prediction: Dict[str, Any],
    plant_identity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return verified advice only when model and identified crop are reliable and compatible.

    Softmax confidence is treated as a screening score, not a calibrated probability.
    """
    plant_identity = plant_identity if isinstance(plant_identity, dict) else {}
    status = prediction.get("status")
    result = _base_result("unavailable", "", prediction)

    if status in ("unavailable", "error"):
        result.update(
            status=status,
            message=prediction.get("message") or "Disease analysis did not complete; no diagnosis was made.",
        )
        return result
    if status == "unsupported":
        result.update(
            status="unsupported",
            message=prediction.get("message") or "The model label has no supported disease mapping.",
        )
        return result
    if status == "uncertain":
        result.update(
            status="uncertain",
            message="The disease prediction is below the configured confidence threshold. "
                    "No disease-specific treatment is recommended.",
        )
        return result
    if status != "success":
        result.update(status="error", message="The disease result has an unknown status; no treatment was selected.")
        return result
    if prediction.get("health_status") == "Healthy":
        result.update(
            status="healthy",
            message="The model did not detect a supported disease. This is not proof that the plant is healthy.",
        )
        return result

    label = normalize_disease_label(prediction.get("raw_label"))
    if label is None:
        result.update(
            status="unsupported",
            message="This disease label has no exact PlantVerse mapping; disease-specific advice is unavailable.",
        )
        return result

    entries, load_error = _load_database()
    if load_error:
        result.update(status="database_error", message=load_error)
        return result
    entry = entries.get(label)
    if entry is None:
        log.warning("No verified treatment mapping for supported disease label %r", label)
    else:
        result["references"] = [{
            "organization": entry["source_organization"],
            "url": entry["source_url"],
            "region": entry["region"],
        }]
    result["treatment_entry_available"] = entry is not None

    identity_reliable = (
        plant_identity.get("status") == "success"
        and isinstance(plant_identity.get("label"), str)
    )
    try:
        identity_reliable = identity_reliable and float(plant_identity.get("score", 0.0)) >= float(
            plant_identity.get("threshold", 0.5)
        )
    except (TypeError, ValueError):
        identity_reliable = False

    compatible = identity_reliable and species_matches_crop(plant_identity.get("label"), prediction.get("plant"))
    result["plant_compatible"] = bool(compatible)
    if not identity_reliable:
        missing_entry = " No verified disease-specific treatment entry exists yet." if entry is None else ""
        identity_message = plant_identity.get("message") or (
            "Plant identity is uncertain or unavailable."
        )
        result.update(
            status="identity_unconfirmed",
            message=f"{identity_message} Crop-specific treatment is withheld; general precautions "
                    f"are provided.{missing_entry}",
        )
        return result
    if not compatible:
        missing_entry = " No verified disease-specific treatment entry exists yet." if entry is None else ""
        result.update(
            status="conflict",
            message=f"The disease model predicts {prediction.get('plant') or 'an unsupported crop'}, "
                    f"which conflicts with the identified plant ({plant_identity.get('label')}). "
                    f"Crop-specific treatment is withheld.{missing_entry}",
        )
        return result

    if entry is None:
        result.update(
            status="unverified",
            message=f"No verified treatment entry is available yet for {label}. "
                    "Only general care guidance is shown; no chemical treatment is suggested.",
        )
        return result

    result.update({
        "status": "available",
        "message": "Verified, region-specific non-chemical guidance is available.",
        "crop": entry["crop"],
        "disease": entry["disease"],
        "symptoms": entry["symptoms"],
        "immediate_actions": list(entry["immediate_actions"]),
        "non_chemical_management": list(entry["non_chemical_management"]),
        "prevention": list(entry["prevention"]),
        "chemical_options": list(entry["chemical_options"]),
        "safety_notes": list(entry["safety_notes"]),
    })
    return result


def general_care_guidance() -> List[str]:
    """Safe, non-disease-specific steps for uncertain, unsupported, or failed analyses."""
    return list(_GENERAL_CARE)


def general_prevention_guidance() -> List[str]:
    """Safe, species-agnostic prevention guidance."""
    return list(_GENERAL_PREVENTION)
