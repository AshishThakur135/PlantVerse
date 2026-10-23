"""Plant Analysis - one upload, two independent analyses (identity + health)."""

import hashlib
from typing import Any, Dict, Optional

import streamlit as st

from treatment_service import (
    general_care_guidance,
    general_prevention_guidance,
    get_treatment,
)
from crop_compatibility import species_matches_crop
import diagnosis_store as store
from ar_location_plant_map import PlantLocationFinder
from image_utils import ImageValidationError, to_jpeg_bytes, validate_image
from plant_disease_detection import analyze_disease
from plant_identification import (
    LANGUAGES,
    get_medicinal_uses_from_wikipedia,
    get_taxonomy_from_wikidata,
    identify_plants,
    resolve_wikipedia_title,
    scientific_name_from_taxonomy,
    translate_text,
)
from ui.chrome import init_state, kicker, section_rule
from ui.presentation import render_taxonomy, upsert_library_entry

init_state()
st.session_state.setdefault("analysis_result", None)
st.session_state.setdefault("analysis_nonce", 0)

STATUS_LABELS = {
    "success": ("✅", "Confident result", "green"),
    "uncertain": ("⚠️", "Uncertain - low confidence", "orange"),
    "unsupported": ("❓", "Unsupported label", "orange"),
    "unavailable": ("⛔", "Model unavailable", "red"),
    "error": ("❌", "Error", "red"),
}


def tr(text: str) -> str:
    return translate_text(text, lang_code)


def status_badge(status: str) -> None:
    icon, label, color = STATUS_LABELS.get(status, ("❔", status, "gray"))
    st.markdown(f":{color}-badge[{icon} {tr(label)}]")  # icon + text, never colour alone


def lookup_plant_details(label: str) -> Dict[str, Any]:
    """Best-effort Wikipedia/Wikidata enrichment; each step may fail independently."""
    details: Dict[str, Any] = {"wiki_title": None, "taxonomy": None, "scientific_name": None,
                               "description": "", "medicinal": None, "notes": []}
    title = resolve_wikipedia_title(label)
    if not title:
        details["notes"].append("No verified Wikipedia article matched this name, so extra details are omitted.")
        return details
    details["wiki_title"] = title
    try:
        details["taxonomy"] = get_taxonomy_from_wikidata(title)
        details["scientific_name"] = scientific_name_from_taxonomy(details["taxonomy"])
    except Exception:  # noqa: BLE001
        details["notes"].append("Taxonomy lookup (Wikidata) failed.")
    try:
        details["description"] = PlantLocationFinder().get_wikipedia_summary(title) or ""
    except Exception:  # noqa: BLE001
        details["notes"].append("Wikipedia summary could not be loaded.")
    try:
        details["medicinal"] = get_medicinal_uses_from_wikipedia(title, target_lang="en")
    except Exception:  # noqa: BLE001
        pass
    return details


def run_analysis(image, image_bytes: bytes, file_id: str, progress=None) -> Dict[str, Any]:
    """Identify localized plant instances and run disease analysis on the same photo."""
    if progress is not None:
        progress.write(tr("Identifying the plant from your photo..."))
    plant = identify_plants(image)
    if progress is not None:
        progress.write(tr("Checking plant health with the disease model..."))
    disease = dict(analyze_disease(image))
    instances = plant.get("instances") or []
    treatment_identity = dict(plant)
    if plant.get("detection_status") != "detected" or len(instances) != 1:
        treatment_identity["status"] = "uncertain"
        treatment_identity["message"] = (
            "The disease model analyzes the full image and cannot attribute a result to one plant "
            "when separate plant instances are not localized or multiple plants are present."
        )
    try:
        reliable_plant = (
            treatment_identity.get("status") == "success"
            and float(treatment_identity.get("score", 0.0)) >= float(treatment_identity.get("threshold", 0.5))
        )
    except (TypeError, ValueError):
        reliable_plant = False
    disease_crop = disease.get("plant")
    compatible = reliable_plant and species_matches_crop(treatment_identity.get("label"), disease_crop)
    treatment = get_treatment(disease, treatment_identity)
    treatment_eligible = treatment["status"] == "available"
    disease["plant_compatible"] = bool(compatible)
    disease["treatment_eligible"] = bool(treatment_eligible)
    disease["treatment"] = treatment
    disease["treatment_message"] = treatment["message"]
    disease["recommendations"] = treatment["immediate_actions"] + treatment["non_chemical_management"]
    disease["preventive_measures"] = (
        treatment["prevention"] if treatment_eligible else general_prevention_guidance()
    )
    disease["references"] = [
        reference["url"] for reference in treatment["references"]
    ]
    return {"file_id": file_id, "plant": plant, "details": {}, "disease": disease,
            "image_bytes": image_bytes, "saved": None, "record": None}


# ------------------------------------------------------------------ header
head_l, head_r = st.columns([1.35, 0.65], vertical_alignment="top")
with head_r:
    with st.container(border=True):
        selected_lang = st.selectbox("Language", list(LANGUAGES.keys()), key="analysis_language")
        lang_code = LANGUAGES[selected_lang]
with head_l:
    kicker(tr("Plant Analysis"), key="kicker-analysis")
    st.header(tr("Plant Analysis"))
    st.markdown(tr(
        "Upload one clear photo of a plant or leaf. PlantVerse will estimate which plant it is and "
        "check its health. Results are AI estimates, not a definitive diagnosis."
    ))

section_rule("section-rule-analysis-input")

# ------------------------------------------------------------------ input
up_col, tips_col = st.columns([1.08, 0.92], gap="large")
image = image_bytes = None
file_id: Optional[str] = None
with up_col:
    with st.container(border=True):
        img_file = st.file_uploader(
            tr("Plant photograph (JPG, PNG or WEBP)"),
            type=["jpg", "jpeg", "png", "webp"],
            key=f"analysis_uploader_{st.session_state.analysis_nonce}",
        )
        if img_file is not None:
            try:
                image = validate_image(img_file)
                image_bytes = to_jpeg_bytes(image)
                file_id = hashlib.sha256(image_bytes).hexdigest()
                st.image(image, caption=f"{img_file.name} • {image.width} × {image.height}px", width="stretch",
                         alt=tr("Uploaded plant photograph"))
            except ImageValidationError as exc:
                st.error(str(exc))
        analyze_clicked = st.button(tr("Analyze Plant"), type="primary", icon=":material/auto_awesome:",
                                    disabled=image is None, key="analyze_plant")
with tips_col:
    with st.container(border=True):
        st.markdown(f"#### :material/photo_camera: {tr('Tips for better results')}")
        for tip in ("Use bright, natural light and avoid glare.",
                    "Fill the frame with one leaf or the whole plant.",
                    "Keep the plant in focus and avoid heavy filters.",
                    "For health checks, show the affected area up close."):
            st.markdown(f"- {tr(tip)}")

# Drop a stale result when a different file is chosen.
current = st.session_state.analysis_result
if current and file_id is not None and current["file_id"] != file_id:
    st.session_state.analysis_result = current = None

if analyze_clicked and image is not None:
    with st.status(tr("Analyzing your plant..."), expanded=True) as progress:
        st.session_state.analysis_result = current = run_analysis(
            image, image_bytes, file_id, progress=progress
        )
        progress.update(label=tr("Analysis complete"), state="complete", expanded=False)

if not current:
    st.stop()

# ------------------------------------------------------------------ results
plant, disease, details = current["plant"], current["disease"], current["details"]
section_rule("section-rule-analysis-results")
kicker(tr("Results"), key="kicker-analysis-results")
id_col, health_col = st.columns(2, gap="large")

with id_col:
    with st.container(border=True):
        instances = plant.get("instances") or []
        if plant.get("detection_message"):
            if plant.get("detection_status") == "detected":
                st.info(tr(plant["detection_message"]))
            else:
                st.warning(tr(plant["detection_message"]))
        if instances:
            identity_heading = (
                tr("Plant Identity") if len(instances) == 1
                else f"{tr('Identified plants')} ({len(instances)})"
            )
            st.markdown(f"### :material/eco: {identity_heading}")
            for index, instance in enumerate(instances, start=1):
                st.markdown(f"#### {tr('Plant')} {index}")
                if instance.get("image") is not None:
                    st.image(instance["image"], width="stretch", alt=f"Detected plant {index} crop")
                if instance.get("label"):
                    status_badge(instance["status"])
                    heading = instance["label"] if instance["status"] == "success" else (
                        f"{instance['label']} ({tr('possible')})"
                    )
                    st.markdown(f"**{tr(heading)}**")
                    st.metric(
                        tr("Species model score"),
                        f"{instance['score'] * 100:.4f}%",
                        help=tr("Softmax score for this detected crop, not a calibrated probability."),
                    )
                    st.metric(
                        tr("Plant detection score"),
                        f"{instance['detection_score'] * 100:.2f}%",
                        help=tr("DETR's COCO potted-plant object-detection score; it is separate from species confidence."),
                    )
                    alternatives = instance.get("candidates", [])[1:4]
                    if alternatives:
                        st.caption(tr("Other species candidates") + ": " + ", ".join(
                            f"{candidate['label']} "
                            f"({candidate.get('confidence_pct', candidate.get('score', 0.0) * 100):.4f}%)"
                            for candidate in alternatives
                        ))
                    if instance["status"] == "uncertain":
                        st.warning(tr(instance["message"]))
                else:
                    st.error(tr(instance.get("message", "This plant crop could not be identified.")))
            if details.get("scientific_name"):
                st.caption(f"{tr('Verified scientific name')}: *{details['scientific_name']}*")
            else:
                st.caption(tr("Scientific name not available from a reliable source."))
        elif "label" in plant:
            st.markdown(f"### :material/eco: {tr('Plant Identity')}")
            status_badge(plant["status"])
            heading = plant["label"] if plant["status"] == "success" else f"{plant['label']} ({tr('possible')})"
            st.markdown(f"## {tr(heading)}")
            sci = details.get("scientific_name")
            st.markdown(f"*{sci}*" if sci else f":gray[{tr('Scientific name not available from a reliable source.')}]")
            st.metric(tr("Whole-photo species score"), f"{plant['score'] * 100:.4f}%",
                      help=tr("Whole-image classification softmax score, not a calibrated probability."))
            if plant["status"] == "uncertain":
                st.warning(tr(plant["message"]))
            others = [candidate for candidate in plant.get("candidates", [])[1:5]]
            if others:
                st.caption(tr("Other species candidates") + ": " + ", ".join(
                    f"{candidate['label']} ({candidate.get('confidence_pct', candidate['score'] * 100):.4f}%)"
                    for candidate in others
                ))
        else:
            st.markdown(f"### :material/eco: {tr('Plant Identity')}")
            st.error(tr(plant["message"]))
        if plant.get("status") == "success" and not current.get("details_loaded"):
            st.caption(tr("Optional plant reference details load from external sources and do not affect this diagnosis."))
            if st.button(tr("Load plant details"), key="load_plant_details", icon=":material/info:"):
                with st.spinner(tr("Loading plant details...")):
                    current["details"] = lookup_plant_details(plant["label"])
                    current["details_loaded"] = True
                st.rerun()
        for note in details.get("notes", []):
            st.caption(f":material/info: {tr(note)}")
        if details.get("description"):
            with st.expander(tr("About this plant")):
                st.markdown(tr(details["description"]))
                st.caption(tr("Source: Wikipedia"))
        if details.get("taxonomy") and "error" not in details["taxonomy"]:
            with st.expander(tr("Taxonomy")):
                render_taxonomy(details["taxonomy"], translate=translate_text, lang_code=lang_code, key_prefix="an")
        if details.get("medicinal"):
            with st.expander(tr("Traditional medicinal notes")):
                for point in details["medicinal"]:
                    st.markdown(f"- {tr(point)}")
                st.caption(tr("Compiled from Wikipedia for reference only; not medical advice."))

with health_col:
    treatment = disease["treatment"]
    with st.container(border=True):
        st.markdown(f"### :material/healing: {tr('Disease Detection')}")
        if disease["status"] == "success" and treatment["status"] in ("conflict", "identity_unconfirmed"):
            st.markdown(f":orange-badge[⚠️ {tr('Class prediction - confirm before treating')}]")
        else:
            status_badge(disease["status"])
        if disease["status"] in ("success", "uncertain"):
            healthy = disease.get("health_status") == "Healthy"
            if healthy:
                st.markdown(f"## {tr('No supported disease detected')}")
                st.markdown(tr("This model result is not proof that the plant is healthy."))
            else:
                name = disease["disease_name"] if disease["status"] == "success" else (
                    f"{disease['disease_name']} ({tr('possible')})"
                )
                st.markdown(f"## {tr(name)}")
                if disease["status"] == "success":
                    st.markdown(f"**{tr('Model estimate')}:** {tr(disease['health_status'])}")
            st.metric(tr("Model confidence"), f"{disease['confidence_pct']}%",
                      help=tr("The model's top softmax score, not a calibrated probability or proof of diagnosis. "
                              "The configured threshold marks lower scores uncertain."))
            st.markdown(f"**{tr('Confidence and Diagnostic Reliability')}:** "
                        f"{tr('Threshold')} {disease.get('threshold', 0.5) * 100:.0f}%. "
                        f"{tr('This model was trained on 38 PlantVillage leaf classes; it can still assign a class '
                              'to unsupported plants or field photos.')}")
            if disease["status"] == "uncertain":
                st.warning(tr(disease["message"]))
            alts = disease.get("top_candidates", [])[1:5]
            if alts:
                st.caption(tr("Other possibilities") + ": " + ", ".join(
                    f"{c['label'] or c['raw_label']} ({c['confidence_pct']}%)" for c in alts))
        elif disease["status"] == "unsupported":
            st.warning(tr(disease["message"]))
            st.caption(f"{tr('Model output')}: {disease.get('raw_label')} ({disease.get('confidence_pct')}%)")
        else:
            st.error(tr(disease["message"]))

        st.markdown(f"#### {tr('Symptoms')}")
        if treatment.get("symptoms"):
            st.markdown(tr(treatment["symptoms"]))
        else:
            st.caption(tr("No disease-specific symptoms are shown because the diagnosis or source mapping is not reliable."))

        st.markdown(f"#### {tr('Recommended Treatment')}")
        if treatment["status"] == "available":
            for item in treatment["immediate_actions"] + treatment["non_chemical_management"]:
                st.markdown(f"- {tr(item)}")
        elif treatment["status"] == "healthy":
            st.info(tr(treatment["message"]))
            st.caption(tr("Preventive guidance is shown below; disease treatment is not indicated by this model output."))
        else:
            st.warning(tr(treatment["message"]))
            for item in general_care_guidance():
                st.markdown(f"- {tr(item)}")

        st.markdown(f"#### {tr('Prevention')}")
        prevention_items = treatment["prevention"] if treatment["status"] == "available" else (
            general_prevention_guidance()
        )
        for item in prevention_items:
            st.markdown(f"- {tr(item)}")

        st.markdown(f"#### {tr('Safety Notes')}")
        for item in treatment["safety_notes"]:
            st.markdown(f"- {tr(item)}")

        st.markdown(f"#### {tr('References')}")
        if treatment["references"]:
            if treatment["status"] != "available":
                st.caption(tr("This reference describes the model-predicted crop and disease only; it is not confirmation for the identified plant."))
            for ref in treatment["references"]:
                st.markdown(f"- [{ref['organization']}]({ref['url']}) — {tr(ref['region'])}")
        elif treatment["status"] == "available":
            st.caption(tr("The verified treatment entry does not include an external reference."))
        else:
            st.caption(tr("No verified disease-specific reference is available for this result."))

        if disease.get("expert_note"):
            st.info(tr(disease["expert_note"]))

st.caption(tr("AI estimates can be wrong. Confirm any diagnosis with a local agricultural extension service or plant pathologist."))

# ------------------------------------------------------------------ save / export / reset
section_rule("section-rule-analysis-actions")
act = st.container(horizontal=True, gap="small")
with act:
    if st.button(tr("Save Diagnosis"), icon=":material/save:", key="save_diag",
                 disabled=current.get("record") is not None):
        record = store.build_record(plant, disease, current["image_bytes"], details.get("scientific_name"))
        persisted = store.save_record(record)
        current["record"], current["saved"] = record, persisted
        entries = [e for e in st.session_state.disease_diagnoses if e.get("id") != record["id"]]
        st.session_state.disease_diagnoses = [store.to_session_entry(record, current["image_bytes"])] + entries[:49]
        if "label" in plant:
            upsert_library_entry({"common_name": plant["label"], "scientific_name": details.get("scientific_name") or "",
                                  "score": plant["score"], "taxonomy": details.get("taxonomy"),
                                  "medicinal": details.get("medicinal"), "description": details.get("description"),
                                  "image_bytes": current["image_bytes"]})
        st.rerun()
    if st.button(tr("Analyze Another Plant"), icon=":material/refresh:", key="reset_analysis"):
        st.session_state.analysis_result = None
        st.session_state.analysis_nonce += 1
        st.rerun()

if current.get("record"):
    if current["saved"]:
        st.success(tr("Diagnosis saved to the local history database."))
    else:
        st.info(tr("Saved for this browser session only. It will be lost when you close the page. "
                   "Set PLANTVERSE_PERSIST_HISTORY=1 on a private/local install to keep history permanently."))
    rec = current["record"]
    d1, d2 = st.columns(2)
    d1.download_button(tr("Download report (.txt)"), store.export_report(rec, "txt"),
                       file_name=f"plantverse-{rec['id'][:8]}.txt", mime="text/plain", key="dl_txt")
    d2.download_button(tr("Download report (.json)"), store.export_report(rec, "json"),
                       file_name=f"plantverse-{rec['id'][:8]}.json", mime="application/json", key="dl_json")

# ------------------------------------------------------------------ optional field sighting (session only)
if disease["status"] in ("success", "uncertain"):
    with st.expander(tr("Log as field sighting on the AR map (optional)")):
        st.caption(tr("Sightings are kept for this browser session only and are shown on the AR Explorer map. "
                      "Coordinates are only used if you add them."))
        s_lat = st.number_input(tr("Latitude"), value=float(st.session_state.get("ar_lat") or 0.0), format="%.6f",
                                min_value=-90.0, max_value=90.0, key="sight_lat")
        s_lon = st.number_input(tr("Longitude"), value=float(st.session_state.get("ar_lon") or 0.0), format="%.6f",
                                min_value=-180.0, max_value=180.0, key="sight_lon")
        s_tag = st.text_input(tr("Location name"), key="sight_tag")
        s_notes = st.text_area(tr("Notes"), key="sight_notes")
        if st.button(tr("Log sighting"), icon=":material/pin_drop:", key="log_sighting"):
            rec = current.get("record") or store.build_record(plant, disease, current["image_bytes"],
                                                              details.get("scientific_name"))
            entry = store.to_session_entry({**rec, "id": store.uuid.uuid4().hex}, current["image_bytes"])
            entry.update(is_sighting=True, latitude=s_lat, longitude=s_lon, notes=s_notes,
                         location_name=s_tag or f"Location ({s_lat:.4f}, {s_lon:.4f})")
            st.session_state.disease_diagnoses = [entry] + st.session_state.disease_diagnoses[:49]
            st.success(tr("Sighting logged for this session."))
