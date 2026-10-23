# ============================================================
# PlantVerse - Plant Identification
# ============================================================

import streamlit as st
import requests
from PIL import Image
import logging
import os
import math
from typing import Any, Dict, Optional
from functools import lru_cache
from bs4 import BeautifulSoup
import re

log = logging.getLogger(__name__)
OPENPLANTS_MODEL_NAME = "domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224"
DEFAULT_PLANT_MODEL_NAME = OPENPLANTS_MODEL_NAME
PLANT_OBJECT_MODEL_NAME = os.getenv("PLANT_OBJECT_MODEL_NAME", "").strip()
PLANT_CONFIDENCE_THRESHOLD = float(os.getenv("PLANT_CONFIDENCE_THRESHOLD", "0.50"))
PLANT_OBJECT_CONFIDENCE_THRESHOLD = float(os.getenv("PLANT_OBJECT_CONFIDENCE_THRESHOLD", "0.50"))
CACHE_TTL = 6 * 3600  # failures from external services expire instead of sticking forever


# ============================================================
# 1. Language Setup
# ============================================================

LANGUAGES = {
    "English": "en",
    "हिन्दी (Hindi)": "hi",
    "తెలుగు (Telugu)": "te",
    "தமிழ் (Tamil)": "ta",
}


# ============================================================
# 2. Translation Function
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def translate_text(text, target_lang):
    """
    Translate English text into the selected language.

    If Google Translate is unavailable, the original English
    text is returned instead of crashing the application.
    """

    # No translation required
    if not text or target_lang == "en":
        return text

    try:
        url = "https://translate.googleapis.com/translate_a/single"

        params = {
            "client": "gtx",
            "sl": "en",
            "tl": target_lang,
            "dt": "t",
            "q": text
        }

        response = requests.get(
            url,
            params=params,
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        response.raise_for_status()

        data = response.json()

        if data and data[0]:
            return "".join(
                part[0]
                for part in data[0]
                if part and len(part) > 0
            )

        return text

    except requests.exceptions.RequestException as e:
        print(f"Translation request failed: {e}")
        return text

    except (ValueError, IndexError, TypeError) as e:
        print(f"Translation returned invalid data: {e}")
        return text

    except Exception as e:
        print(f"Translation error: {e}")
        return text


# ============================================================
# 3. Plant Classifier
# ============================================================

@st.cache_resource(show_spinner=False)
def load_classifier():
    """Load the configured species classifier once (failures are not cached)."""
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from transformers import pipeline

    model_name = os.getenv("PLANT_SPECIES_MODEL_NAME", DEFAULT_PLANT_MODEL_NAME)
    return pipeline("image-classification", model=model_name)


@st.cache_resource(show_spinner=False)
def load_plant_detector():
    """Load the Apache-2.0 COCO object detector once for potted-plant localization."""
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
    from transformers import pipeline

    return pipeline("object-detection", model=PLANT_OBJECT_MODEL_NAME)


def identify_plant(image: Image.Image, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Predict species. Returns a dict with status success|uncertain|unavailable|error.

    Never raises and never invents a label: on failure no label/score is returned.
    """
    threshold = PLANT_CONFIDENCE_THRESHOLD if threshold is None else threshold
    try:
        classifier = load_classifier()
    except Exception as exc:  # noqa: BLE001
        log.warning("Species model load failed: %s", exc)
        model_name = os.getenv("PLANT_SPECIES_MODEL_NAME", DEFAULT_PLANT_MODEL_NAME)
        return {"status": "unavailable", "success": False,
                "message": (f"The plant identification model '{model_name}' could not be loaded "
                            f"({type(exc).__name__}). Check your internet connection for the first-time "
                            "download and verify that the model dependencies are installed.")}
    try:
        preds = classifier(image.convert("RGB"), top_k=5)
        cands = []
        for p in preds or []:
            if not isinstance(p, dict):
                continue
            label = p.get("label")
            try:
                score = float(p.get("score"))
            except (TypeError, ValueError):
                continue
            if isinstance(label, str) and label.strip() and math.isfinite(score) and 0.0 <= score <= 1.0:
                cands.append({
                    "label": label.replace("_", " ").strip(),
                    "score": score,
                    "confidence_pct": round(score * 100, 4),
                })
        cands.sort(key=lambda c: c["score"], reverse=True)
    except Exception as exc:  # noqa: BLE001
        log.warning("Species inference failed: %s", exc)
        return {"status": "error", "success": False,
                "message": f"Plant identification failed ({type(exc).__name__}). Please try again."}
    if not cands:
        return {"status": "error", "success": False, "message": "The model returned no valid prediction."}
    top = cands[0]
    uncertain = top["score"] < threshold
    return {
        "status": "uncertain" if uncertain else "success",
        "success": not uncertain,
        "label": top["label"],
        "score": top["score"],
        "candidates": cands,
        "threshold": threshold,
        "message": (f"Low model score ({top['score'] * 100:.4f}% < {threshold * 100:.2f}%). "
                    "The top five candidates are shown; treat them as possibilities and try a clearer, "
                    "closer photo.") if uncertain else "",
    }


def _intersection_over_union(box_a: Dict[str, float], box_b: Dict[str, float]) -> float:
    left = max(box_a["xmin"], box_b["xmin"])
    top = max(box_a["ymin"], box_b["ymin"])
    right = min(box_a["xmax"], box_b["xmax"])
    bottom = min(box_a["ymax"], box_b["ymax"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, box_a["xmax"] - box_a["xmin"]) * max(0.0, box_a["ymax"] - box_a["ymin"])
    area_b = max(0.0, box_b["xmax"] - box_b["xmin"]) * max(0.0, box_b["ymax"] - box_b["ymin"])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _unique_plant_boxes(detections: Any, image_size: tuple[int, int]) -> list[Dict[str, Any]]:
    """Keep distinct COCO potted-plant boxes, suppressing duplicate overlapping boxes."""
    width, height = image_size
    valid = []
    for detection in detections or []:
        if not isinstance(detection, dict) or str(detection.get("label", "")).casefold() != "potted plant":
            continue
        try:
            score = float(detection["score"])
            raw_box = detection["box"]
            box = {
                "xmin": max(0.0, min(float(raw_box["xmin"]), float(width))),
                "ymin": max(0.0, min(float(raw_box["ymin"]), float(height))),
                "xmax": max(0.0, min(float(raw_box["xmax"]), float(width))),
                "ymax": max(0.0, min(float(raw_box["ymax"]), float(height))),
            }
        except (KeyError, TypeError, ValueError):
            continue
        if not math.isfinite(score) or not PLANT_OBJECT_CONFIDENCE_THRESHOLD <= score <= 1.0:
            continue
        if box["xmax"] <= box["xmin"] or box["ymax"] <= box["ymin"]:
            continue
        valid.append({"detection_score": score, "box": box})

    accepted: list[Dict[str, Any]] = []
    for candidate in sorted(valid, key=lambda item: item["detection_score"], reverse=True):
        if all(_intersection_over_union(candidate["box"], item["box"]) < 0.65 for item in accepted):
            accepted.append(candidate)
    return accepted


def identify_plants(image: Image.Image, threshold: Optional[float] = None) -> Dict[str, Any]:
    """Detect separate potted-plant instances and identify each cropped plant.

    DETR's COCO labels include potted plants, not all botanical objects. When it
    is explicitly configured, retain the whole-image classifier result when it
    finds none or cannot load. By default skip this large optional model and
    avoid competing with the species and disease classifiers for memory.
    """
    rgb_image = image.convert("RGB")
    if not PLANT_OBJECT_MODEL_NAME:
        fallback = identify_plant(rgb_image, threshold)
        fallback.update({
            "instances": [],
            "detection_status": "disabled",
            "detection_message": (
                "Optional plant localization is disabled to keep model memory use lower. "
                "Species identification classifies the full photo; use a clear photo of one plant."
            ),
        })
        return fallback

    try:
        detector = load_plant_detector()
        detections = detector(rgb_image, threshold=PLANT_OBJECT_CONFIDENCE_THRESHOLD)
    except Exception as exc:  # noqa: BLE001
        log.warning("Plant object detection unavailable: %s", exc)
        fallback = identify_plant(rgb_image, threshold)
        fallback.update({
            "instances": [],
            "detection_status": "unavailable",
            "detection_message": (
                f"Separate plant localization is unavailable ({type(exc).__name__}); "
                "the result below classifies the entire photo and may not identify every plant."
            ),
        })
        return fallback

    boxes = _unique_plant_boxes(detections, rgb_image.size)
    if not boxes:
        fallback = identify_plant(rgb_image, threshold)
        fallback.update({
            "instances": [],
            "detection_status": "no_instances",
            "detection_message": (
                "No separate potted-plant objects were localized. The result below is a whole-photo "
                "classification and may not identify every plant in the image."
            ),
        })
        return fallback

    width, height = rgb_image.size
    instances = []
    for candidate in boxes:
        box = candidate["box"]
        pad_x = (box["xmax"] - box["xmin"]) * 0.08
        pad_y = (box["ymax"] - box["ymin"]) * 0.08
        crop_box = (
            max(0, int(box["xmin"] - pad_x)),
            max(0, int(box["ymin"] - pad_y)),
            min(width, int(box["xmax"] + pad_x)),
            min(height, int(box["ymax"] + pad_y)),
        )
        crop = rgb_image.crop(crop_box)
        identification = identify_plant(crop, threshold)
        instances.append({
            **identification,
            "detection_score": candidate["detection_score"],
            "box": crop_box,
            "image": crop,
        })

    primary = next((item for item in instances if item.get("label")), instances[0])
    result = dict(primary)
    result.update({
        "instances": instances,
        "detection_status": "detected",
        "detection_message": (
            f"Localized {len(instances)} separate potted-plant object(s). "
            "Each species score is the classifier's softmax score for that crop, not a calibrated probability."
        ),
    })
    return result


def predict_species(image: Image.Image):
    """Backward-compatible (label, score); raises RuntimeError if identification fails."""
    res = identify_plant(image)
    if "label" not in res:
        raise RuntimeError(res["message"])
    return res["label"], res["score"]


def scientific_name_from_taxonomy(taxonomy: Optional[Dict[str, Any]]) -> Optional[str]:
    """Scientific name only when Wikidata supplied a Species rank - never guessed."""
    if not isinstance(taxonomy, dict) or "error" in taxonomy:
        return None
    for key, value in taxonomy.items():
        if str(key).lower() == "species" and value:
            return str(value)
    return None


# ============================================================
# 4. Wikipedia Search
# ============================================================

def _tokens(text: str) -> set:
    return set(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())


def canonical_species_name(label: str) -> str:
    """Drop botanical author names from binomials before external lookups."""
    words = re.findall(r"[A-Za-z]+", label.replace("_", " "))
    if len(words) < 2 or not words[0][0].isupper():
        return label
    epithet_index = 2 if len(words) > 2 and words[1].lower() == "x" else 1
    if epithet_index >= len(words) or not words[epithet_index][0].islower():
        return label
    return f"{words[0]} {words[epithet_index]}"


def title_matches_label(label: str, title: str) -> bool:
    """True only if the page title and model label share all words of the shorter one."""
    if "disambiguation" in title.lower():
        return False
    a, b = _tokens(label), _tokens(title.split("(")[0])
    return bool(a and b) and (a <= b or b <= a)


@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def resolve_wikipedia_title(name: str) -> Optional[str]:
    """Return a Wikipedia title whose name matches ``name``, else None.

    The first search hit is NOT trusted; it must match the predicted label.
    """
    try:
        response = requests.get(
            "https://en.wikipedia.org/w/api.php",
            params={"action": "query", "list": "search", "srsearch": canonical_species_name(name), "srlimit": 5,
                    "format": "json", "utf8": 1},
            timeout=10,
            headers={"User-Agent": "PlantVerse/1.0 (educational project)"},
        )
        response.raise_for_status()
        results = response.json().get("query", {}).get("search", [])
    except (requests.exceptions.RequestException, ValueError, AttributeError) as exc:
        log.warning("Wikipedia lookup failed: %s", exc)
        return None
    for item in results:
        title = item.get("title") if isinstance(item, dict) else None
        if title and title_matches_label(canonical_species_name(name), title):
            return title
    return None


def get_wikipedia_title(name: str):
    """Backward compatible: verified title, or the original name when none matches."""
    return resolve_wikipedia_title(name) or name


# ============================================================
# 5. Wikidata Taxonomy
# ============================================================

@st.cache_data(ttl=CACHE_TTL, show_spinner=False)
def get_taxonomy_from_wikidata(label: str):
    """
    Get plant taxonomy from Wikidata.

    If Wikidata fails, return an error dictionary instead
    of crashing the application.
    """

    search_url = "https://www.wikidata.org/w/api.php"

    search_params = {
        "action": "wbsearchentities",
        "search": label,
        "language": "en",
        "format": "json"
    }

    try:

        # ----------------------------------------------------
        # Search Wikidata
        # ----------------------------------------------------

        response = requests.get(
            search_url,
            params=search_params,
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        response.raise_for_status()

        if "json" not in response.headers.get(
            "Content-Type",
            ""
        ).lower():
            return {
                "error": "Wikidata returned a non-JSON response."
            }

        search = response.json()

        results = search.get("search", [])

        if not results:
            return {
                "error": f"No Wikidata entity found for '{label}'."
            }

        eid = results[0].get("id")

        if not eid:
            return {
                "error": f"No valid Wikidata ID found for '{label}'."
            }


        # ----------------------------------------------------
        # Fetch Wikidata Entity
        # ----------------------------------------------------

        @lru_cache(None)
        def fetch(e):

            try:

                entity_url = (
                    f"https://www.wikidata.org/"
                    f"wiki/Special:EntityData/{e}.json"
                )

                response = requests.get(
                    entity_url,
                    timeout=10,
                    headers={
                        "User-Agent": "PlantVerse/1.0"
                    }
                )

                response.raise_for_status()

                if "json" not in response.headers.get(
                    "Content-Type",
                    ""
                ).lower():
                    print(
                        f"Wikidata entity {e} "
                        f"returned non-JSON response."
                    )
                    return {}

                data = response.json()

                return data.get(
                    "entities",
                    {}
                ).get(
                    e,
                    {}
                )

            except requests.exceptions.RequestException as err:
                print(
                    f"Failed to fetch Wikidata entity "
                    f"{e}: {err}"
                )
                return {}

            except (ValueError, KeyError, TypeError) as err:
                print(
                    f"Invalid Wikidata data for "
                    f"{e}: {err}"
                )
                return {}

            except Exception as err:
                print(
                    f"Unexpected Wikidata error "
                    f"for {e}: {err}"
                )
                return {}


        # ----------------------------------------------------
        # Find Taxonomic Entity
        # ----------------------------------------------------

        def find_taxon(e, depth=0):

            if depth > 5:
                return None

            ent = fetch(e)

            if not ent:
                return None

            claims = ent.get("claims", {})

            # P225 = taxon scientific name
            # P171 = parent taxon

            if (
                "P225" in claims
                and "P171" in claims
            ):
                return e

            # Try instance of / subclass of
            for prop in ("P31", "P279"):

                for claim in claims.get(prop, []):

                    try:

                        mainsnak = claim.get(
                            "mainsnak",
                            {}
                        )

                        datavalue = mainsnak.get(
                            "datavalue",
                            {}
                        )

                        value = datavalue.get(
                            "value",
                            {}
                        )

                        if isinstance(value, dict):
                            nid = value.get("id")

                            if nid:
                                result = find_taxon(
                                    nid,
                                    depth + 1
                                )

                                if result:
                                    return result

                    except Exception:
                        continue

            return None


        taxon_e = find_taxon(eid)

        if not taxon_e:

            return {
                "error": (
                    f"No taxonomic information "
                    f"found for '{label}'."
                )
            }


        # ----------------------------------------------------
        # Collect Taxonomy
        # ----------------------------------------------------

        taxonomy = {}

        def collect(e, lvl=0):

            if lvl > 20:
                return

            ent = fetch(e)

            if not ent:
                return

            claims = ent.get(
                "claims",
                {}
            )

            # ------------------------------------------------
            # Scientific Name - P225
            # ------------------------------------------------

            sci = ""

            try:

                p225 = claims.get(
                    "P225",
                    []
                )

                if p225:

                    sci = (
                        p225[0]
                        .get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", "")
                    )

            except Exception:
                sci = ""


            # ------------------------------------------------
            # Taxonomic Rank - P105
            # ------------------------------------------------

            rank_id = ""

            try:

                p105 = claims.get(
                    "P105",
                    []
                )

                if p105:

                    rank_id = (
                        p105[0]
                        .get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                        .get("id", "")
                    )

            except Exception:
                rank_id = ""


            # ------------------------------------------------
            # Get Rank Name
            # ------------------------------------------------

            rank = ""

            if rank_id:

                rank_entity = fetch(rank_id)

                rank = (
                    rank_entity
                    .get("labels", {})
                    .get("en", {})
                    .get("value", "")
                )


            if sci:

                taxonomy[
                    rank.capitalize()
                    if rank
                    else f"Rank{lvl}"
                ] = sci


            # ------------------------------------------------
            # Parent Taxon - P171
            # ------------------------------------------------

            parent = None

            try:

                p171 = claims.get(
                    "P171",
                    []
                )

                if p171:

                    parent = (
                        p171[0]
                        .get("mainsnak", {})
                        .get("datavalue", {})
                        .get("value", {})
                        .get("id")
                    )

            except Exception:
                parent = None


            if parent:
                collect(
                    parent,
                    lvl + 1
                )


        collect(taxon_e)

        return taxonomy


    except requests.exceptions.RequestException as e:

        print(
            f"Wikidata request failed: {e}"
        )

        return {
            "error": "Unable to connect to Wikidata."
        }


    except (ValueError, KeyError, TypeError) as e:

        print(
            f"Wikidata returned invalid data: {e}"
        )

        return {
            "error": "Wikidata returned invalid data."
        }


    except Exception as e:

        print(
            f"Wikidata error: {e}"
        )

        return {
            "error": "Unable to retrieve taxonomy information."
        }


# ============================================================
# 6. Medicinal Uses from Wikipedia
# ============================================================

def get_medicinal_uses_from_wikipedia(
    title: str,
    target_lang="en"
):
    """
    Extract medicinal/traditional medicine information
    from the Wikipedia page.

    If Wikipedia is unavailable, return None.
    """

    try:

        url = "https://en.wikipedia.org/w/api.php"

        # ----------------------------------------------------
        # Get Wikipedia sections
        # ----------------------------------------------------

        sections_response = requests.get(
            url,
            params={
                "action": "parse",
                "page": title,
                "prop": "sections",
                "format": "json"
            },
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        sections_response.raise_for_status()

        if "json" not in sections_response.headers.get(
            "Content-Type",
            ""
        ).lower():
            return None

        sections_data = sections_response.json()

        sections = (
            sections_data
            .get("parse", {})
            .get("sections", [])
        )

        # ----------------------------------------------------
        # Find medicinal section
        # ----------------------------------------------------

        section_index = next(
            (
                sec.get("index")
                for sec in sections
                if (
                    "medicinal"
                    in sec.get(
                        "line",
                        ""
                    ).lower()
                    or
                    "traditional medicine"
                    in sec.get(
                        "line",
                        ""
                    ).lower()
                )
            ),
            None
        )

        if not section_index:
            return None


        # ----------------------------------------------------
        # Get section text
        # ----------------------------------------------------

        text_response = requests.get(
            url,
            params={
                "action": "parse",
                "page": title,
                "format": "json",
                "prop": "text",
                "section": section_index
            },
            timeout=10,
            headers={
                "User-Agent": "PlantVerse/1.0"
            }
        )

        text_response.raise_for_status()

        if "json" not in text_response.headers.get(
            "Content-Type",
            ""
        ).lower():
            return None

        text_data = text_response.json()

        html = (
            text_data
            .get("parse", {})
            .get("text", {})
            .get("*", "")
        )

        if not html:
            return None


        # ----------------------------------------------------
        # Parse HTML
        # ----------------------------------------------------

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        bullet_points = []


        # ----------------------------------------------------
        # Extract useful sentences
        # ----------------------------------------------------

        for paragraph in paragraphs:

            text = paragraph.get_text().strip()

            # Remove citation numbers
            text = re.sub(
                r"\[\d+\]",
                "",
                text
            )

            if len(text.split()) < 5:
                continue


            sentences = re.split(
                r"\.\s+",
                text
            )

            for sentence in sentences:

                sentence = (
                    sentence
                    .strip()
                    .strip(".")
                )

                if len(sentence.split()) < 5:
                    continue

                if sentence.lower().startswith(
                    "there is insufficient"
                ):
                    continue


                # Translate if necessary
                if target_lang != "en":

                    sentence = translate_text(
                        sentence,
                        target_lang
                    )


                bullet_points.append(
                    f"{sentence}."
                )


        if bullet_points:

            return bullet_points[:5]

        return None


    except requests.exceptions.RequestException as e:

        print(
            f"Wikipedia medicinal-use request failed: {e}"
        )

        return None


    except (ValueError, KeyError, TypeError) as e:

        print(
            f"Wikipedia medicinal-use data error: {e}"
        )

        return None


    except Exception as e:

        print(
            f"Medicinal-use extraction error: {e}"
        )

        return None


# Identification UI lives in app_pages/identify.py and reuses the functions above.
