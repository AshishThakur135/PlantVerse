from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ui.chrome import definition_row

PROFILE_RANKS = ("Class", "Order", "Family", "Tribe", "Genus")


def _rank_match(taxonomy: Dict[str, Any], rank: str) -> Optional[str]:
    for key, value in taxonomy.items():
        if str(key).lower() == rank.lower() and value:
            return str(value)
    return None


def split_taxonomy(taxonomy: Optional[Dict[str, Any]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    if not taxonomy or "error" in taxonomy:
        return [], []

    featured: List[Tuple[str, str]] = []
    used = set()
    for rank in PROFILE_RANKS:
        value = _rank_match(taxonomy, rank)
        if value:
            featured.append((rank, value))
            used.add(rank.lower())

    remaining = [
        (str(key), str(value))
        for key, value in taxonomy.items()
        if str(key).lower() not in used and value
    ]
    return featured, remaining


def render_taxonomy(
    taxonomy: Optional[Dict[str, Any]],
    translate=None,
    lang_code: str = "en",
    featured_only: bool = False,
    key_prefix: str = "taxon",
) -> None:
    def tx(value: str) -> str:
        if translate is None:
            return value
        return translate(value, lang_code)

    if not taxonomy:
        st.caption(tx("No taxonomy information found."))
        return

    if "error" in taxonomy:
        st.caption(tx(str(taxonomy["error"])))
        return

    featured, remaining = split_taxonomy(taxonomy)
    rows = featured if featured_only else featured + remaining
    if not rows:
        st.caption(tx("No taxonomy information found."))
        return

    with st.container(border=True):
        cols = st.columns(min(len(rows), 3))
        for idx, (rank, value) in enumerate(rows):
            col = cols[idx % len(cols)]
            with col:
                st.caption(f":material/account_tree: {tx(rank).upper()}")
                st.markdown(f"*{value}*")


def remaining_taxonomy(
    taxonomy: Optional[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    _, remaining = split_taxonomy(taxonomy)
    return remaining


def taxon_photo_url(plant_info: Optional[Dict[str, Any]]) -> Optional[str]:
    if not plant_info:
        return None
    photo = plant_info.get("default_photo") or {}
    return (
        photo.get("medium_url")
        or photo.get("url")
        or photo.get("square_url")
    )


def photo_attribution(plant_info: Optional[Dict[str, Any]]) -> str:
    if not plant_info:
        return ""
    photo = plant_info.get("default_photo") or {}
    return str(photo.get("attribution") or "")


def extra_taxon_fields(plant_info: Optional[Dict[str, Any]]) -> List[Tuple[str, str]]:
    if not plant_info:
        return []

    rows: List[Tuple[str, str]] = []
    wikipedia_url = plant_info.get("wikipedia_url")
    if wikipedia_url:
        rows.append(("Wikipedia", f"[Open source page]({wikipedia_url})"))

    matched = plant_info.get("matched_term")
    if matched:
        rows.append(("Matched term", str(matched)))

    iconic = plant_info.get("iconic_taxon_name")
    if iconic:
        rows.append(("Iconic taxon", str(iconic)))

    return rows


def upsert_library_entry(entry: Dict[str, Any]) -> None:
    entries: List[Dict[str, Any]] = st.session_state.library_entries
    key = (entry.get("scientific_name") or entry.get("common_name") or "").lower()
    entries = [
        item
        for item in entries
        if (item.get("scientific_name") or item.get("common_name") or "").lower() != key
    ]
    entries.insert(0, entry)
    st.session_state.library_entries = entries[:24]
