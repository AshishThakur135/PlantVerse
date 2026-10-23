"""Crop compatibility checks for combining plant identity and disease predictions."""

import re
from typing import Any


_CROP_SPECIES = {
    "Apple": {"malus domestica", "malus pumila"},
    "Blueberry": {"vaccinium corymbosum", "vaccinium angustifolium", "vaccinium virgatum"},
    "Cherry": {"prunus avium", "prunus cerasus"},
    "Corn (Maize)": {"zea mays"},
    "Grape": {"vitis vinifera", "vitis labrusca", "vitis rotundifolia"},
    "Orange": {"citrus sinensis", "citrus reticulata", "citrus paradisi", "citrus aurantium"},
    "Peach": {"prunus persica"},
    "Bell Pepper": {"capsicum annuum"},
    "Potato": {"solanum tuberosum"},
    "Raspberry": {"rubus idaeus"},
    "Soybean": {"glycine max"},
    "Squash": {"cucurbita pepo", "cucurbita maxima", "cucurbita moschata"},
    "Strawberry": {"fragaria vesca", "fragaria ananassa"},
    "Tomato": {"solanum lycopersicum", "lycopersicon esculentum"},
}


def species_matches_crop(species: Any, crop: Any) -> bool:
    """Match a scientific species label to a PlantVillage crop, never by genus alone."""
    if not isinstance(species, str) or not isinstance(crop, str):
        return False
    words = re.findall(r"[a-z]+", species.lower())
    if len(words) < 2:
        return False
    if words[1] == "x":
        if len(words) < 3:
            return False
        words.pop(1)
    binomial = " ".join(words[:2])
    return binomial in _CROP_SPECIES.get(crop, set())
