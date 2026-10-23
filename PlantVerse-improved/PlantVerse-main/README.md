---
title: PlantVerse — Botanical Intelligence & Pathology Framework
emoji: 🌿
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: "1.46.1"
app_file: app.py
pinned: false
---

# 🌿 PlantVerse — Botanical Intelligence & Plant Pathology

**Discover, diagnose, and explore plants — visually, pathologically, geographically, and culturally.**  
PlantVerse is a comprehensive, multilingual platform combining neural plant identification, foliar disease diagnosis, dynamic biological taxonomy, ethnobotanical pharmacology, and geospatial field habitat mapping.

---


## ✨ Features

- **Plant Analysis** (one page): upload one photo, press **Analyze Plant**, get per-plant candidate identities from localized COCO potted-plant objects (DETR + OpenPlants ViT), plus a *Disease Detection* card (PlantVillage, 38 classes). Both models use the same validated image; if one fails the other result is still shown.
- Honest results: every analysis has a status (`success`, `uncertain`, `unsupported`, `unavailable`, `error`). If a model cannot run, **no prediction is made up**. Plant identification always shows the model's top result and alternatives, including low-confidence guesses. Unknown disease labels are reported as unsupported.
- Disease-specific treatment is shown only for an exact disease mapping with a verified source, a reliable disease result, and a confident compatible plant identification. Conflicting or uncertain results retain general safety and prevention guidance but withhold crop-specific treatment.
- The treatment service currently has source-backed entries for tomato late blight and potato late blight (Cornell University / New York State IPM). The other 36 model classes explicitly show that a verified disease-specific treatment entry is not yet available; legacy unsourced pesticide suggestions are not rendered.
- Scientific names only appear when Wikidata supplies a species rank; Wikipedia hits must match the predicted name.
- Disease details: symptoms, possible causes, next steps, prevention, expert-confirmation note.
- Save Diagnosis, history in the Plant Library, `.txt`/`.json` report download, optional SQLite persistence.
- Explore, Plant Library, AR Explorer, About, 4-language UI (translations via a public Google endpoint; English text is kept if it fails).
- Optional REST API (`ml_service`).

## 📦 Requirements

Python **3.10 – 3.12**. Install: `pip install -r requirements.txt` (add `pip install -r requirements-dev.txt` for tests).

## 🚀 Setup & Run

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py               # web app
python -m ml_service.app           # optional API on http://127.0.0.1:8000
```

Windows PowerShell (from this workspace):

```powershell
Set-Location 'C:\Users\Asus\Downloads\PlantVerse-improved\PlantVerse-main'
& 'C:\Users\Asus\Downloads\PlantVerse-improved\.venv\Scripts\python.exe' -m streamlit run app.py
```

**Model setup.** Models are downloaded from Hugging Face on first analysis (needs internet; first run is slow and uses disk). `domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224` identifies species across 14,829 scientific-name labels; `Khawajaa/plant-disease-detector` detects leaf health and disease with EfficientNet-B4. The exact uploaded 614 × 595 test image produced `Epipremnum aureum` at 93.8723% with OpenPlants versus 0.1161% with FloraSense. This is a result for that image, not a general accuracy guarantee. The disease model is a closed-set classifier for 38 PlantVillage leaf classes, and can return plausible-looking crop labels for plants it was not trained on. Its confidence is a softmax score, not a calibrated probability. After download, models are cached locally and no network is needed for inference (Wikipedia/Wikidata/translation still need it, but are optional). A local Keras model is used only if `PLANT_DISEASE_KERAS_MODEL_PATH` is set.
Optional `facebook/detr-resnet-50` object detection can localize COCO `potted plant` objects; OpenPlants then classifies each detected crop. It is disabled by default because it adds a large model and memory demand alongside the species and disease models. Set `PLANT_OBJECT_MODEL_NAME=facebook/detr-resnet-50` to enable it on a machine with sufficient memory. Its object-detection score and the per-crop species softmax score are displayed separately and neither is a calibrated probability. This detector does not cover every unpotted plant, may miss occluded/small plants, and cannot guarantee that every plant was localized. When disabled or when no potted plant is detected, the app clearly marks its result as a whole-photo classification. The disease classifier still analyzes the complete photo, so crop-specific treatment is withheld when plant identity is not reliable or compatible.
Plant identification and disease analysis render before optional Wikipedia/Wikidata reference enrichment. Select **Load plant details** to request taxonomy, description, and traditional-use information; external lookups are not needed for diagnosis. Subsequent inference is faster after all required model weights have been downloaded and the cached models initialized.

## 🔧 Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PLANT_CONFIDENCE_THRESHOLD` | `0.50` | Below this, plant ID is "uncertain" |
| `DISEASE_CONFIDENCE_THRESHOLD` | `0.50` | Screening threshold: lower top softmax scores are marked uncertain; this is not a calibrated diagnostic probability |
| `PLANT_SPECIES_MODEL_NAME` | `domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224` | Hugging Face plant species classifier |
| `PLANT_OBJECT_MODEL_NAME` | *(unset)* | Optional Apache-2.0 COCO object detector for localizing potted plants; adds substantial memory use |
| `PLANT_OBJECT_CONFIDENCE_THRESHOLD` | `0.50` | COCO potted-plant box score cutoff; not a species confidence threshold |
| `PLANT_DISEASE_MODEL_NAME` | `Khawajaa/plant-disease-detector` | Hugging Face repository containing `best_model.pth` |
| `PLANT_DISEASE_KERAS_MODEL_PATH` | *(unset)* | Opt-in local Keras model; `PLANT_DISEASE_KERAS_RESCALE` sets pixel scale (default `1.0`) |
| `PLANT_DISEASE_API_URL` / `PLANT_DISEASE_API_TIMEOUT` | *(unset)* / `12` | Use the ML API; falls back to the local model on failure |
| `MAX_IMAGE_SIZE_BYTES`, `MIN_IMAGE_SIDE_PX`, `MAX_IMAGE_SIDE_PX` | 10 MB, 64, 8000 | Upload limits (app and API) |
| `PLANTVERSE_PERSIST_HISTORY` | `0` | `1` = save history in SQLite (`data/plantverse.db`, path via `PLANTVERSE_DB_PATH`) |
| `CORS_ALLOWED_ORIGINS` | *(empty = no CORS)* | Comma-separated approved origins for the API |
| `PREDICT_TIMEOUT_SECONDS`, `PORT`, `HOST` | `30`, `8000`, `127.0.0.1` | API runtime |

No secrets are required or stored in the code.

## 💾 History & privacy

Without `PLANTVERSE_PERSIST_HISTORY=1`, saved diagnoses live **only in the current browser session** and the UI says so. When enabled, a record stores metadata only (time, labels, confidences, status, recommendations, an image SHA-256) and never the photo. The app has **no login**, so enable persistence only on a private/local install; do not enable it on a public deployment where visitors would share one database.

## 📡 REST API

| Endpoint | Behaviour |
|---|---|
| `GET /health` | Liveness: `status: "ok"` plus `modelReady` / `modelBackend` (always 200) |
| `GET /ready` | 200 when the model is loaded, otherwise 503 |
| `POST /api/disease/predict` | multipart `image` (or `file`). 200 result; 400 missing/empty; 413 too large; 422 invalid image; 503 model unavailable; 504 timeout; 500 generic error (no internals leaked) |
| `POST /api/disease/diagnosis` | Validates a payload (422 on bad fields). **Does not store anything** (`saved: false`) |

Predict responses keep `confidence` (percent) and add `confidenceScore` (0–1), `status`, `rawLabel`, and a structured `treatment` result. The disease-only REST API has no independent species identification, so it withholds crop-specific treatment and returns general safety guidance. Breaking changes vs 1.0: `/health` returns `status: "ok"` (was `"healthy"`; `modelLoaded` kept as an alias), no fabricated "Healthy Plant" when the model is missing (503 instead), CORS is closed unless `CORS_ALLOWED_ORIGINS` is set, and the API binds to `127.0.0.1` unless `HOST` is set.

## 🧪 Testing

```bash
pip install -r requirements-dev.txt
python -m pytest
```

Tests mock models and external services; they need no internet or model download.

## 🩹 Troubleshooting

- *"model could not be loaded"*: check internet for the first download, free disk space, and `pip install -r requirements.txt`.
- *Plant identity is uncertain or disease advice is withheld*: compare the displayed top five species guesses, upload a sharp close-up of the plant, and confirm its species before using crop-specific disease advice. Missing source-backed entries are called out explicitly rather than filled with unsupported chemical advice.
- *Everything says uncertain*: use a sharper, closer photo of one leaf, or tune the thresholds.
- *Translations missing*: the translation service is unreachable; English is shown.
- *History not kept*: set `PLANTVERSE_PERSIST_HISTORY=1` (local use only).

## ⚠️ Known limitations

- Predictions are AI estimates, not agricultural or medical diagnoses. The OpenPlants model predicts only among its 14,829 known species and does not use location context; similar species can be difficult to distinguish. Its card does not report a held-out accuracy figure. The disease model only knows the 38 PlantVillage classes (lab-style leaf photos); field photos often give lower accuracy. The disease model card reports 98.68% top-1 accuracy on its held-out PlantVillage test split; that is not a guarantee for an individual field photo.
- The label order/preprocessing of a custom Keras model cannot be verified automatically.
- Field sightings and the Plant Library are session-only.
- Both model label sets and preprocessing are read from their published model configurations/cards; unfamiliar disease labels are reported as unsupported.
- `.huggingface.yml` pins Streamlit 1.46.1 / Python 3.10; the UI uses newer Streamlit features (`st.space`, horizontal containers), so verify/raise this pin before deploying to Spaces.

## ☁️ Deployment

Run `streamlit run app.py` behind HTTPS. Keep persistence off for public hosts, set `CORS_ALLOWED_ORIGINS` for the API, and run the API behind a reverse proxy that enforces request-size limits. Pre-download the models at build time on hosts with no outbound network.

## 📂 Project structure

```
app.py                      # Navigation (Explore, Plant Analysis, Plant Library, AR Explorer, About)
app_pages/analysis.py       # Combined Plant Analysis page
image_utils.py              # Shared image validation
plant_identification.py     # Species model + Wikipedia/Wikidata/translation helpers
plant_disease_detection.py  # Disease engine (structured statuses, no fabricated fallback)
disease_recommendations.py  # Knowledge base + strict label normalisation
diagnosis_store.py          # Session/SQLite history and report export
ml_service/                 # Optional Starlette API
tests/                      # pytest suite
```

## 🦠 Supported Plant Pathology Classes (38 Classes)

Adapted from the PlantVillage dataset referenced by [animesh1012/machineLearning/Plant_Disease_Prediction](https://github.com/animesh1012/machineLearning/tree/main/Plant_Disease_Prediction):

1. **Apple:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
2. **Blueberry:** Healthy
3. **Cherry:** Powdery Mildew, Healthy
4. **Corn (Maize):** Cercospora Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
5. **Grape:** Black Rot, Esca (Black Measles), Isariopsis Leaf Blight, Healthy
6. **Orange / Citrus:** Citrus Greening (Huanglongbing)
7. **Peach:** Bacterial Spot, Healthy
8. **Bell Pepper:** Bacterial Spot, Healthy
9. **Potato:** Early Blight, Late Blight, Healthy
10. **Raspberry:** Healthy
11. **Soybean:** Healthy
12. **Squash:** Powdery Mildew
13. **Strawberry:** Leaf Scorch, Healthy
14. **Tomato:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Two-Spotted Spider Mites, Target Spot, Tomato Yellow Leaf Curl Virus (TYLCV), Tomato Mosaic Virus (ToMV), Healthy

---


## 📜 Attribution & References

- **Plant Disease Prediction Base**: The classification taxonomy, model parameters, and training methodology are derived from [animesh1012/machineLearning/Plant_Disease_Prediction](https://github.com/animesh1012/machineLearning/tree/main/Plant_Disease_Prediction).
- **Dataset Foundation**: [PlantVillage](https://github.com/spMohanty/PlantVillage-Dataset) (Hughes & Salathé), comprising ~87,000 foliar images across 38 crop condition classes.
- **OpenPlants Specimen Classifier**: domai-tb/OpenPlants-Identification-ViT-Base-Patch16-224 (Hugging Face).
- **Geospatial & Biodiversity Records**: [iNaturalist Open API](https://api.inaturalist.org/v1).
- **Taxonomic Ontology**: [Wikidata Entity Claims API](https://www.wikidata.org).
