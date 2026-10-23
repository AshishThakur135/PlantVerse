# PlantVerse — Plant Disease Detection ML Service

Standalone, isolated Python machine learning microservice for plant leaf disease prediction.

## Overview
Based on the convolutional neural network architecture and 38-class PlantVillage dataset taxonomy from [animesh1012/machineLearning/Plant_Disease_Prediction](https://github.com/animesh1012/machineLearning/tree/main/Plant_Disease_Prediction).

Exposes standard REST endpoints for image classification, health status evaluation, and diagnosis logging.

## Directory Structure
```
ml_service/
├── app.py                     # ASGI Web Service (Health, Predict, Diagnosis)
├── model/                     # Directory for custom .keras or .h5 weights
├── services/
│   └── disease_predictor.py   # Model loading and inference engine
├── utils/
│   └── preprocessing.py       # Image validation and tensor resizing
├── requirements.txt           # Python dependencies
└── README.md                  # Service documentation
```

## Running the Service

### 1. Install Dependencies
```bash
pip install -r ml_service/requirements.txt
```

### 2. Start the Server
```bash
uvicorn ml_service.app:app --host 0.0.0.0 --port 8000
```
Or directly:
```bash
python ml_service/app.py
```

## API Endpoints

### `GET /health`
Returns service status, model loading status, and active backend.

**Response:**
```json
{
  "status": "ok",
  "service": "PlantVerse Plant Disease Prediction ML API",
  "version": "1.0.0",
  "uptimeSeconds": 14.2,
  "modelBackend": "transformers",
  "modelLoaded": true,
  "threshold": 0.5
}
```

### `POST /api/disease/predict`
Accepts `multipart/form-data` with an `image` field.

**Example cURL Request:**
```bash
curl -X POST http://localhost:8000/api/disease/predict \
  -F "image=@static/sample_leaves/tomato_early_blight.jpg"
```

**Response:** The result includes the measured confidence, top candidates, and a structured `treatment` object with a status such as `available`, `identity_unconfirmed`, `unverified`, `uncertain`, or `healthy`. Because this disease-only API does not identify the plant independently, crop-specific treatment is withheld. Recommendations are only populated from the source-backed treatment database; currently it has entries for tomato and potato late blight. Other labels receive a clear safe fallback instead of legacy unsourced pesticide advice.

### `POST /api/disease/diagnosis`
Validate or save a structured diagnosis record.

**Request Body (JSON):**
```json
{
  "plant": "Tomato",
  "disease": "Early Blight",
  "healthStatus": "Diseased",
  "confidence": 78.4,
  "location": {
    "latitude": 28.6139,
    "longitude": 77.2090
  }
}
```

## Model Configuration
- **Hugging Face Model (Default)**: Khawajaa EfficientNet-B4 at `Khawajaa/plant-disease-detector`; override its repository using `PLANT_DISEASE_MODEL_NAME`. The reported confidence is the model's top softmax score, not a calibrated probability.
- **Local Keras Model**: Place `trained_plant_disease_model.keras` into `ml_service/model/` and set `PLANT_DISEASE_KERAS_MODEL_PATH=ml_service/model/trained_plant_disease_model.keras`.
- **Diagnostic Confidence Threshold**: Set via `DISEASE_CONFIDENCE_THRESHOLD` (default: `0.50`).
