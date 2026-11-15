
from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel


# --------------------------------------------------
# 1. Load trained ML pipeline
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = BASE_DIR / "breast_cancer_pipeline.pkl"
DATASET_PATH = BASE_DIR.parent / "breast-cancer.csv"

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

pipeline = joblib.load(MODEL_PATH)

# Get feature names from the trained model
FEATURE_NAMES = list(pipeline.feature_names_in_)

# Ensure ID is not part of model input
if "id" in FEATURE_NAMES or "diagnosis" in FEATURE_NAMES:
    raise ValueError(
        "Invalid model features. The model must be trained "
        "without 'id' and 'diagnosis'. Retrain the pipeline."
    )

print(f"Model loaded successfully: {MODEL_PATH}")
print(f"Total expected features: {len(FEATURE_NAMES)}")


# --------------------------------------------------
# 2. Initialize FastAPI
# --------------------------------------------------

app = FastAPI(
    title="TumorAI",
    description="Educational breast cancer prediction API.",
    version="1.0.0"
)


# --------------------------------------------------
# 3. Configure CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# 4. Request schema
# --------------------------------------------------

class PredictionInput(BaseModel):
    features: dict[str, float]


# --------------------------------------------------
# 5. Basic routes
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "message": "TumorAI API is running!"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
        "total_features": len(FEATURE_NAMES)
    }


@app.get("/features")
def get_features():
    return {
        "features": FEATURE_NAMES
    }


# --------------------------------------------------
# 6. Prediction endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):

    input_features = data.features

    # Validate incoming feature names
    expected_features = set(FEATURE_NAMES)
    received_features = set(input_features.keys())

    missing_features = expected_features - received_features
    unexpected_features = received_features - expected_features

    if missing_features or unexpected_features:
        raise HTTPException(
            status_code=422,
            detail={
                "missing_features": sorted(missing_features),
                "unexpected_features": sorted(unexpected_features)
            }
        )

    # Arrange input according to training feature order
    input_df = pd.DataFrame(
        [[input_features[feature] for feature in FEATURE_NAMES]],
        columns=FEATURE_NAMES
    )

    # Predict class
    prediction = int(pipeline.predict(input_df)[0])

    # Get class probabilities
    probabilities = pipeline.predict_proba(input_df)[0]

    # Find probability of malignant class (1)
    classes = list(pipeline.classes_)
    malignant_index = classes.index(1)

    malignant_probability = float(
        probabilities[malignant_index] * 100
    )

    result = "Malignant" if prediction == 1 else "Benign"

    return {
        "prediction": prediction,
        "result": result,
        "malignant_probability": round(malignant_probability, 2),
        "disclaimer": (
            "Educational prediction only. "
            "This is not a medical diagnosis."
        )
    }


# --------------------------------------------------
# 7. Demo endpoint
# --------------------------------------------------

@app.get("/demo")
def demo():

    if not DATASET_PATH.exists():
        raise HTTPException(
            status_code=500,
            detail="Demo dataset not found."
        )

    df = pd.read_csv(DATASET_PATH)

    # Ensure dataset contains all required features
    missing_columns = [
        feature for feature in FEATURE_NAMES
        if feature not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Dataset is missing required columns.",
                "missing_columns": missing_columns
            }
        )

    # Select one real sample
    sample = df.sample(n=1).iloc[0]

    demo_features = {
        feature: float(sample[feature])
        for feature in FEATURE_NAMES
    }

    return {
        "features": demo_features
    }
