from contextlib import asynccontextmanager
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import shap

from fastapi import FastAPI

from app.schemas import PatientInput

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "models" / "best_calibrated_model.joblib"
BASE_MODEL_PATH = BASE_DIR / "models" / "best_base_tree_model.joblib"
IMPUTER_PATH = BASE_DIR / "models" / "imputer.joblib"
SCALER_PATH = BASE_DIR / "models" / "scaler.joblib"


FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


FEATURE_NAMES = {
    "age": "Age",
    "sex": "Sex",
    "cp": "Chest Pain Type",
    "trestbps": "Resting Blood Pressure",
    "chol": "Cholesterol",
    "fbs": "Fasting Blood Sugar",
    "restecg": "Resting ECG",
    "thalach": "Maximum Heart Rate",
    "exang": "Exercise-Induced Angina",
    "oldpeak": "ST Depression",
    "slope": "ST Segment Slope",
    "ca": "Major Vessels",
    "thal": "Thalassemia",
}


CATEGORICAL_VALUES = {
    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic",
    },
    "sex": {
        0: "Female",
        1: "Male",
    },
    "fbs": {
        0: "≤ 120 mg/dl",
        1: "> 120 mg/dl",
    },
    "exang": {
        0: "No",
        1: "Yes",
    },
    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy",
    },
}

model = None
base_model = None
imputer = None
scaler = None
explainer = None

@asynccontextmanager
async def lifespan(app: FastAPI):

    global model
    global base_model
    global imputer
    global scaler
    global explainer

    print("Loading ML artifacts...")

    model = joblib.load(MODEL_PATH)
    base_model = joblib.load(BASE_MODEL_PATH)
    imputer = joblib.load(IMPUTER_PATH)
    scaler = joblib.load(SCALER_PATH)

    explainer = shap.TreeExplainer(base_model)

    print("ML artifacts loaded successfully.")

    yield

    print("Shutting down API...")

app = FastAPI(
    title="CardioVascular Risk API",
    description="Heart disease risk prediction with calibrated probabilities and SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():

    return {
        "message": "CardioVascular Risk API",
        "status": "running",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "explainer_loaded": explainer is not None,
    }



@app.post("/predict")
def predict(patient: PatientInput):

    patient_data = patient.model_dump()

    X = pd.DataFrame(
        [patient_data],
        columns=FEATURES,
    )

    X_imputed = imputer.transform(X)

    X_scaled = scaler.transform(X_imputed)

    X_scaled = pd.DataFrame(
        X_scaled,
        columns=FEATURES,
    )

    probability = model.predict_proba(X_scaled)[0][1]

    prediction = int(probability >= 0.5)

    shap_values = explainer.shap_values(X_scaled)

    if isinstance(shap_values, list):

        values = shap_values[1][0]

    else:

        values = shap_values

        if len(values.shape) == 3:
            values = values[0, :, 1]

        elif len(values.shape) == 2:
            values = values[0]
    contributions = []

    for feature, shap_value in zip(FEATURES, values):

        contributions.append(
            {
                "feature": feature,
                "feature_name": FEATURE_NAMES[feature],
                "value": patient_data[feature],
                "shap_value": float(shap_value),
                "direction": (
                    "increases risk"
                    if shap_value > 0
                    else "decreases risk"
                ),
            }
        )

    contributions.sort(
        key=lambda x: abs(x["shap_value"]),
        reverse=True,
    )

    for item in contributions:

        feature = item["feature"]

        if feature in CATEGORICAL_VALUES:

            item["display_value"] = CATEGORICAL_VALUES[
                feature
            ].get(
                int(item["value"]),
                str(item["value"]),
            )

        else:

            item["display_value"] = item["value"]

    return {
        "prediction": prediction,
        "risk_probability": round(
            float(probability),
            4,
        ),
        "risk_percentage": round(
            float(probability) * 100,
            2,
        ),
        "risk_label": (
            "High Risk"
            if probability >= 0.5
            else "Lower Risk"
        ),
        "feature_contributions": contributions,
    }