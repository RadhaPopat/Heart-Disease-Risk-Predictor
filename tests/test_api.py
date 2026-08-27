import pytest
from fastapi.testclient import TestClient

from app.main import app


# ============================================================
# FastAPI Test Client
# ============================================================

@pytest.fixture
def client():
    """
    Creates a FastAPI TestClient and triggers the application's
    startup/lifespan events.

    This is important because app.main loads the ML artifacts
    (imputer, scaler, model, SHAP explainer) during startup.
    """
    with TestClient(app) as test_client:
        yield test_client


# ============================================================
# Sample Patient
# ============================================================

SAMPLE_PATIENT = {
    "age": 59,
    "sex": 1,
    "cp": 4,
    "trestbps": 138,
    "chol": 271,
    "fbs": 0,
    "restecg": 2,
    "thalach": 182,
    "exang": 0,
    "oldpeak": 0,
    "slope": 1,
    "ca": 0,
    "thal": 3
}


# ============================================================
# Test 1: Root Endpoint
# ============================================================

def test_root(client):

    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "CardioVascular Risk API"
    assert data["status"] == "running"


# ============================================================
# Test 2: Health Endpoint
# ============================================================

def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


# ============================================================
# Test 3: Valid Prediction
# ============================================================

def test_prediction(client):

    response = client.post(
        "/predict",
        json=SAMPLE_PATIENT
    )

    assert response.status_code == 200

    data = response.json()

    # --------------------------------------------------------
    # Check prediction
    # --------------------------------------------------------

    assert "prediction" in data

    assert data["prediction"] in [0, 1]

    # --------------------------------------------------------
    # Check probability
    # --------------------------------------------------------

    assert "risk_probability" in data

    assert 0 <= data["risk_probability"] <= 1

    # --------------------------------------------------------
    # Check percentage
    # --------------------------------------------------------

    assert "risk_percentage" in data

    assert 0 <= data["risk_percentage"] <= 100

    # --------------------------------------------------------
    # Check risk label
    # --------------------------------------------------------

    assert "risk_label" in data

    assert data["risk_label"] in [
        "Lower Risk",
        "Moderate Risk",
        "Higher Risk"
    ]


# ============================================================
# Test 4: Feature Contributions / SHAP
# ============================================================

def test_prediction_contains_feature_contributions(client):

    response = client.post(
        "/predict",
        json=SAMPLE_PATIENT
    )

    assert response.status_code == 200

    data = response.json()

    # SHAP explanations should be present
    assert "feature_contributions" in data

    contributions = data["feature_contributions"]

    # We have 13 input features
    assert len(contributions) == 13

    # --------------------------------------------------------
    # Check structure of each contribution
    # --------------------------------------------------------

    for contribution in contributions:

        assert "feature" in contribution

        assert "feature_name" in contribution

        assert "value" in contribution

        assert "shap_value" in contribution

        assert "direction" in contribution

        assert "display_value" in contribution

        assert contribution["direction"] in [
            "increases risk",
            "decreases risk"
        ]


# ============================================================
# Test 5: Invalid Blood Pressure
# ============================================================

def test_invalid_blood_pressure(client):

    invalid_patient = SAMPLE_PATIENT.copy()

    # Unrealistic blood pressure
    invalid_patient["trestbps"] = 500

    response = client.post(
        "/predict",
        json=invalid_patient
    )

    # Pydantic validation should reject the request
    assert response.status_code == 422


# ============================================================
# Test 6: Invalid Age
# ============================================================

def test_invalid_age(client):

    invalid_patient = SAMPLE_PATIENT.copy()

    # Outside allowed age range
    invalid_patient["age"] = 200

    response = client.post(
        "/predict",
        json=invalid_patient
    )

    assert response.status_code == 422


# ============================================================
# Test 7: Invalid Cholesterol
# ============================================================

def test_invalid_cholesterol(client):

    invalid_patient = SAMPLE_PATIENT.copy()

    # Unrealistic cholesterol value
    invalid_patient["chol"] = 1000

    response = client.post(
        "/predict",
        json=invalid_patient
    )

    assert response.status_code == 422


# ============================================================
# Test 8: Invalid Heart Rate
# ============================================================

def test_invalid_heart_rate(client):

    invalid_patient = SAMPLE_PATIENT.copy()

    # Unrealistic maximum heart rate
    invalid_patient["thalach"] = 500

    response = client.post(
        "/predict",
        json=invalid_patient
    )

    assert response.status_code == 422


# ============================================================
# Test 9: Invalid Chest Pain Type
# ============================================================

def test_invalid_chest_pain_type(client):

    invalid_patient = SAMPLE_PATIENT.copy()

    # UCI dataset allows cp values from 1 to 4
    invalid_patient["cp"] = 10

    response = client.post(
        "/predict",
        json=invalid_patient
    )

    assert response.status_code == 422


# ============================================================
# Test 10: Missing Required Feature
# ============================================================

def test_missing_required_feature(client):

    incomplete_patient = SAMPLE_PATIENT.copy()

    # Remove required feature
    del incomplete_patient["age"]

    response = client.post(
        "/predict",
        json=incomplete_patient
    )

    assert response.status_code == 422