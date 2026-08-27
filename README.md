# CardioVascular Risk Prediction System

An end-to-end machine learning system that predicts cardiovascular disease risk from patient clinical data and provides interpretable feature-level explanations using SHAP.

The project combines a machine learning pipeline, experiment tracking, a FastAPI REST API, a React frontend, explainable AI, and automated testing into a complete ML application.

> **Disclaimer:** This project is developed for educational and analytical purposes only. It is not a certified medical diagnostic system and should not be used for clinical decision-making.

---

## Overview

Early identification of cardiovascular risk can help identify patients who may require further medical evaluation.

This system uses the **Cleveland Heart Disease dataset** to train and evaluate multiple classification models. The selected model predicts the probability of cardiovascular disease and provides SHAP-based explanations showing which patient features contributed toward increasing or decreasing the predicted risk.

The project covers the complete machine learning workflow:

- Data preprocessing and ETL
- Feature transformation
- Model training and evaluation
- Experiment tracking with MLflow
- Random Forest model selection
- SHAP-based explainability
- FastAPI prediction service
- React frontend
- Automated API and data pipeline testing

---

## System Architecture

```text
                    Patient Clinical Data
                            │
                            ▼
                   ┌─────────────────┐
                   │   React Frontend│
                   │     (Vite)      │
                   └────────┬────────┘
                            │
                       HTTP POST
                            │
                            ▼
                   ┌─────────────────┐
                   │     FastAPI     │
                   │      API        │
                   └────────┬────────┘
                            │
                            ▼
                ┌──────────────────────┐
                │ Data Preprocessing   │
                │                      │
                │ • Imputation         │
                │ • Scaling            │
                └──────────┬───────────┘
                           │
                           ▼
                  ┌──────────────────┐
                  │  Random Forest   │
                  │  Classification  │
                  └────────┬─────────┘
                           │
                 ┌─────────┴─────────┐
                 ▼                   ▼
          Risk Probability          SHAP
                 │                   │
                 └─────────┬─────────┘
                           ▼
                  ┌──────────────────┐
                  │ Prediction +     │
                  │ Explanation      │
                  └────────┬─────────┘
                           │
                           ▼
                    React Frontend
```

---

## Dataset

The model uses the **Cleveland Heart Disease dataset** and processes 13 clinical features.

| Feature | Description | Values / Unit |
|---|---|---|
| `age` | Patient age | Years |
| `sex` | Biological sex | 0 = Female, 1 = Male |
| `cp` | Chest pain type | 1–4 |
| `trestbps` | Resting blood pressure | mmHg |
| `chol` | Serum cholesterol | mg/dl |
| `fbs` | Fasting blood sugar > 120 mg/dl | 0 = No, 1 = Yes |
| `restecg` | Resting ECG results | 0–2 |
| `thalach` | Maximum heart rate achieved | bpm |
| `exang` | Exercise-induced angina | 0 = No, 1 = Yes |
| `oldpeak` | ST depression induced by exercise | Numerical |
| `slope` | Peak exercise ST segment slope | 1–3 |
| `ca` | Number of major vessels | 0–3 |
| `thal` | Thalassemia status | 1–3 |

---

## Machine Learning Pipeline

The project follows a structured ML pipeline:

```text
Raw Dataset
     │
     ▼
Data Cleaning
     │
     ▼
Feature Processing
     │
     ├── Missing Value Imputation
     │
     └── Feature Scaling
     │
     ▼
Train / Test Split
     │
     ▼
Model Training
     │
     ├── Logistic Regression
     ├── Random Forest
     └── XGBoost
     │
     ▼
Model Evaluation
     │
     ▼
Random Forest Selection
     │
     ▼
Model Artifact
     │
     ▼
FastAPI Inference
     │
     ▼
SHAP Explanation
```

---

## Model Performance

Three classification models were evaluated using discrimination and probability calibration metrics.

Random Forest achieved the strongest overall performance and the lowest Brier Score, making it the selected model for probability estimation and API inference.

| Model | ROC-AUC | PR-AUC | Brier Score | Log Loss |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.9556 | 0.9341 | 0.0970 | 0.3359 |
| **Random Forest** | **0.9600** | **0.9478** | **0.0862** | **0.3079** |
| XGBoost | 0.9491 | 0.9103 | 0.0914 | 0.3298 |

### Selected Model

**Random Forest Classifier**

The Random Forest model was selected based primarily on its probability calibration performance, represented by the lowest Brier Score.

---

## Explainable AI with SHAP

The system uses **SHAP (SHapley Additive exPlanations)** to provide feature-level explanations for individual predictions.

Each prediction includes:

- Risk probability
- Risk percentage
- Risk classification
- Individual feature contributions
- Direction of contribution

### Interpretation

- **Positive SHAP value:** pushes the prediction toward higher cardiovascular risk.
- **Negative SHAP value:** pushes the prediction toward lower cardiovascular risk.
- Larger absolute SHAP values indicate a stronger contribution to the individual prediction.

### Example

```text
Risk Probability: 24.94%
Risk Classification: Lower Risk

Feature Contributions:

Major Vessels          : -0.1021
Thalassemia            : -0.0954
Chest Pain Type        : +0.0795
Age                    : +0.0577
Maximum Heart Rate     : -0.0471
ST Depression          : -0.0406
Exercise-Induced Angina: -0.0300
Sex                    : +0.0255
```

This allows the system to provide more insight than a simple binary prediction.

---

## Web Application

The project includes a React-based frontend that allows users to:

- Enter patient clinical information
- Submit data to the FastAPI backend
- View the predicted risk percentage
- View the risk classification
- View the model probability
- Inspect individual SHAP feature contributions

The frontend communicates with the backend through the `/predict` REST endpoint.

---

## Repository Structure

```text
CardioVascular-Risk/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── schemas.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.css
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── models/
│   ├── imputer.joblib
│   └── scaler.joblib
│
├── notebooks/
│
├── src/
│   ├── etl.py
│   ├── explain.py
│   └── train.py
│
├── tests/
│   ├── test_api.py
│   └── test_etl.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Local Setup

## Prerequisites

Make sure the following are installed:

- Python 3.9+
- Node.js 18+
- npm
- Git

---

## 1. Clone the Repository

```bash
git clone https://github.com/RADHAPOPAT/CardioVascular-Risk.git

cd CardioVascular-Risk
```

---

## 2. Create Python Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv

source venv/bin/activate
```

---

## 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Run the Backend API

From the project root:

```bash
uvicorn app.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### Swagger API Documentation

FastAPI automatically provides interactive API documentation at:

```text
http://127.0.0.1:8000/docs
```

You can use Swagger UI to test the API endpoints directly from the browser.

---

## 5. Run the React Frontend

Open a **new terminal**.

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend will normally be available at:

```text
http://localhost:5173
```

---

# API Endpoints

## Health Check

### `GET /health`

Checks whether the backend service is running.

### Response

```json
{
  "status": "healthy"
}
```

---

## Predict Cardiovascular Risk

### `POST /predict`

Accepts patient clinical information and returns the predicted cardiovascular risk along with SHAP feature contributions.

### Request

```json
{
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
```

### Response

```json
{
  "prediction": 0,
  "risk_probability": 0.2494,
  "risk_percentage": 24.94,
  "risk_label": "Lower Risk",
  "feature_contributions": [
    {
      "feature": "ca",
      "feature_name": "Major Vessels",
      "value": 0,
      "shap_value": -0.1021,
      "direction": "decreases risk",
      "display_value": 0
    }
  ]
}
```

---

# Testing

The project includes automated tests for both the API and data processing pipeline.

Run the complete test suite:

```bash
pytest -v
```

Current test coverage includes:

### API Tests

- Root endpoint
- Health endpoint
- Prediction endpoint
- SHAP feature contributions
- Invalid blood pressure
- Invalid age
- Invalid cholesterol
- Invalid heart rate
- Invalid chest pain type
- Missing required features

### ETL Tests

- Processed training dataset exists
- Processed testing dataset exists
- Training and testing datasets contain consistent features
- No infinite values are present

Example successful test run:

```text
14 passed
```

---

# Experiment Tracking with MLflow

MLflow is used to track machine learning experiments, including:

- Model parameters
- Evaluation metrics
- Training runs
- Model comparisons

Start the MLflow dashboard with:

```bash
mlflow ui
```

Then open:

```text
http://127.0.0.1:5000
```

---

# Tech Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP

### Backend

- FastAPI
- Pydantic
- Uvicorn

### Frontend

- React
- Vite
- JavaScript
- CSS

### MLOps & Testing

- MLflow
- Pytest
- Joblib
- Git/GitHub

---

# Key Features

- End-to-end machine learning pipeline
- Multiple model comparison
- Probability-based cardiovascular risk prediction
- Random Forest model selection
- SHAP explainability
- REST API using FastAPI
- Interactive React frontend
- Input validation using Pydantic
- Automated API testing
- Automated ETL testing
- MLflow experiment tracking
- Modular project structure

---

# Future Improvements

Potential improvements include:

- [ ] Add Docker support for frontend and backend
- [ ] Add GitHub Actions for CI/CD
- [ ] Deploy the FastAPI backend to a cloud platform
- [ ] Deploy the React frontend
- [ ] Add persistent storage for prediction history
- [ ] Add authentication and user management
- [ ] Add model monitoring
- [ ] Add automated model retraining
- [ ] Add more advanced visualization of SHAP explanations

---

# Author

**Radha Popat**

Computer Engineering Student  
Nirma University

- GitHub: https://github.com/RADHAPOPAT
- LinkedIn: https://linkedin.com/in/radha-popat-06954131a

---

## Disclaimer

This project is intended strictly for educational, research, and software engineering demonstration purposes.

The predictions generated by this system should **not** be interpreted as medical advice, diagnosis, or treatment recommendations. Any real-world medical decision should be made by a qualified healthcare professional.
