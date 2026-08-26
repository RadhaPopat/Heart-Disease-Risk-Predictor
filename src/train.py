from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss
)

from xgboost import XGBClassifier


# ============================================================
# Paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "train.csv"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test.csv"
)

MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# MLflow Configuration
# ============================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"

mlflow.set_tracking_uri(
    MLFLOW_TRACKING_URI
)

mlflow.set_experiment(
    "CardioVascular-Risk"
)


# ============================================================
# Load data
# ============================================================

def load_data():

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop(columns=["target"])
    y_train = train_df["target"]

    X_test = test_df.drop(columns=["target"])
    y_test = test_df["target"]

    return X_train, X_test, y_train, y_test


# ============================================================
# Define models
# ============================================================

def get_models():

    models = {

        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=3,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            random_state=42
        )
    }

    return models


# ============================================================
# Calibration
# ============================================================

def calibrate_model(model):

    calibrated_model = CalibratedClassifierCV(
        estimator=model,
        method="sigmoid",
        cv=5
    )

    return calibrated_model


# ============================================================
# Evaluate model
# ============================================================

def evaluate_model(model, X_test, y_test):

    probabilities = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(
        y_test,
        probabilities
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities
    )

    brier = brier_score_loss(
        y_test,
        probabilities
    )

    loss = log_loss(
        y_test,
        probabilities
    )

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": loss
    }


# ============================================================
# Main training pipeline
# ============================================================

def train_models():

    print("Loading processed data...")

    X_train, X_test, y_train, y_test = load_data()

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    models = get_models()

    results = []

    best_model = None
    best_model_name = None
    best_brier = float("inf")

    # ========================================================
    # Train each model
    # ========================================================

    for model_name, base_model in models.items():

        print("\n" + "=" * 60)
        print(f"Training: {model_name}")
        print("=" * 60)

        # ----------------------------------------------------
        # Calibration
        # ----------------------------------------------------

        calibrated_model = calibrate_model(
            base_model
        )

        # ----------------------------------------------------
        # MLflow run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name=f"{model_name}_calibrated"
        ):

            # Train
            calibrated_model.fit(
                X_train,
                y_train
            )

            # Evaluate
            metrics = evaluate_model(
                calibrated_model,
                X_test,
                y_test
            )

            # ------------------------------------------------
            # Print metrics
            # ------------------------------------------------

            print(
                f"ROC-AUC:    {metrics['roc_auc']:.4f}"
            )

            print(
                f"PR-AUC:     {metrics['pr_auc']:.4f}"
            )

            print(
                f"Brier:      {metrics['brier_score']:.4f}"
            )

            print(
                f"Log Loss:   {metrics['log_loss']:.4f}"
            )

            # ------------------------------------------------
            # MLflow parameters
            # ------------------------------------------------

            mlflow.log_param(
                "model",
                model_name
            )

            mlflow.log_param(
                "calibration_method",
                "sigmoid"
            )

            mlflow.log_param(
                "calibration_cv",
                5
            )

            # ------------------------------------------------
            # MLflow metrics
            # ------------------------------------------------

            mlflow.log_metric(
                "roc_auc",
                metrics["roc_auc"]
            )

            mlflow.log_metric(
                "pr_auc",
                metrics["pr_auc"]
            )

            mlflow.log_metric(
                "brier_score",
                metrics["brier_score"]
            )

            mlflow.log_metric(
                "log_loss",
                metrics["log_loss"]
            )

            # ------------------------------------------------
            # Log model
            # ------------------------------------------------

            mlflow.sklearn.log_model(
                calibrated_model,
                name="model",
                serialization_format="cloudpickle"
            )

            results.append(
                {
                    "model": model_name,
                    **metrics
                }
            )

        # ====================================================
        # Select best model
        # ====================================================

        if metrics["brier_score"] < best_brier:

            best_brier = metrics["brier_score"]

            best_model = calibrated_model

            best_model_name = model_name

    # ========================================================
    # Save best model
    # ========================================================

    best_model_path = (
        MODELS_DIR
        / "best_calibrated_model.joblib"
    )

    joblib.dump(
        best_model,
        best_model_path
    )

    # ========================================================
    # Results
    # ========================================================

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    print(
        results_df.to_string(index=False)
    )

    print("\nBest calibrated model:")
    print(best_model_name)

    print(
        f"Brier Score: {best_brier:.4f}"
    )

    print(
        f"\nSaved model to:\n{best_model_path}"
    )


if __name__ == "__main__":
    train_models()