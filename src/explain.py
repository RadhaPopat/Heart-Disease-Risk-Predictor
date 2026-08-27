from pathlib import Path

import joblib
import pandas as pd
import shap

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
    "thal": "Thalassemia"
}

CATEGORICAL_VALUES = {
    "cp": {
        1: "Typical Angina",
        2: "Atypical Angina",
        3: "Non-anginal Pain",
        4: "Asymptomatic"
    },

    "sex": {
        0: "Female",
        1: "Male"
    },

    "fbs": {
        0: "≤ 120 mg/dl",
        1: "> 120 mg/dl"
    },

    "exang": {
        0: "No",
        1: "Yes"
    },

    "restecg": {
        0: "Normal",
        1: "ST-T Wave Abnormality",
        2: "Left Ventricular Hypertrophy"
    }
}

BASE_DIR = Path(__file__).resolve().parent.parent

def format_feature_value(feature, value):

    if feature in CATEGORICAL_VALUES:

        mapping = CATEGORICAL_VALUES[feature]

        return mapping.get(
            int(value),
            str(value)
        )

    return value

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "best_base_tree_model.joblib"
)

SCALER_PATH = (
    BASE_DIR
    / "models"
    / "scaler.joblib"
)

TEST_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "test.csv"
)


def load_resources():

    model = joblib.load(MODEL_PATH)

    scaler = joblib.load(SCALER_PATH)

    test_df = pd.read_csv(TEST_PATH)

    X_test_scaled = test_df.drop(
        columns=["target"]
    )

    y_test = test_df["target"]

    X_test_original = pd.DataFrame(
        scaler.inverse_transform(X_test_scaled),
        columns=X_test_scaled.columns,
        index=X_test_scaled.index
    )

    return (
        model,
        X_test_scaled,
        X_test_original,
        y_test
    )

def create_explainer(model):

    explainer = shap.TreeExplainer(model)

    return explainer

def explain_patient(
    explainer,
    X,
    patient_index=0
):

    patient = X.iloc[[patient_index]]

    shap_values = explainer.shap_values(patient)

    return patient, shap_values

def get_feature_contributions(
    explainer,
    patient
):
    """
    Extract SHAP contributions for the heart-disease class.

    Handles SHAP output formats used by different SHAP versions.
    """

    shap_values = explainer.shap_values(patient)

    if isinstance(shap_values, list):

        values = shap_values[1][0]

        base_value = explainer.expected_value[1]

    else:

        shap_array = shap_values

        if shap_array.ndim == 3:

            # First patient, class 1
            values = shap_array[0, :, 1]

        elif shap_array.ndim == 2:

            # (samples, features)
            values = shap_array[0]

        else:

            raise ValueError(
                f"Unexpected SHAP array shape: "
                f"{shap_array.shape}"
            )

        expected_value = explainer.expected_value

        if hasattr(expected_value, "ndim"):

            if expected_value.ndim == 2:
                base_value = expected_value[0, 1]

            elif expected_value.ndim == 1:
                base_value = expected_value[1]

            else:
                base_value = expected_value.item()

        else:

            try:
                base_value = expected_value[1]
            except (IndexError, TypeError):
                base_value = expected_value

    values = values.reshape(-1)

    if len(values) != len(patient.columns):

        raise ValueError(
            f"SHAP values contain {len(values)} values, "
            f"but the patient has {len(patient.columns)} features."
        )


    contributions = pd.DataFrame(
        {
            "feature": patient.columns,
            "scaled_value": patient.iloc[0].values,
            "shap_value": values
        }
    )

    contributions["direction"] = (
        contributions["shap_value"]
        .apply(
            lambda x:
            "increases risk"
            if x > 0
            else "decreases risk"
        )
    )

    contributions["absolute_shap"] = (
        contributions["shap_value"].abs()
    )

    contributions = contributions.sort_values(
        "absolute_shap",
        ascending=False
    )

    contributions["feature_name"] = (
        contributions["feature"].map(FEATURE_NAMES)
    )

    return contributions, base_value

def main():

    print("=" * 60)
    print("SHAP MODEL EXPLAINABILITY")
    print("=" * 60)

    (
        model,
        X_test,
        X_test_original,
        y_test
    ) = load_resources()

    print(
        f"\nLoaded model: {type(model).__name__}"
    )

    print(
        f"Test samples: {len(X_test)}"
    )

    explainer = create_explainer(model)

    print(
        "\nTreeExplainer initialized successfully."
    )

    patient, shap_values = explain_patient(
        explainer,
        X_test,
        patient_index=0
    )

    contributions, base_value = (
        get_feature_contributions(
            explainer,
            patient
        )
    )

    original_patient = X_test_original.iloc[[0]]

    original_values = original_patient.iloc[0].to_dict()

    contributions["value"] = contributions["feature"].map(
        original_values
    )


    contributions["display_value"] = (
        contributions.apply(
            lambda row: format_feature_value(
                row["feature"],
                row["value"]
            ),
            axis=1
        )
    )


    print("\n")
    print("=" * 60)
    print("PATIENT EXPLANATION")
    print("=" * 60)

    print("\nPatient features (original clinical values):")

    print(
        original_patient.to_string(
            index=False
        )
    )

    print("\nFeature contributions:")

    print(
        contributions[
            [
                "feature_name",
                "display_value",
                "shap_value",
                "direction"
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSHAP base value: {base_value}"
    )


if __name__ == "__main__":
    main()