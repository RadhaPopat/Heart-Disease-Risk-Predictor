from pathlib import Path

import joblib
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_PATH = BASE_DIR / "data" / "raw" / "processed.cleveland.data"

PROCESSED_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

COLUMN_NAMES = [
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
    "num"
]

def extract_data() -> pd.DataFrame:

    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=COLUMN_NAMES,
        na_values="?"
    )

    return df

def create_target(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["num"] = (df["num"] > 0).astype(int)

    return df

def filter_physiological_outliers(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df = df[
        (df["age"].between(18, 100)) &
        (df["trestbps"].between(70, 250)) &
        (df["chol"].between(80, 700)) &
        (df["thalach"].between(40, 250)) &
        (df["oldpeak"].between(0, 10))
    ]

    return df

def split_features_target(df: pd.DataFrame):

    X = df.drop(columns=["num"])
    y = df["num"]

    return X, y

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test

def run_etl():
    print("Starting ETL pipeline...")

    df = extract_data()

    print(f"Raw dataset shape: {df.shape}")

    df = create_target(df)

    df = filter_physiological_outliers(df)

    print(f"After filtering: {df.shape}")

    X, y = split_features_target(df)

    print("\nTarget distribution:")
    print(y.value_counts())

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    print(f"\nTraining shape: {X_train.shape}")
    print(f"Testing shape: {X_test.shape}")

    imputer = SimpleImputer(strategy="median")

    X_train_imputed = imputer.fit_transform(X_train)

    X_test_imputed = imputer.transform(X_test)

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train_imputed
    )

    X_test_scaled = scaler.transform(
        X_test_imputed
    )

    joblib.dump(
        imputer,
        MODELS_DIR / "imputer.joblib"
    )

    joblib.dump(
        scaler,
        MODELS_DIR / "scaler.joblib"
    )


    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X.columns,
        index=X_train.index
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X.columns,
        index=X_test.index
    )

    train_processed = X_train_scaled.copy()
    train_processed["target"] = y_train

    test_processed = X_test_scaled.copy()
    test_processed["target"] = y_test


    train_processed.to_csv(
        PROCESSED_DIR / "train.csv",
        index=False
    )

    test_processed.to_csv(
        PROCESSED_DIR / "test.csv",
        index=False
    )

    print("\nETL pipeline completed successfully.")

    print("\nGenerated files:")

    print(
        PROCESSED_DIR / "train.csv"
    )

    print(
        PROCESSED_DIR / "test.csv"
    )

    print(
        MODELS_DIR / "imputer.joblib"
    )

    print(
        MODELS_DIR / "scaler.joblib"
    )


if __name__ == "__main__":
    run_etl()