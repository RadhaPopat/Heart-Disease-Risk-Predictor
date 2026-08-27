import pandas as pd
import numpy as np


def test_processed_train_data_exists():

    train_path = "data/processed/train.csv"

    df = pd.read_csv(train_path)

    # We expect 242 training samples from our 80/20 split
    assert len(df) == 242

    # 13 input features
    assert df.shape[1] == 14


def test_processed_test_data_exists():

    test_path = "data/processed/test.csv"

    df = pd.read_csv(test_path)

    # We expect 61 testing samples
    assert len(df) == 61

    # 13 input features
    assert df.shape[1] == 14


def test_train_and_test_have_same_features():

    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    assert list(train.columns) == list(test.columns)


def test_no_infinite_values():

    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    assert not np.isinf(train.select_dtypes(include=np.number)).any().any()
    assert not np.isinf(test.select_dtypes(include=np.number)).any().any()