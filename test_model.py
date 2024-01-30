import pandas as pd
import pytest
import numpy as np
import pickle
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from ml.model import train_model, compute_model_metrics, inference, process_data

@pytest.fixture
def data():
    """Function to generate fake Pandas data."""
    df = pd.DataFrame(
        {
            "id": [1, 2, 3], 
            "numeric_feat": [3.14, 2.72, 1.62], 
            "categorical_feat": ["dots", "one_colour", "one_colour"],
            "target_feature": ["dog", "dog", "cat"],
        }
    )
    return df

def test_type_process_data(data):
    X, y, encoder, lb = process_data(
        data, categorical_features=["categorical_feat"], label="target_feature", training=True, encoder=None, lb=None
    )
    X_2, y_2, encoder_2, lb_2 = process_data(
        data, categorical_features=["categorical_feat"], label="target_feature", training=False, encoder=encoder, lb=lb
    )

    assert type(X) == np.ndarray
    assert type(y) == np.ndarray
    assert type(encoder) == OneHotEncoder
    assert type(lb) == LabelBinarizer
    assert type(X_2) == np.ndarray
    assert type(y_2) == np.ndarray
    assert type(encoder_2) == OneHotEncoder
    assert type(lb_2) == LabelBinarizer

def test_train_model(data):
    X, y, _, _ = process_data(
        data, categorical_features=["categorical_feat"], label="target_feature", training=True, encoder=None, lb=None
    )

    model = train_model(X, y)
    assert type(model) == LogisticRegression

def test_inference(data):
    X, y, encoder, lb = process_data(
        data, categorical_features=["categorical_feat"], label="target_feature", training=True, encoder=None, lb=None
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert type(preds) == np.ndarray

def test_compute_model_metrics(data):
    X, y, encoder, lb = process_data(
        data, categorical_features=["categorical_feat"], label="target_feature", training=True, encoder=None, lb=None
    )
    model = train_model(X, y)
    preds = inference(model, X)

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert type(precision) == np.float64
    assert type(recall) == np.float64
    assert type(fbeta) == np.float64
